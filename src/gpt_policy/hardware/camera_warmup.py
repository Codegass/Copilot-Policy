"""Discard camera startup transients before recording or robot initialization."""

from __future__ import annotations

from collections import deque
from io import BytesIO
import math
import threading
from time import monotonic

import numpy as np
from PIL import Image

from .camera import CapturedImage


def warm_up_cameras(cameras, *, minimum_s=3.0, stable_s=1.0, timeout_s=15.0) -> dict:
    """Wait for every camera's brightness and color balance to stop drifting.

    Keep the same streams open afterwards: reopening them restarts auto white
    balance. Only temporal changes are checked; a colored scene need not be gray.
    """
    if not all(math.isfinite(v) for v in (minimum_s, stable_s, timeout_s)) or not (
        0 < stable_s <= minimum_s < timeout_s
    ):
        raise ValueError("camera warmup requires 0 < stable_s <= minimum_s < timeout_s")
    names = [camera.name for camera in cameras.cameras]
    if not names or len(set(names)) != len(names):
        raise ValueError("camera warmup requires distinct named cameras")
    history = deque()
    started = monotonic()
    discarded = 0
    pending = names
    stop = threading.Event()
    timer = threading.Timer(timeout_s, stop.set)
    timer.daemon = True
    timer.start()
    try:
        while monotonic() - started < timeout_s and not stop.is_set():
            # The deadline also interrupts a V4L2 camera that never produces a
            # frame. Ctrl+C propagates to main's normal initialization cleanup.
            images = cameras.capture(stop=stop)
            if set(images) != set(names):
                raise RuntimeError("camera warmup received an incomplete camera snapshot")
            levels = np.stack([_frame_levels(images[name]) for name in names])
            now = monotonic()
            discarded += 1
            history.append((now, levels))
            # Retain one sample at/before the cutoff to cover a full second,
            # rather than checking only adjacent frames (which misses slow drift).
            while len(history) > 2 and history[1][0] <= now - stable_s:
                history.popleft()
            if len(history) < 3 or now - history[0][0] < stable_s:
                continue
            spread = np.ptp(np.stack([entry[1] for entry in history]), axis=0)
            pending = [name for name, delta in zip(names, spread) if np.any(delta > 0.02)]
            if not pending and minimum_s <= now - started < timeout_s and not stop.is_set():
                return {
                    "elapsed_s": round(now - started, 3),
                    "discarded_frames_per_camera": discarded,
                    "cameras": names,
                    "minimum_s": minimum_s,
                    "stable_s": stable_s,
                }
    except InterruptedError:
        if not stop.is_set():
            raise
    finally:
        timer.cancel()
        timer.join()
    raise TimeoutError(f"相机预热超时（{timeout_s:g} 秒），画面尚未稳定或未出帧：{', '.join(pending or names)}")


def _frame_levels(frame: CapturedImage) -> np.ndarray:
    if frame.rgb_data is not None:
        rgb = np.frombuffer(frame.rgb_data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    else:
        with Image.open(BytesIO(frame.data)) as image:
            rgb = np.asarray(image.convert("RGB"))
    sample = rgb[::max(1, frame.height // 48), ::max(1, frame.width // 64)]
    mean = sample.mean(axis=(0, 1))
    # RGB levels detect exposure changes; channel proportions also catch color
    # shifts in dim scenes. The floor avoids amplifying near-black sensor noise.
    return np.concatenate((mean / 255.0, mean / max(float(mean.sum()), 30.0)))
