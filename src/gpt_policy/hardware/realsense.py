"""RGB capture from named RealSense serials; one owner per camera."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from .camera import CapturedImage
from ..recording.jpeg import encode_jpeg


class RealSenseCameraSet:
    def __init__(self, specs, width=640, height=480, fps=30):
        import pyrealsense2 as rs

        self.rs = rs
        self.cameras = []
        try:
            for name, serial in specs:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
                pipeline.start(config)
                self.cameras.append(SimpleNamespace(
                    name=name, serial=serial, width=width, height=height, pipeline=pipeline,
                ))
        except BaseException:
            self.close()
            raise

    def capture(self, stop=None):
        images = {}
        for camera in self.cameras:
            deadline = time.monotonic() + 3.0
            while True:
                if stop is not None and stop.is_set():
                    raise InterruptedError("RealSense capture stopped")
                ok, frames = camera.pipeline.try_wait_for_frames(200)
                if ok and frames.get_color_frame():
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"No RGB frame from {camera.name} ({camera.serial})")
            frame = frames.get_color_frame()
            rgb = np.asanyarray(frame.get_data()).copy()
            images[camera.name] = CapturedImage(
                camera.name, encode_jpeg(rgb, 85), "image/jpeg", camera.width, camera.height,
                time.time(), rgb.tobytes(), frame.get_timestamp() / 1000,
                "realsense_" + str(frame.get_frame_timestamp_domain()),
            )
        return images

    def describe(self, images=None):
        return [{
            "name": c.name, "serial": c.serial, "device": c.serial, "format": "RGB8",
            "width": c.width, "height": c.height,
            **({"captured_at": str(images[c.name].captured_at)} if images and c.name in images else {}),
        } for c in self.cameras]

    def close(self):
        errors = []
        for camera in self.cameras:
            try:
                camera.pipeline.stop()
            except Exception as exc:
                errors.append(str(exc))
        self.cameras.clear()
        if errors:
            raise RuntimeError(f"RealSense cleanup: {errors}")
