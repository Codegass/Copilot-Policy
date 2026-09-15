"""Apply and verify manual D405 exposure on an already running camera stream.

D405 RGB comes from its Stereo Module. Exposure uses the depth UVC extension
unit, not the standard V4L2 exposure control. Selectors follow librealsense's
src/ds/ds-private.h: unit 3, exposure 3 (uint32), auto exposure 11 (uint8).
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
from pathlib import Path
import struct
import time


class _Query(ctypes.Structure):
    _fields_ = [
        ("unit", ctypes.c_uint8),
        ("selector", ctypes.c_uint8),
        ("query", ctypes.c_uint8),
        ("size", ctypes.c_uint16),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]


_XU_QUERY = 0xC0000000 | (ctypes.sizeof(_Query) << 16) | (ord("u") << 8) | 0x21
_GAIN = 0x00980913


def _xu(fd: int, selector: int, request: int, size: int, value: int = 0) -> int:
    data = (ctypes.c_uint8 * size).from_buffer_copy(value.to_bytes(size, "little"))
    query = _Query(3, selector, request, size, data)
    # D405 applies exposure at a frame boundary. Until then GET_CUR can return
    # EBUSY, even with no competing owner. A running stream is required.
    for attempt in range(21):
        try:
            fcntl.ioctl(fd, _XU_QUERY, query, True)
            break
        except OSError as exc:
            if exc.errno != errno.EBUSY or attempt == 20:
                raise
            time.sleep(0.05)
    return int.from_bytes(bytes(data), "little")


def _gain(fd: int, value: int | None = None) -> int:
    control = bytearray(struct.pack("<Ii", _GAIN, value or 0))
    fcntl.ioctl(fd, 0xC008561B if value is None else 0xC008561C, control, True)
    return struct.unpack("<Ii", control)[1]


def _require_d405(path: str) -> None:
    node = Path("/sys/class/video4linux") / Path(path).resolve().name
    for parent in node.resolve().parents:
        if (parent / "idVendor").is_file():
            identity = tuple((parent / name).read_text().strip().lower()
                             for name in ("idVendor", "idProduct"))
            if identity == ("8086", "0b5b"):
                return
            break
    raise ValueError(f"D405 camera_controls cannot be applied to {path}")


def _read(fd: int) -> dict[str, int]:
    return {"auto_exposure": _xu(fd, 11, 0x81, 1),
            "exposure_us": _xu(fd, 3, 0x81, 4), "gain": _gain(fd)}


def configure_d405_cameras(specs, *, exposure_us: int, gain: int) -> list[dict]:
    """Set fixed exposure/gain before warmup; fail before robot initialization.

    The caller must keep these cameras streaming so pending controls can apply.
    Return before/after values for the run log. This helper does not start/stop
    streams, change white balance, or reset a device.
    """
    for name, value in (("exposure_us", exposure_us), ("gain", gain)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"camera_controls.{name} must be a positive integer")
    specs = list(specs)
    if not specs:
        raise ValueError("camera_controls requires explicitly named D405 devices")
    for _, path in specs:
        _require_d405(path)
    target = {"auto_exposure": 0, "exposure_us": exposure_us, "gain": gain}
    results = []
    for name, path in specs:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        try:
            low, high = (_xu(fd, 3, request, 4) for request in (0x82, 0x83))
            if not low <= exposure_us <= high:
                raise ValueError(f"{name}: exposure_us must be within {low}..{high}")
            query = bytearray(68)
            struct.pack_into("<I", query, 0, _GAIN)
            fcntl.ioctl(fd, 0xC0445624, query, True)
            low, high, step = struct.unpack_from("<iii", query, 40)
            if not low <= gain <= high or (gain - low) % max(step, 1):
                raise ValueError(f"{name}: gain must be within {low}..{high}, step {step}")
            before = _read(fd)
            if before["auto_exposure"]:
                _xu(fd, 11, 0x01, 1, 0)
            if before["exposure_us"] != exposure_us or before["auto_exposure"]:
                _xu(fd, 3, 0x01, 4, exposure_us)
            if before["gain"] != gain or before["auto_exposure"]:
                _gain(fd, gain)
            after = _read(fd)
            if after != target:
                raise RuntimeError(f"{name}: D405 controls did not apply: expected {target}, got {after}")
            results.append({"name": name, "device": path, "before": before, "after": after})
        finally:
            os.close(fd)
    return results
