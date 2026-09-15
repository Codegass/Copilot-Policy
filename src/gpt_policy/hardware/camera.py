"""Direct Linux camera capture for the Codex observation loop.

This module talks to V4L2 directly. It does not use a camera service or
OpenCV; all implementation code stays in this repository.
"""

from __future__ import annotations

import base64
import ctypes
import errno
import fcntl
import mmap
import os
import re
import select
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


_BUF_TYPE = 1
_MEMORY_MMAP = 1
_FIELD_NONE = 1
_FORMAT_SIZE = 208
_ENUM_FMT = 0xC0405602
_SET_FMT = 0xC0D05605
_REQBUFS = 0xC0145608
_QUERYBUF = 0xC0585609
_QBUF = 0xC058560F
_DQBUF = 0xC0585611
_STREAMON = 0x40045612
_STREAMOFF = 0x40045613
_BUF_FLAG_ERROR = 0x0040


def _fourcc(value: bytes) -> int:
    return int.from_bytes(value, byteorder="little")


_YUYV = _fourcc(b"YUYV")
_UYVY = _fourcc(b"UYVY")
_RGB3 = _fourcc(b"RGB3")
_GREY = _fourcc(b"GREY")
_MJPG = _fourcc(b"MJPG")
_Z16 = _fourcc(b"Z16 ")
_PREFERRED_FORMATS = (_YUYV, _RGB3, _MJPG, _UYVY, _Z16, _GREY)
_FORMAT_NAMES = {
    _Z16: "Z16 depth",
    _YUYV: "YUYV",
    _UYVY: "UYVY",
    _RGB3: "RGB3",
    _GREY: "GREY",
    _MJPG: "MJPG",
}


class _Timeval(ctypes.Structure):
    _fields_ = [("seconds", ctypes.c_long), ("microseconds", ctypes.c_long)]


class _Timecode(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 16)]


class _BufferMemory(ctypes.Union):
    _fields_ = [
        ("offset", ctypes.c_uint32),
        ("userptr", ctypes.c_ulong),
        ("planes", ctypes.c_void_p),
        ("fd", ctypes.c_uint64),
    ]


class _V4L2Buffer(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", _Timeval),
        ("timecode", _Timecode),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("memory_union", _BufferMemory),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _FormatDescription(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("description", ctypes.c_char * 32),
        ("pixelformat", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


@dataclass(frozen=True)
class CapturedImage:
    """One image in the format accepted by Codex app-server."""

    name: str
    data: bytes
    mime_type: str
    width: int
    height: int
    captured_at: float
    rgb_data: bytes | None = None
    source_timestamp_s: float | None = None
    source_clock: str | None = None

    def data_url(self) -> str:
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


class IncompleteFrameError(ValueError):
    """A V4L2 buffer does not contain one complete uncompressed frame."""


class V4L2Camera:
    """A small single-plane V4L2 mmap streamer."""

    def __init__(
        self,
        path: str,
        name: str | None = None,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.path = path
        self.name = name or Path(path).name
        self.fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        self._maps: list[mmap.mmap] = []
        self.dropped_frames = 0
        try:
            self.width, self.height, self.pixelformat, self.bytesperline = (
                self._configure(width, height)
            )
            self._start()
        except Exception:
            self.close()
            raise

    @property
    def format_name(self) -> str:
        return _FORMAT_NAMES.get(self.pixelformat, f"0x{self.pixelformat:08x}")

    def capture(self, stop: threading.Event | None = None) -> CapturedImage:
        # Codex and the robot can spend many seconds between observations.
        # During that time every mmap buffer fills, so a single DQBUF returns
        # a frame from before the previous action. Drain that backlog first.
        self._discard_queued_frames()
        while True:
            if stop is not None and stop.is_set():
                raise InterruptedError("相机采集已停止")
            readable, _, _ = select.select(
                [self.fd], [], [], 0.2 if stop is not None else None
            )
            if not readable:
                continue
            buffer = _V4L2Buffer()
            buffer.type = _BUF_TYPE
            buffer.memory = _MEMORY_MMAP
            try:
                fcntl.ioctl(self.fd, _DQBUF, buffer, True)
            except OSError as exc:
                if exc.errno in {errno.EAGAIN, errno.EINTR}:
                    continue
                raise RuntimeError(f"读取相机 {self.path} 失败: {exc}") from exc
            try:
                if buffer.flags & _BUF_FLAG_ERROR:
                    self.dropped_frames += 1
                    continue
                payload = bytes(self._maps[buffer.index][: buffer.bytesused])
                try:
                    data, mime, rgb_data = self._encode(payload)
                except IncompleteFrameError:
                    self.dropped_frames += 1
                    continue
            finally:
                fcntl.ioctl(self.fd, _QBUF, buffer, True)
            return CapturedImage(
                self.name, data, mime, self.width, self.height, time.time(), rgb_data,
                buffer.timestamp.seconds + buffer.timestamp.microseconds / 1e6,
                "v4l2_monotonic" if buffer.flags & 0x2000 else "v4l2_unknown",
            )

    def _discard_queued_frames(self) -> None:
        """Drain completed mmap buffers, then requeue them for a fresh frame."""
        drained: list[_V4L2Buffer] = []
        try:
            # A buffer cannot be dequeued twice until it is requeued, so this
            # is naturally bounded by the configured mmap buffer count.
            while len(drained) < len(self._maps):
                buffer = _V4L2Buffer()
                buffer.type = _BUF_TYPE
                buffer.memory = _MEMORY_MMAP
                try:
                    fcntl.ioctl(self.fd, _DQBUF, buffer, True)
                except OSError as exc:
                    if exc.errno == errno.EAGAIN:
                        break
                    if exc.errno == errno.EINTR:
                        continue
                    raise RuntimeError(f"清空相机 {self.path} 旧帧失败: {exc}") from exc
                drained.append(buffer)
        finally:
            for buffer in drained:
                fcntl.ioctl(self.fd, _QBUF, buffer, True)

    def close(self) -> None:
        if getattr(self, "fd", None) is None:
            return
        try:
            fcntl.ioctl(self.fd, _STREAMOFF, ctypes.c_int(_BUF_TYPE))
        except OSError:
            pass
        for mapped in self._maps:
            mapped.close()
        self._maps.clear()
        os.close(self.fd)
        self.fd = None  # type: ignore[assignment]

    def _configure(self, width: int, height: int) -> tuple[int, int, int, int]:
        available = self._formats()
        selected = next((code for code in _PREFERRED_FORMATS if code in available), None)
        if selected is None:
            names = ", ".join(_FORMAT_NAMES.get(code, hex(code)) for code in available)
            raise RuntimeError(f"相机 {self.path} 没有可用的图像格式: {names or 'none'}")
        fmt = bytearray(_FORMAT_SIZE)
        struct.pack_into("<I", fmt, 0, _BUF_TYPE)
        struct.pack_into("<IIIIII", fmt, 8, width, height, selected, _FIELD_NONE, 0, 0)
        fcntl.ioctl(self.fd, _SET_FMT, fmt, True)
        actual_width, actual_height, actual_format, _, stride, _ = struct.unpack_from(
            "<IIIIII", fmt, 8
        )
        if actual_format not in _PREFERRED_FORMATS:
            raise RuntimeError(f"相机 {self.path} 拒绝了图像格式")
        request = bytearray(20)
        struct.pack_into("<IIII", request, 0, 4, _BUF_TYPE, _MEMORY_MMAP, 0)
        fcntl.ioctl(self.fd, _REQBUFS, request, True)
        count = struct.unpack_from("<I", request, 0)[0]
        if count == 0:
            raise RuntimeError(f"相机 {self.path} 没有可用的采集缓冲区")
        for index in range(count):
            buffer = _V4L2Buffer()
            buffer.index = index
            buffer.type = _BUF_TYPE
            buffer.memory = _MEMORY_MMAP
            fcntl.ioctl(self.fd, _QUERYBUF, buffer, True)
            mapped = mmap.mmap(
                self.fd,
                buffer.length,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=buffer.memory_union.offset,
            )
            self._maps.append(mapped)
            fcntl.ioctl(self.fd, _QBUF, buffer, True)
        return actual_width, actual_height, actual_format, stride

    def _formats(self) -> set[int]:
        formats: set[int] = set()
        for index in range(64):
            description = _FormatDescription(index=index, type=_BUF_TYPE)
            try:
                fcntl.ioctl(self.fd, _ENUM_FMT, description, True)
            except OSError:
                break
            formats.add(description.pixelformat)
        return formats

    def _start(self) -> None:
        fcntl.ioctl(self.fd, _STREAMON, ctypes.c_int(_BUF_TYPE))

    def _encode(self, payload: bytes) -> tuple[bytes, str, bytes | None]:
        if self.pixelformat == _MJPG:
            return payload, "image/jpeg", None
        rgb = self._to_rgb(payload)
        return _encode_png(rgb), "image/png", rgb.tobytes()

    def _to_rgb(self, payload: bytes) -> np.ndarray:
        if self.pixelformat == _RGB3:
            row_bytes = self.width * 3
        else:
            row_bytes = self.width * (1 if self.pixelformat == _GREY else 2)
        raw = np.frombuffer(payload, dtype=np.uint8)
        stride = max(self.bytesperline, row_bytes)
        expected = self.height * stride
        if raw.size < expected:
            raise IncompleteFrameError(
                f"incomplete {self.format_name} frame: got {raw.size} bytes, "
                f"need {expected}"
            )
        raw = raw[:expected].reshape(self.height, stride)[:, :row_bytes]
        if self.pixelformat == _RGB3:
            return raw.reshape(self.height, self.width, 3)
        if self.pixelformat == _GREY:
            return np.repeat(raw[:, :, None], 3, axis=2)
        if self.pixelformat == _Z16:
            depth = raw.view("<u2").reshape(self.height, self.width)
            valid = depth[depth > 0]
            if valid.size == 0:
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            low, high = np.percentile(valid, (2, 98))
            if high <= low:
                scaled = np.where(depth > 0, 255, 0)
            else:
                scaled = np.clip((depth - low) * 255.0 / (high - low), 0, 255)
                scaled[depth == 0] = 0
            gray = scaled.astype(np.uint8)
            return np.repeat(gray[:, :, None], 3, axis=2)
        pairs = raw.reshape(self.height, self.width // 2, 4).astype(np.float32)
        if self.pixelformat == _YUYV:
            y0, u, y1, v = (pairs[:, :, i] for i in range(4))
        else:
            u, y0, v, y1 = (pairs[:, :, i] for i in range(4))
        y = np.empty((self.height, self.width), dtype=np.float32)
        y[:, 0::2], y[:, 1::2] = y0, y1
        u = np.repeat(u, 2, axis=1)
        v = np.repeat(v, 2, axis=1)
        # UVC YUYV uses studio-range BT.601. Expanding Y=16..235 to RGB
        # 0..255 restores the contrast seen through the normal D405 pipeline.
        luminance = 1.164383 * (y - 16)
        rgb = np.stack(
            (
                luminance + 1.596027 * (v - 128),
                luminance - 0.391762 * (u - 128) - 0.812968 * (v - 128),
                luminance + 2.017232 * (u - 128),
            ),
            axis=2,
        )
        return np.clip(rgb, 0, 255).astype(np.uint8)


class CameraSet:
    """Open one or more cameras and capture one synchronized-enough snapshot."""

    def __init__(self, specs: Iterable[tuple[str, str]], width: int = 640, height: int = 480):
        parsed = list(specs)
        if not parsed:
            parsed = [("camera", discover_camera(width, height))]
        self.cameras = []
        try:
            for name, path in parsed:
                self.cameras.append(V4L2Camera(path, name, width, height))
        except BaseException:
            self.close()
            raise

    def capture(self, stop: threading.Event | None = None) -> dict[str, CapturedImage]:
        return {camera.name: camera.capture(stop=stop) for camera in self.cameras}

    def describe(self, images: dict[str, CapturedImage] | None = None) -> list[dict[str, str | int]]:
        return [
            {
                "name": camera.name,
                "device": camera.path,
                "format": camera.format_name,
                "width": camera.width,
                "height": camera.height,
                "dropped_frames": camera.dropped_frames,
                **({"captured_at": str(images[camera.name].captured_at)}
                   if images is not None and camera.name in images else {}),
            }
            for camera in self.cameras
        ]

    def close(self) -> None:
        for camera in self.cameras:
            camera.close()


def discover_camera(width: int = 640, height: int = 480) -> str:
    """Find the first V4L2 node that can produce a normal RGB frame."""
    for path in _camera_candidates():
        try:
            camera = V4L2Camera(path, width=width, height=height)
        except (OSError, RuntimeError):
            continue
        camera.close()
        return path
    raise RuntimeError("没有发现可用相机，请用 --camera name=/dev/videoN 指定设备")


def _camera_candidates() -> list[str]:
    """Return color-capable nodes in a useful order without starting streams."""
    paths = sorted(
        (entry for entry in os.listdir("/dev") if re.fullmatch(r"video\d+", entry)),
        key=lambda value: int(value[5:]),
    )
    candidates: list[tuple[int, str]] = []
    priority = {_YUYV: 0, _RGB3: 1, _MJPG: 2, _UYVY: 3, _Z16: 4}
    for entry in paths:
        path = f"/dev/{entry}"
        try:
            fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
            formats = _enumerate_formats(fd)
            os.close(fd)
        except OSError:
            continue
        scores = [priority[code] for code in formats if code in priority]
        if scores:
            candidates.append((min(scores), path))
    if candidates:
        return [path for _, path in sorted(candidates)]
    return []


def _enumerate_formats(fd: int) -> set[int]:
    formats: set[int] = set()
    for index in range(64):
        description = _FormatDescription(index=index, type=_BUF_TYPE)
        try:
            fcntl.ioctl(fd, _ENUM_FMT, description, True)
        except OSError:
            break
        formats.add(description.pixelformat)
    return formats


def _encode_png(rgb: np.ndarray) -> bytes:
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("PNG 编码器只接受 RGB 图像")
    rows = b"".join(b"\0" + row.tobytes() for row in rgb)
    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return signature + _png_chunk(b"IHDR", header) + _png_chunk(
        b"IDAT", zlib.compress(rows, 6)
    ) + _png_chunk(b"IEND", b"")


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
