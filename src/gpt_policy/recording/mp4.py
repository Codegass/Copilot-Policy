"""Minimal Motion-JPEG MP4 muxer with no external runtime dependency."""

from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import BinaryIO


def _atom(name: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, name) + payload


def _full(name: bytes, version: int, flags: int, payload: bytes) -> bytes:
    return _atom(name, bytes([version]) + flags.to_bytes(3, "big") + payload)


_MATRIX = struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)


class MjpegMp4Writer:
    """Append JPEG samples and publish the MP4 index when closed."""

    def __init__(self, path: Path, width: int, height: int, fps: int = 2) -> None:
        self.path, self.width, self.height, self.fps = path, width, height, fps
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream: BinaryIO = path.open("x+b")
        self.stream.write(_atom(b"ftyp", b"isom\0\0\x02\0isomiso2mp41"))
        self.mdat_start = self.stream.tell()
        self.stream.write(struct.pack(">I4sQ", 1, b"mdat", 0))
        self.offsets: list[int] = []
        self.sizes: list[int] = []

    def append(self, jpeg: bytes) -> None:
        self.offsets.append(self.stream.tell())
        self.sizes.append(len(jpeg))
        self.stream.write(jpeg)
        self.stream.flush()

    def close(self) -> None:
        if self.stream.closed:
            return
        end = self.stream.tell()
        self.stream.seek(self.mdat_start + 8)
        self.stream.write(struct.pack(">Q", end - self.mdat_start))
        self.stream.seek(end)
        self.stream.write(self._moov())
        self.stream.flush()
        self.stream.close()

    def _moov(self) -> bytes:
        count = len(self.sizes)
        created = int(time.time()) + 2082844800
        mvhd = _full(
            b"mvhd", 0, 0,
            struct.pack(">IIII", created, created, self.fps, count)
            + struct.pack(">Ih10x", 0x10000, 0x100) + _MATRIX + b"\0" * 24 + struct.pack(">I", 2),
        )
        tkhd = _full(
            b"tkhd", 0, 7,
            struct.pack(">IIIII8xhhH2x", created, created, 1, 0, count, 0, 0, 0)
            + _MATRIX + struct.pack(">II", self.width << 16, self.height << 16),
        )
        mdhd = _full(b"mdhd", 0, 0, struct.pack(">IIIIHH", created, created, self.fps, count, 0x55C4, 0))
        hdlr = _full(b"hdlr", 0, 0, b"\0" * 4 + b"vide" + b"\0" * 12 + b"Robot camera\0")
        vmhd = _full(b"vmhd", 0, 1, struct.pack(">HHHH", 0, 0, 0, 0))
        url = _full(b"url ", 0, 1, b"")
        dinf = _atom(b"dinf", _full(b"dref", 0, 0, struct.pack(">I", 1) + url))
        stbl = self._sample_table(count)
        minf = _atom(b"minf", vmhd + dinf + stbl)
        mdia = _atom(b"mdia", mdhd + hdlr + minf)
        return _atom(b"moov", mvhd + _atom(b"trak", tkhd + mdia))

    def _sample_table(self, count: int) -> bytes:
        compressor = b"Robot Motion JPEG"
        visual = (
            b"\0" * 6 + struct.pack(">H", 1) + b"\0" * 16
            + struct.pack(">HHII4xH", self.width, self.height, 0x480000, 0x480000, 1)
            + bytes([len(compressor)]) + compressor.ljust(31, b"\0")
            + struct.pack(">Hh", 24, -1)
        )
        stsd = _full(b"stsd", 0, 0, struct.pack(">I", 1) + _atom(b"jpeg", visual))
        stts = _full(b"stts", 0, 0, struct.pack(">III", 1, count, 1))
        stsc = _full(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, 1, 1))
        stsz = _full(b"stsz", 0, 0, struct.pack(">II", 0, count) + b"".join(struct.pack(">I", size) for size in self.sizes))
        stco = _full(b"co64", 0, 0, struct.pack(">I", count) + b"".join(struct.pack(">Q", offset) for offset in self.offsets))
        stss = _full(b"stss", 0, 0, struct.pack(">I", count) + b"".join(struct.pack(">I", i + 1) for i in range(count)))
        return _atom(b"stbl", stsd + stts + stsc + stsz + stco + stss)

    def __enter__(self) -> "MjpegMp4Writer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
