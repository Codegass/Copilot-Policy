"""Encode ordered input blocks for new providers, independently of Codex."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from ..input.manifest import ImagePart, TextPart
from .models import AgentTurn, ImageInput


@dataclass(frozen=True)
class EncodedImage:
    data: str
    mime_type: str


def content_blocks(
    turn: AgentTurn, convert_images: bool, jpeg_quality: int,
) -> list[TextPart | EncodedImage]:
    if not 1 <= jpeg_quality <= 95:
        raise ValueError("camera_jpeg_quality 必须在 1 到 95 之间")
    blocks: list[TextPart | EncodedImage] = []
    for part in turn.content or ():
        if isinstance(part, TextPart):
            blocks.append(part)
        elif isinstance(part, ImagePart):
            mime, _ = mimetypes.guess_type(part.path.name)
            if mime is None or not mime.startswith("image/"):
                raise ValueError(f"无法识别图片类型: {part.path}")
            if part.label:
                blocks.append(TextPart(f"Image: {part.label}"))
            blocks.append(_encode(part.path.read_bytes(), mime))
        else:
            raise TypeError(f"不支持的输入块类型: {type(part)!r}")
    blocks.append(TextPart(turn.observation))
    for name, captured in (turn.images or {}).items():
        blocks.append(TextPart(f"Camera image: {name}"))
        blocks.append(_camera(captured, convert_images, jpeg_quality))
    return blocks


def _encode(data: bytes, mime_type: str) -> EncodedImage:
    return EncodedImage(base64.b64encode(data).decode("ascii"), mime_type)


def _camera(image: ImageInput, convert: bool, quality: int) -> EncodedImage:
    if not convert or image.mime_type == "image/jpeg":
        return _encode(image.data, image.mime_type)
    if image.rgb_data is not None:
        rgb = Image.frombytes("RGB", (image.width, image.height), image.rgb_data)
    else:
        with Image.open(BytesIO(image.data)) as source:
            rgb = source.convert("RGB")
    with BytesIO() as stream:
        rgb.save(stream, format="JPEG", quality=quality)
        return _encode(stream.getvalue(), "image/jpeg")
