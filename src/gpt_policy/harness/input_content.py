"""Convert user-facing input parts into Codex app-server content items."""

from __future__ import annotations

import base64
import mimetypes
from typing import Any, Iterable

from ..input.manifest import ContentPart, ImagePart, TextPart


# Codex turn/start counts Unicode text characters, including image labels.
# Binary image content is subject to separate provider limits.
MAX_INPUT_CHARS = 1048576
FIRST_TURN_RESERVE_CHARS = 65536


def validate_input_size(parts=(), observation="", camera_names=(), *, reserve_chars=0):
    """Check the same text sent on the wire without reading/encoding image bytes."""
    count = len(observation) + sum(len(f"Camera image: {name}") for name in camera_names)
    for part in parts or ():
        if isinstance(part, TextPart):
            count += len(part.text)
        elif isinstance(part, ImagePart) and part.label:
            count += len(f"Image: {part.label}")
    if count + reserve_chars > MAX_INPUT_CHARS:
        raise ValueError(f"Codex input is too large: {count} text characters + {reserve_chars} reserved, "
                         f"limit {MAX_INPUT_CHARS}. Prepare a shorter demonstration or reduce input text; "
                         "no input was truncated.")
    return count


def to_app_server_items(parts: Iterable[ContentPart]) -> list[dict[str, Any]]:
    """Preserve text/image order while producing native image input items."""
    items: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            items.append({"type": "text", "text": part.text})
            continue
        if isinstance(part, ImagePart):
            item: dict[str, Any] = {"type": "image", "url": _data_url(part)}
            if part.label:
                items.append({"type": "text", "text": f"Image: {part.label}"})
            if part.detail:
                item["detail"] = part.detail
            items.append(item)
            continue
        raise TypeError(f"不支持的输入块类型: {type(part)!r}")
    return items


def _data_url(part: ImagePart) -> str:
    mime_type, _ = mimetypes.guess_type(part.path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(f"无法识别图片类型: {part.path}")
    encoded = base64.b64encode(part.path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
