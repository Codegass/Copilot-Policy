"""Translate neutral text/images to Claude stream-json input."""

from ...media import EncodedImage
from ....input.manifest import TextPart


def user_message(blocks: list[TextPart | EncodedImage]) -> dict:
    content = []
    for block in blocks:
        if isinstance(block, TextPart):
            content.append({"type": "text", "text": block.text})
        else:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": block.mime_type, "data": block.data,
            }})
    return {"type": "user", "message": {"role": "user", "content": content}}
