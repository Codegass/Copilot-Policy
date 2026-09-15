"""Encode image bytes as native ACP content instead of file attachments."""

from ...media import EncodedImage
from ....input.manifest import TextPart


def prompt_blocks(blocks: list[TextPart | EncodedImage]) -> list[dict]:
    # ACP interprets a leading slash as a CLI command; task text is always data.
    result = [{"type": "text", "text": "Robot observation and task input:\n"}]
    for block in blocks:
        if isinstance(block, TextPart):
            result.append({"type": "text", "text": block.text})
        else:
            result.append({"type": "image", "data": block.data, "mimeType": block.mime_type})
    return result
