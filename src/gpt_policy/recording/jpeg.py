"""JPEG encoding through Pillow's native codec."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


def encode_jpeg(rgb: np.ndarray, quality: int = 70) -> bytes:
    """Encode an RGB frame without a Python loop over pixels or JPEG blocks."""
    with BytesIO() as stream:
        Image.fromarray(rgb).save(stream, format="JPEG", quality=quality)
        return stream.getvalue()
