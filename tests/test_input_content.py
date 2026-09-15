from __future__ import annotations

import base64
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from gpt_policy.hardware.camera import CapturedImage
from gpt_policy.harness.codex import CodexAppServer
from gpt_policy.harness.input_content import to_app_server_items, validate_input_size, MAX_INPUT_CHARS
from gpt_policy.input import ImagePart, TextPart


class InputContentTest(unittest.TestCase):
    def test_size_counts_unicode_text_and_image_labels_but_not_base64(self):
        overhead = len("Image: x") + len("abc")
        parts = (TextPart("中" * (MAX_INPUT_CHARS - overhead)), ImagePart(Path("unread.png"), label="x"))
        self.assertEqual(validate_input_size(parts, "abc"), MAX_INPUT_CHARS)
        with self.assertRaisesRegex(ValueError, "too large"):
            validate_input_size(parts, "abcd")
        with self.assertRaisesRegex(ValueError, "too large"):
            validate_input_size(parts, "abc", camera_names=("left",))

    def test_transport_preserves_interleaving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.png"
            image_path.write_bytes(b"png-bytes")
            items = to_app_server_items(
                (
                    TextPart("前面的文字"),
                    ImagePart(image_path, label="示例"),
                    TextPart("后面的文字"),
                )
            )

        self.assertEqual([item["type"] for item in items], ["text", "text", "image", "text"])
        self.assertEqual(items[1]["text"], "Image: 示例")
        self.assertTrue(items[2]["url"].startswith("data:image/png;base64,"))
        encoded = items[2]["url"].split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), b"png-bytes")

    def test_size_matches_actual_english_labels_and_unchanged_user_text(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.png"
            image_path.write_bytes(b"png-bytes")
            parts = (TextPart("用户原文"), ImagePart(image_path, label="示例"))
            images = {"left": CapturedImage("left", b"png-bytes", "image/png", 1, 1, 0.0)}
            items = CodexAppServer._input("实时状态", images, parts)

        actual_chars = sum(len(item["text"]) for item in items if item["type"] == "text")
        self.assertEqual(validate_input_size(parts, "实时状态", camera_names=images), actual_chars)
        self.assertIn({"type": "text", "text": "Camera image: left"}, items)

    def test_codex_input_appends_live_observation_after_static_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.png"
            image_path.write_bytes(b"png-bytes")
            items = CodexAppServer._input(
                "实时状态",
                None,
                (TextPart("静态提示"), ImagePart(image_path)),
            )

        self.assertEqual(items[0], {"type": "text", "text": "静态提示"})
        self.assertEqual(items[1]["type"], "image")
        self.assertEqual(items[2], {"type": "text", "text": "实时状态"})

    def test_live_camera_image_can_be_converted_to_jpeg(self) -> None:
        image = CapturedImage(
            "left",
            b"unused-png",
            "image/png",
            2,
            1,
            0.0,
            bytes((255, 0, 0, 0, 255, 0)),
        )

        items = CodexAppServer._input(
            "实时状态",
            {"left": image},
            convert_camera_images_to_jpeg=True,
            camera_jpeg_quality=85,
        )

        url = items[-1]["url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as decoded:
            self.assertEqual(decoded.format, "JPEG")
            self.assertEqual(decoded.size, (2, 1))

    def test_live_camera_conversion_can_be_disabled(self) -> None:
        image = CapturedImage("left", b"png-bytes", "image/png", 1, 1, 0.0)

        items = CodexAppServer._input(
            "实时状态", {"left": image}, convert_camera_images_to_jpeg=False
        )

        self.assertTrue(items[-1]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(items[-1]["url"].split(",", 1)[1]), b"png-bytes")

    def test_existing_camera_jpeg_is_not_reencoded(self) -> None:
        image = CapturedImage("left", b"jpeg-bytes", "image/jpeg", 1, 1, 0.0)

        items = CodexAppServer._input(
            "实时状态", {"left": image}, convert_camera_images_to_jpeg=True
        )

        self.assertEqual(base64.b64decode(items[-1]["url"].split(",", 1)[1]), b"jpeg-bytes")


if __name__ == "__main__":
    unittest.main()
