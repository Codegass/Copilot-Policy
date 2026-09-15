"""Recover complete JPEG samples after an interrupted MP4 finalization.

Usage: python -m gpt_policy.recording.recover /path/to/run left
The original recording is never edited or overwritten.
"""

import argparse
import json
from pathlib import Path

from .mp4 import MjpegMp4Writer


def recover(directory, camera):
    if not camera or Path(camera).name != camera:
        raise ValueError("camera must be a simple name")
    output = directory / f"{camera}.recovered.mp4"
    writer = None
    try:
        with (directory / f"{camera}.mp4").open("rb") as source, (directory / "video-frames.jsonl").open() as index:
            for line in index:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break  # a killed writer may leave one partial final line
                if row["camera"] != camera:
                    continue
                source.seek(row["jpeg_offset"])
                jpeg = source.read(row["jpeg_size"])
                if len(jpeg) != row["jpeg_size"] or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
                    break
                if writer is None:
                    writer = MjpegMp4Writer(output, row["width"], row["height"], row["fps"])
                writer.append(jpeg)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"No complete samples found for {camera}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("camera")
    args = parser.parse_args()
    print(recover(args.directory, args.camera))
