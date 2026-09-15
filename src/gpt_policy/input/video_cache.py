"""Content-addressed cache for decoded and Codex-selected video frames."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from .manifest import VideoPart
from .video import (
    CandidateFrame,
    FfmpegVideoExtractor,
    FrameSelection,
    SelectedFrame,
    VideoExtraction,
    VideoMetadata,
)


class CacheableFrameSelector(Protocol):
    def cache_identity(self) -> dict[str, Any]: ...

    def select(
        self, instruction: str, video: VideoPart, extraction: VideoExtraction
    ) -> FrameSelection: ...

    def review_identity(self): ...

    def review(self, instruction, video, frames, selection, limit): ...


@dataclass(frozen=True)
class CachedVideoProcessing:
    cache_key: str
    cache_hit: bool
    cache_dir: Path
    extraction: VideoExtraction
    selection: FrameSelection
    keyframes: tuple[CandidateFrame, ...]


class VideoProcessingCache:
    """Publish complete cache entries atomically and serialize equal work."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def resolve(
        self,
        instruction: str,
        video: VideoPart,
        extractor: FfmpegVideoExtractor,
        selector: CacheableFrameSelector,
    ) -> CachedVideoProcessing:
        size = video.path.stat().st_size
        if size > extractor.config.max_file_bytes:
            raise ValueError(
                f"视频文件过大: {size} bytes，最大允许 "
                f"{extractor.config.max_file_bytes} bytes"
            )
        source_hash = _sha256_file(video.path)
        identity = {
            "cache_format": 2,
            "video_sha256": source_hash,
            "instruction": instruction,
            "label": video.label,
            "extractor": {
                "algorithm": "windowed-pts-v2",
                "config": asdict(extractor.config),
                "event_times": list(extractor.event_times),
                "end_time_s": extractor.end_time_s,
            },
            "selector": selector.cache_identity(),
        }
        views = extractor.view_identity()
        if len(views) > 1:
            identity["views"] = views
        cache_key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        lock_dir = self.root / ".locks"
        lock_dir.mkdir(exist_ok=True)
        cache_dir = self.root / cache_key
        lock_path = lock_dir / f"{cache_key}.lock"

        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            cached = _load_entry(
                cache_key, cache_dir, video.path, identity, cache_hit=True
            )
            if cached is not None:
                return self._review(cached, instruction, video, selector)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

            temporary = Path(
                tempfile.mkdtemp(prefix=f".{cache_key[:12]}-", dir=self.root)
            )
            try:
                extraction = self._media(video.path, extractor, source_hash, temporary)
                selection = selector.select(instruction, video, extraction)
                mandatory = {0, len(extraction.candidates) - 1}
                # Events enrich candidates for visual selection. They are not
                # all independent semantic stages to force into model context.
                choices = {c.index: c for c in selection.selected}
                for i in mandatory:
                    choices.setdefault(i, SelectedFrame(i, "Timeline boundary or recorded event; verify the visible outcome."))
                selection = replace(selection, selected=tuple(choices[i] for i in sorted(choices)))
                chosen = tuple(
                    extraction.candidates[item.index] for item in selection.selected
                )
                keyframes = extractor.export_keyframes(extraction, chosen, temporary)
                if _sha256_file(video.path) != source_hash or extractor.view_identity() != views:
                    raise RuntimeError("Source videos changed during demonstration selection")
                _write_entry(
                    temporary,
                    identity,
                    extraction,
                    selection,
                    keyframes,
                )
                os.replace(temporary, cache_dir)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

            published = _load_entry(
                cache_key, cache_dir, video.path, identity, cache_hit=False
            )
            if published is None:
                raise RuntimeError("视频缓存发布后无法读取")
            return self._review(published, instruction, video, selector)

    @staticmethod
    def _review(cached, instruction, video, selector):
        # A separate final-review cache reuses previously paid window selections.
        limit = min(24, 48 // max(1 + len(f.views) for f in cached.keyframes))
        identity = hashlib.sha256(json.dumps({"selected": [asdict(c) for c in cached.selection.selected],
            "summary": cached.selection.summary, "review": selector.review_identity(), "limit": limit},
            ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        path = cached.cache_dir / "review-v1.json"
        hit = False
        try:
            value = json.loads(path.read_text())
            if value["source"] != identity:
                raise ValueError("Changed window selection")
            selection = FrameSelection(tuple(SelectedFrame(**c) for c in value["selected"]),
                                       value["summary"], value["wire"])
            indices = [c.index for c in selection.selected]
            available = {f.index for f in cached.keyframes}
            if (not min(2, len(available)) <= len(indices) <= limit or indices != sorted(set(indices))
                    or not set(indices) <= available or indices[0] != min(available)
                    or indices[-1] != max(available)):
                raise ValueError("Invalid reviewed frame indices")
            hit = True
        except (OSError, ValueError, KeyError, TypeError):
            selection = selector.review(instruction, video, cached.keyframes, cached.selection, limit)
            value = {"source": identity, "selected": [asdict(c) for c in selection.selected],
                     "summary": selection.summary, "wire": selection.wire}
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
            temporary.replace(path)
        selected = {c.index for c in selection.selected}
        return replace(cached, selection=selection,
                       keyframes=tuple(f for f in cached.keyframes if f.index in selected),
                       cache_hit=cached.cache_hit and hit)

    def _media(self, source, extractor, source_hash, destination):
        identity = {"source": source_hash, "config": asdict(extractor.config),
                    "events": list(extractor.event_times), "end_time_s": extractor.end_time_s, "version": 2}
        views = extractor.view_identity()
        if len(views) > 1:
            identity["views"] = views
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        root = self.root / "media" / key
        root.parent.mkdir(exist_ok=True)
        with (root.parent / f"{key}.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                data = json.loads((root / "media.json").read_text())
                frames = tuple(_load_frame(root, f) for f in data["candidates"])
                if not frames or any(not image.path.is_file() for f in frames for image in (f, *f.views.values())):
                    raise ValueError("Missing media frames")
                metadata = VideoMetadata(**data["metadata"])
            except (OSError, ValueError, KeyError, TypeError):
                temp = Path(tempfile.mkdtemp(prefix=".media-", dir=root.parent))
                try:
                    extraction = extractor.extract_candidates(source, temp)
                    if _sha256_file(source) != source_hash or extractor.view_identity() != views:
                        raise RuntimeError("Source video changed during extraction")
                    data = {"metadata": extraction.metadata.record(), "candidates": [
                        f.record(temp) for f in extraction.candidates]}
                    (temp / "media.json").write_text(json.dumps(data, allow_nan=False))
                    if root.exists():
                        shutil.rmtree(root)
                    os.replace(temp, root)
                finally:
                    if temp.exists():
                        shutil.rmtree(temp)
                metadata = extraction.metadata
                frames = tuple(_load_frame(root, f) for f in data["candidates"])
        # Hard links preserve self-contained semantic entries without decoding again.
        def copy_frame(frame):
            path = destination / frame.path.relative_to(root)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Adjacent top-camera timestamps can legitimately resolve to the
            # same lower-rate wrist-camera frame. Reuse that already copied
            # file instead of treating it as a conflicting duplicate.
            if frame.path.resolve() == path.resolve() or path.exists():
                return replace(frame, path=path, views={name: copy_frame(f) for name, f in frame.views.items()})
            try:
                os.link(frame.path, path)
            except OSError:
                shutil.copy2(frame.path, path)
            return replace(frame, path=path, views={name: copy_frame(f) for name, f in frame.views.items()})
        copied = [copy_frame(frame) for frame in frames]
        return VideoExtraction(source, metadata, tuple(copied))


def _write_entry(
    root: Path,
    identity: dict[str, Any],
    extraction: VideoExtraction,
    selection: FrameSelection,
    keyframes: tuple[CandidateFrame, ...],
) -> None:
    value = {
        "identity": identity,
        "metadata": extraction.metadata.record(),
        "candidates": [item.record(root) for item in extraction.candidates],
        "selection": {
            "selected": [asdict(item) for item in selection.selected],
            "summary": selection.summary,
            "wire": selection.wire,
        },
        "keyframes": [item.record(root) for item in keyframes],
    }
    path = root / "cache.json"
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _load_entry(
    cache_key: str,
    root: Path,
    source: Path,
    expected_identity: dict[str, Any],
    cache_hit: bool,
) -> CachedVideoProcessing | None:
    try:
        value = json.loads((root / "cache.json").read_text(encoding="utf-8"))
        if value.get("identity") != expected_identity:
            return None
        metadata = VideoMetadata(**value["metadata"])
        candidates = tuple(_load_frame(root, item) for item in value["candidates"])
        raw_selection = value["selection"]
        selected = tuple(SelectedFrame(**item) for item in raw_selection["selected"])
        wire = raw_selection["wire"]
        if not isinstance(wire, dict):
            return None
        selection = FrameSelection(selected, str(raw_selection["summary"]), wire)
        keyframes = tuple(_load_frame(root, item) for item in value["keyframes"])
        if not candidates or not keyframes or len(keyframes) != len(selected):
            return None
        if any(item.index != index for index, item in enumerate(candidates)):
            return None
        selected_indices = tuple(item.index for item in selected)
        if len(set(selected_indices)) != len(selected_indices):
            return None
        if any(not 0 <= index < len(candidates) for index in selected_indices):
            return None
        if tuple(item.index for item in keyframes) != tuple(item.index for item in selected):
            return None
        if any(not image.path.is_file() for item in (*candidates, *keyframes) for image in (item, *item.views.values())):
            return None
        # Old caches forced every numeric event into the final keyframes. Keep
        # their original model decisions plus endpoints, without repeating vision
        # calls. The exact host-added marker had no semantic annotations.
        pairs = [(frame, choice) for frame, choice in zip(keyframes, selected)
                 if choice.index in (0, len(candidates) - 1)
                 or choice.reason != "Timeline boundary or recorded event; verify the visible outcome."
                 or any((choice.stage, choice.left, choice.right, choice.result))]
        keyframes = tuple(frame for frame, _ in pairs)
        selection = replace(selection, selected=tuple(choice for _, choice in pairs))
        extraction = VideoExtraction(source, metadata, candidates)
        return CachedVideoProcessing(
            cache_key, cache_hit, root, extraction, selection, keyframes
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _load_frame(root: Path, value: dict[str, Any]) -> CandidateFrame:
    relative = Path(value["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("缓存帧路径必须位于缓存目录内")
    return CandidateFrame(
        int(value["index"]), float(value["timestamp_s"]), root / relative, value.get("frame_index"),
        {name: _load_frame(root, f) for name, f in value.get("views", {}).items()},
        value.get("capture_delta_s"),
    )


def _sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"计算缓存键时视频文件发生变化: {path}")
    return digest.hexdigest()
