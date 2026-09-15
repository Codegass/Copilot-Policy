"""Small, append-only JSONL recorder for reproducible robot runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from ..hardware.camera import CapturedImage
from ..harness.usage import attach_usage, usage_summary


_OUTCOMES = {
    "completed": "success",
    "failed": "failed",
    "give_up": "give_up",
    "budget_exhausted": "give_up",
    "interrupted": "interrupted",
    "unreviewed": "unreviewed",
}


class RunRecorder:
    """Persist protocol messages, states, decisions, results, and camera frames."""

    def __init__(self, root: Path, metadata: Mapping[str, Any]) -> None:
        self.root = root
        self.frames = root / "frames"
        self.frames.mkdir(parents=True, exist_ok=False)
        self.events = root / "events.jsonl"
        self._stream = self.events.open("x", encoding="utf-8")
        self.started_at = time.time()
        self.metadata = dict(metadata)
        self.transcript: list[dict[str, Any]] = []
        self.recording: dict[str, Any] | None = None
        self.task_status: str | None = None
        self.model_task_status: str | None = None
        self.human_outcome: str | None = None
        self.execution_finished_at: float | None = None
        _save(self.root / "config.json", self.metadata)
        self.write("run_started", dict(metadata))
        attach_usage(root)

    def write(self, kind: str, payload: Mapping[str, Any] | None = None) -> None:
        values = dict(payload or {})
        if kind == "terminal":
            self.model_task_status = "give_up" if values["name"] == "give_up" else "completed"
        elif kind == "human_evaluation":
            if values.get("outcome") not in {"success", "failed"}:
                raise ValueError("Human outcome must be success or failed")
            self.human_outcome = values["outcome"]
            self.task_status = "completed" if self.human_outcome == "success" else "failed"
        event = {"at_s": time.time(), "event": kind, **values}
        if kind == "execution_finished":
            self.execution_finished_at = event["at_s"]
        self._stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._stream.flush()
        self._transcript_event(kind, values)

    def observation(
        self,
        step: int,
        text: str,
        state: Mapping[str, Any],
        cameras: list[dict[str, Any]],
        images: Mapping[str, CapturedImage],
        *,
        attempt: int = 0,
    ) -> None:
        image_records = []
        for name, image in images.items():
            suffix = ".jpg" if image.mime_type == "image/jpeg" else ".png"
            retry = f"-retry-{attempt:02d}" if attempt else ""
            path = self.frames / f"step-{step:05d}{retry}-{_safe_name(name)}{suffix}"
            path.write_bytes(image.data)
            image_records.append(
                {
                    "name": name,
                    "path": str(path.relative_to(self.root)),
                    "mime_type": image.mime_type,
                    "width": image.width,
                    "height": image.height,
                    "captured_at": image.captured_at,
                    "source_timestamp_s": image.source_timestamp_s,
                    "source_clock": image.source_clock,
                }
            )
        self.write(
            "observation",
            {"step": step, "input_json": text, "state": state, "cameras": cameras, "images": image_records,
             **({"attempt": attempt} if attempt else {})},
        )

    def set_recording(self, details: Mapping[str, Any]) -> None:
        self.recording = dict(details)
        _save(self.root / "recording.json", self.recording)

    def close(self, status: str = "completed", error: str | None = None) -> Path:
        # Model termination is not a physical success/failure label. Preserve
        # runtime failures when there was no model conclusion or human verdict.
        task_status = self.task_status or (
            "unreviewed" if self.model_task_status or status in {"completed", "give_up", "budget_exhausted"}
            else status
        )
        self.task_status = task_status
        outcome = _OUTCOMES[task_status]
        verdict = {
            "task_status": task_status, "outcome": outcome,
            "model_outcome": _OUTCOMES.get(self.model_task_status),
            "human_outcome": self.human_outcome,
            "outcome_source": "human" if self.human_outcome else "unreviewed" if task_status == "unreviewed" else "runtime",
        }
        self.write("run_finished", {"status": status, **verdict, "error": error})
        _save(self.root / "transcript.json", self.transcript)
        _save(
            self.root / "status.json",
            {
                "state": status,
                "error": error,
                "elapsed_s": (self.execution_finished_at or time.time()) - self.started_at,
                "recording": self.recording,
                **self.metadata,
                **verdict,
                "usage": usage_summary(),
            },
        )
        self._stream.close()
        # All producers are stopped by the caller before closing the recorder.
        # Paths to recordings/frames are relative, so the whole run can move.
        destination = self.root.with_name(f"{self.root.name}_{outcome}")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Final recording directory already exists: {destination}")
        self.root.rename(destination)
        attach_usage(destination)
        self.root = destination
        self.frames = destination / "frames"
        self.events = destination / "events.jsonl"
        return destination

    def _transcript_event(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "protocol":
            self.transcript.append({"role": "system", "content": payload.get("base_instructions", "")})
            _save(self.root / "protocol.json", payload)
        elif kind == "input_manifest":
            self.transcript.append({"role": "user", "content": payload})
        elif kind == "observation":
            self.transcript.append({
                "role": "user",
                "content": payload.get("input_json", ""),
                "images": payload.get("images", []),
            })
        elif kind == "model_decision":
            self.transcript.append({"role": "assistant", "tool_call": payload.get("decision")})
        elif kind in {"execution_result", "tool_error"}:
            self.transcript.append({"role": "tool", "content": payload})
        elif kind in {"human_evaluation", "model_retry", "model_retry_exhausted"}:
            self.transcript.append({"role": "user", "content": {"event": kind, **payload}})
        if kind in {"protocol", "input_manifest", "observation", "model_decision", "execution_result", "tool_error", "human_evaluation", "model_retry", "model_retry_exhausted"}:
            _save(self.root / "transcript.json", self.transcript)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
