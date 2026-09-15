"""Human-readable terminal output; recording and model payloads stay untouched."""

from __future__ import annotations

from contextlib import contextmanager
import json
import sys
from typing import Any

from rich.console import Console
from rich.padding import Padding
from rich.rule import Rule
from rich.text import Text
from rich.tree import Tree


class RunConsole:
    def __init__(self, console: Console | None = None):
        self.console = console if console is not None else Console(highlight=False, markup=False)

    def header(self, instruction, model, machine, record_dir, budget):
        self.heading(_text(f"GPT POLICY · {str(machine).upper()} · {model}", "bold"))
        self.fields({"task": instruction, "recording_directory": str(record_dir), "decision_budget": budget})

    def message(self, message: str, style: str = "dim"):
        self.console.print(_text(message, "" if self.console.no_color else style))

    @contextmanager
    def waiting(self, message: str):
        # Plain SSH pipes / log files get one ordinary line and no cursor codes.
        if self.console.is_interactive and not self.console.no_color:
            with self.console.status(_text(message), spinner="dots", refresh_per_second=4):
                yield
        else:
            self.message(message + "...")
            yield

    def decision(self, step: int, action: str, arguments: dict):
        sides = _sides(arguments)
        title = f"STEP {step:03d} · {_action(action)}"
        if sides:
            title += " · " + "/".join(side.upper() for side in sides)
        self.event(title, "REQUESTED", arguments, "cyan")

    def error(self, step: int, action: str, error: dict):
        status = "NOT EXECUTED" if error.get("error") == "motion_not_executed" else "REJECTED"
        self.event(f"STEP {step:03d} · {_action(action)}", status, error, "yellow")

    def returned(self, step: int, action: str):
        # Returning from a tool is not proof of grasp/task success or settling.
        self.message(f"STEP {step:03d} · {_action(action)} · RESULT RECORDED")

    def home(self, result: Any):
        self.event("RETURN HOME", "FEEDBACK", result, "cyan")

    def task_result(self) -> str | None:
        """Collect only a final label, after recording and device shutdown."""
        if sys.stdin is None or not sys.stdin.isatty():
            self.message("无交互终端，未记录人工判定。", "yellow")
            return None
        while True:
            try:
                answer = self.console.input("任务结果 [s 成功 / f 失败]: ", markup=False).strip().lower()
            except (EOFError, KeyboardInterrupt, OSError):
                self.message("未选择结果，未记录人工判定。", "yellow")
                return None
            if answer in {"s", "success", "成功"}:
                return "success"
            if answer in {"f", "failed", "失败"}:
                return "failed"
            self.message("请输入 s（成功）或 f（失败），回车不默认选择。", "yellow")

    def finished(self, status: str, directory=None, *, task_status=None, error=None):
        labels = {"completed": "SUCCESS", "failed": "FAILED", "give_up": "GIVE UP",
                  "budget_exhausted": "GIVE UP · DECISION BUDGET EXHAUSTED", "interrupted": "INTERRUPTED",
                  "unreviewed": "UNREVIEWED"}
        task_status = task_status or status
        color = "green" if task_status == "completed" else "red" if task_status == "failed" else "yellow"
        self.event("TASK", labels.get(task_status, task_status.upper()),
                   {"error": error} if task_status == status else {}, color)
        if task_status != status and status in {"failed", "interrupted"}:
            self.event("FINALIZATION", labels.get(status, status.upper()), {"error": error},
                       "red" if status == "failed" else "yellow")
        if directory is not None:
            self.message(f"Recording saved: {directory}")

    def event(self, title: str, status: str, payload: Any, style="cyan"):
        heading = _text(title, "bold")
        heading.append(" · ", "dim")
        heading.append(status, style)
        self.console.print()
        self.heading(heading)
        self.fields(payload)

    def usage(self, summary):
        if not summary or not summary["calls"]:
            return
        tokens = summary["tokens"]
        cost = summary["estimated_cost_usd"]
        price = f"${cost:.6f}" if cost is not None else (
            f"${summary['priced_cost_usd']:.6f} known subtotal; {summary['unpriced_calls']} unpriced call(s)")
        self.message(f"Model usage: {summary['calls']} calls | input {tokens['input_tokens']:,} "
                     f"(cached {tokens['cached_input_tokens']:,}) | output {tokens['output_tokens']:,} | "
                     f"API-equivalent estimate: {price}")
        if summary["usage_incomplete_calls"]:
            self.message(f"Token usage incomplete/unavailable for {summary['usage_incomplete_calls']} call(s).", "yellow")

    def heading(self, text: Text):
        # Rich's Rule truncates long titles. Wrap instead on narrow terminals.
        if text.cell_len + 4 <= self.console.width:
            self.console.print(Rule(text, align="left", style="dim"))
        else:
            self.console.print(text)
            self.console.print(Rule(style="dim"))

    def fields(self, payload: Any):
        if not isinstance(payload, dict):
            self.console.print(Padding(_text(_scalar(payload)), (0, 2)))
            return
        # Notes remain verbatim and fully expanded. Display shaping never edits
        # the source dict passed to the recorder, tool executor or next model turn.
        prose = {"note", "summary", "hindsight", "reason"}
        for key, value in payload.items():
            if key == "_wire" or value is None or key in prose:
                continue
            node = _node(_label(key), value, key)
            self.console.print(Padding(node, (0, 2)))
        for key, value in payload.items():
            if key in prose and value is not None:
                self.console.print(Padding(_text(_label(key), "bold"), (0, 2)))
                self.console.print(Padding(_text(str(value)), (0, 4)))


def _text(value: str, style: str = "") -> Text:
    # Text treats model-supplied [brackets] as literal text, not Rich markup.
    return Text(value, style=style, overflow="fold", no_wrap=False)


def _action(value: str) -> str:
    return value.replace("_", " ").upper()


def _label(value: str) -> str:
    labels = {"pose_xyzquat": "TCP pose", "gripper": "Gripper opening (0–1)",
              "positions": "Gripper openings (0–1)", "poses": "Waypoints",
              "pixel_xy": "Pixel (x, y)", "reference_pixel_xy": "Reference pixel (x, y)",
              "sdk_status": "SDK status", "sdk_status_name": "SDK status name",
              "best_translation_error_m": "Best translation error (m)",
              "best_rotation_error_rad": "Best rotation error (rad)"}
    return labels.get(value, value.replace("_", " ").capitalize())


def _scalar(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _node(label: str, value: Any, key: str = "") -> Tree:
    node = Tree(_text(label, "bold"), guide_style="dim", highlight=False)
    if key == "pose_xyzquat" and isinstance(value, (list, tuple)) and len(value) == 7:
        node.add(_text("Position (m)  " + "  ".join(f"{axis}={_scalar(v)}" for axis, v in zip("xyz", value[:3]))))
        node.add(_text("Quaternion (xyzw)  " + "  ".join(f"q{axis}={_scalar(v)}" for axis, v in zip("xyzw", value[3:]))))
    elif isinstance(value, dict) and value:
        for field, item in value.items():
            if field != "_wire" and item is not None:
                node.add(_node(_label(field), item, field))
    elif isinstance(value, (list, tuple)) and any(isinstance(item, (dict, list, tuple)) for item in value):
        for index, item in enumerate(value):
            # Preserve indices and held waypoints; null list entries must not
            # shift the correspondence between left/right arm trajectories.
            name = f"Waypoint {index + 1}" if key == "poses" else f"[{index}]"
            node.add(_node(name, item))
    else:
        node.label.append("  ", "dim")
        node.label.append(_scalar(value), "not bold")
    return node


def _sides(value: Any) -> list[str]:
    selected = set()
    def visit(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key in {"left", "right"} and child is not None:
                    selected.add(key)
                elif key != "_wire":
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)
    return [side for side in ("left", "right") if side in selected]
