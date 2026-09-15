"""Read current run recordings as historical demonstration data."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import json
import math
from pathlib import Path

from .action_sampling import event_times
from .video_cache import _sha256_file


def _invalid_number(value):
    raise ValueError(f"Invalid number: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_invalid_number)


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line, parse_constant=_invalid_number)


def nearest_index(values, value):
    i = bisect_left(values, value)
    return min((j for j in (i - 1, i) if 0 <= j < len(values)), key=lambda j: abs(values[j] - value))


class RecordedDemo:
    def __init__(self, root, extractor, with_actions):
        self.root = Path(root)
        names = ["config.json", "status.json", "video-frames.jsonl", "events.jsonl"]
        if with_actions:
            names.append("states.jsonl")
        self.source_hashes = {name: _sha256_file(self.root / name) for name in names}
        self.video = self.root / "top.mp4"
        self.pts = extractor.frame_times(self.video)
        self.frames = [r for r in read_rows(self.root / "video-frames.jsonl") if r["camera"] == "top"]
        if len(self.frames) != len(self.pts) or any(r["frame_index"] != i for i, r in enumerate(self.frames)):
            raise ValueError("Recorded video and frame index have different frame counts/order")
        self.capture_times = [float(r["captured_at_s"]) for r in self.frames]
        if any(not math.isfinite(t) for t in self.capture_times) or any(b < a for a, b in zip(self.capture_times, self.capture_times[1:])):
            raise ValueError("Recorded camera host times must be finite and nondecreasing")
        self.views = {"top": (self.video, self.pts, self.capture_times)}
        for camera in ("left", "right"):
            path = self.root / f"{camera}.mp4"
            if not path.is_file():
                continue
            pts = extractor.frame_times(path)
            rows = [r for r in read_rows(self.root / "video-frames.jsonl") if r["camera"] == camera]
            clock = [float(r["captured_at_s"]) for r in rows]
            if (len(rows) != len(pts) or any(r["frame_index"] != i for i, r in enumerate(rows))
                    or any(not math.isfinite(t) for t in clock)
                    or any(b < a for a, b in zip(clock, clock[1:]))):
                raise ValueError(f"Recorded {camera} video and capture index do not match")
            self.views[f"{camera}_wrist"] = (path, pts, clock)
        config = read_json(self.root / "config.json")
        status = read_json(self.root / "status.json")
        self.end_at = self.capture_times[-1]
        self.requests = []
        outcomes = {}
        terminal_status = None
        for row in read_rows(self.root / "events.jsonl"):
            event = row.get("event")
            if event == "terminal":
                self.end_at = min(self.end_at, float(row["at_s"]))
                terminal_status = "completed" if row.get("name") == "done" else "give_up" if row.get("name") == "give_up" else None
                break
            if with_actions and event == "model_decision":
                decision = row["decision"]
                self.requests.append({"at_s": row["at_s"], "step": row["step"],
                                      "name": decision["name"], "arguments": decision["arguments"]})
            elif with_actions and event in ("execution_result", "tool_error"):
                outcomes[row["step"]] = event
        for request in self.requests:
            request["reported_result"] = outcomes.get(request["step"], "not_recorded")
        task_status = status.get("task_status") or terminal_status or "unknown"
        self.end_pts = self.pts[max(0, bisect_right(self.capture_times, self.end_at) - 1)]
        self.metadata = {
            "title": config["instruction"], "source": str(self.root.resolve()),
            "demonstrator": "policy_rollout_excerpt", "coverage": "sampled",
            "outcome": task_status, "finalization_state": status.get("state"),
            "source_robot": {"model": config.get("robot_model"),
                             "machine": config.get("settings", {}).get("machine"),
                             "interfaces": [config.get("interface"), config.get("right_interface")]},
            "coordinate_frame": "Source arm base frames and source TCP, metres, radians, quaternion xyzw. "
                                "No verified transform to the current robot. Gripper normalized: 0 closed, 1 open. "
                                "Samples are measured feedback; gripper_command is the command reported with feedback. "
                                "Requested tools are historical intent, not proof of execution or contact.",
        }
        self.samples = []
        self.state_times = []
        if with_actions:
            offsets = []
            for row in read_rows(self.root / "states.jsonl"):
                t = float(row["observed_at_s"])
                if t > self.end_at:
                    break
                if not math.isfinite(t) or (self.state_times and t <= self.state_times[-1]):
                    raise ValueError("State host timestamps must strictly increase")
                if "observed_monotonic_s" in row:
                    offsets.append(t - float(row["observed_monotonic_s"]))
                state = row["state"]
                arms = state.get("arms", {"left": state})
                sample = {"t_s": t}
                for side, arm in arms.items():
                    if side not in ("left", "right"):
                        continue
                    sample[side] = {"joint_measured_rad": arm["joint_positions_rad"],
                                    "eef_measured_xyz_xyzw": arm["tcp_xyzquat"],
                                    "gripper_measured": arm["gripper_normalized"]}
                    if "gripper_command_normalized" in arm:
                        sample[side]["gripper_command"] = arm["gripper_command_normalized"]
                    if "timestamp_s" in arm:
                        sample[side]["feedback_timestamp_s"] = arm["timestamp_s"]
                        sample[side]["feedback_timestamp_source"] = arm.get("timestamp_source", "sdk_clock_unspecified")
                self.samples.append(sample)
                self.state_times.append(t)
            if not self.samples:
                raise ValueError("video+action requires recorded measured states")
            if offsets and max(offsets) - min(offsets) > .1:
                raise ValueError("Host clock shifted during recording; state/video alignment is ambiguous")

    def event_pts(self):
        return sorted({self.pts[nearest_index(self.capture_times, t)] for t in event_times(self.samples)
                       if self.capture_times[0] <= t <= self.end_at})

    def verify_sources(self):
        if any(_sha256_file(self.root / name) != digest for name, digest in self.source_hashes.items()):
            raise RuntimeError("Recording changed during demonstration preparation")

    def attach(self, keyframes):
        """Join on host capture/read time, explicitly retaining mismatch and gaps."""
        for i, frame in enumerate(keyframes):
            n = frame["video_frame_index"]
            captured = self.capture_times[n]
            nearest = nearest_index(self.state_times, captured)
            delta = self.state_times[nearest] - captured
            frame["alignment"] = {"method": "nearest_host_read_time", "captured_at_s": captured,
                                  "state_observed_at_s": self.state_times[nearest], "delta_s": delta,
                                  "max_delta_s": .1, "exposure_synchronized": False}
            frame["state"] = {side: arm for side, arm in self.samples[nearest].items() if side != "t_s"} if abs(delta) <= .1 else None
            if i + 1 == len(keyframes):
                continue
            end = self.capture_times[keyframes[i+1]["video_frame_index"]]
            lo = bisect_left(self.state_times, captured)
            hi = bisect_left(self.state_times, end)
            # Include a sample at the upper boundary only when it actually matches.
            if hi < len(self.state_times) and self.state_times[hi] == end:
                hi += 1
            samples = [{**s, "t_s": s["t_s"] - self.capture_times[0]} for s in self.samples[lo:hi]]
            requests = [{**r, "at_s": r["at_s"] - self.capture_times[0]} for r in self.requests
                        if captured <= r["at_s"] < end]
            frame["action"] = {"kind": "measured_bimanual_segment", "time_basis": "host_elapsed_s",
                               "samples": samples, "requested_tools": requests,
                               "max_sample_gap_s": max((b["t_s"] - a["t_s"] for a, b in zip(samples, samples[1:])), default=0)}
