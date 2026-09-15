"""Continuous measured state recording, separate from planned trajectories."""

from __future__ import annotations

import json
import math
import threading
import time


class StateRecorder:
    """Sample feedback without accumulating a memory queue or blocking control.

    timestamp_s inside each arm is the SDK feedback timestamp. observed_at_s
    is the host read time. Sampling is best effort, never labelled native CAN Hz.
    """

    def __init__(self, robot, directory, hz=50.0):
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError("recording.state_hz must be finite and positive")
        self.robot, self.hz = robot, hz
        self.path = directory / "states.jsonl"
        self.stream = self.path.open("x", encoding="utf-8", buffering=1)
        self._stop = threading.Event()
        self.error = None
        self.errors = 0
        self.samples = 0
        self.overruns = 0
        self.thread = threading.Thread(target=self._run, name="robot-state-recorder", daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop.is_set():
            start = time.monotonic()
            try:
                state = self.robot.state()
                self.stream.write(json.dumps({
                    "observed_at_s": time.time(), "observed_monotonic_s": time.monotonic(),
                    "state": state,
                }, allow_nan=False) + "\n")
                self.samples += 1
            except Exception as exc:
                self.error = exc
                self.errors += 1
                self.stream.write(json.dumps({
                    "observed_at_s": time.time(),
                    "observed_monotonic_s": time.monotonic(),
                    "error": repr(exc),
                }, allow_nan=False) + "\n")
            elapsed = time.monotonic() - start
            self.overruns += int(elapsed > 1 / self.hz)
            self._stop.wait(max(0.0, 1 / self.hz - elapsed))

    def check(self):
        return None

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise TimeoutError("State recording did not stop within 5 seconds")
        self.stream.close()
        result = {
            "path": self.path.name,
            "samples": self.samples,
            "requested_hz": self.hz,
            "overruns": self.overruns,
            "errors": self.errors,
        }
        if self.error is not None:
            result["error"] = repr(self.error)
        return result
