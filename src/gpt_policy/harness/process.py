"""Bounded JSON-line subprocess I/O used only by the new providers."""

from __future__ import annotations

import json
import os
import queue
import select
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping

from .errors import AgentError, AgentProtocolError, AgentTimeoutError
from .waiting import receive_event


class JsonProcess:
    def __init__(
        self, command: list[str], cwd: Path, env: Mapping[str, str] | None = None,
    ) -> None:
        self.process = subprocess.Popen(
            command, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
            start_new_session=True,
        )
        self._events: queue.Queue[dict[str, Any] | Exception] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=12)
        self._closed = False
        assert self.process.stdin is not None
        os.set_blocking(self.process.stdin.fileno(), False)
        self._readers = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for thread in self._readers:
            thread.start()

    def send(self, value: dict[str, Any], deadline: float | None = None) -> None:
        if self._closed:
            raise AgentError("Agent 进程已关闭")
        assert self.process.stdin is not None
        deadline = time.monotonic() + 30 if deadline is None else deadline
        data = memoryview((json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        descriptor = self.process.stdin.fileno()
        try:
            while data:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [descriptor], [], remaining)[1]:
                    raise AgentTimeoutError("写入 Agent 输入超时")
                try:
                    written = os.write(descriptor, data[:65536])
                except BlockingIOError:
                    continue
                data = data[written:]
        except (OSError, ValueError) as exc:
            raise AgentError(f"无法写入 Agent 进程: {exc}") from exc

    def receive(self, deadline: float) -> dict[str, Any]:
        if self._closed:
            raise AgentError("Agent 进程已关闭")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AgentTimeoutError("等待 Agent 响应超时")
        try:
            event = receive_event(self._events, deadline=deadline)
        except queue.Empty as exc:
            raise AgentTimeoutError("等待 Agent 响应超时") from exc
        if isinstance(event, Exception):
            detail = "".join(self._stderr)[-2000:].strip()
            raise AgentProtocolError(f"Agent 输出已中断: {event}; {detail}") from event
        return event

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._signal(signal.SIGTERM)
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._signal(signal.SIGKILL)
                self.process.wait(timeout=2)
        # Descendants may retain pipe handles even after their parent exits.
        self._signal(signal.SIGKILL)
        for thread in self._readers:
            thread.join(timeout=1)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()

    def _signal(self, number: int) -> None:
        try:
            os.killpg(self.process.pid, number)
        except ProcessLookupError:
            pass
        except PermissionError:
            # macOS may report EPERM for a process group whose leader has
            # just exited. Never hide a denied signal to a live child.
            if self.process.poll() is None:
                raise

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("Agent JSON line 必须是对象")
                    self._events.put(value)
        except Exception as exc:
            self._events.put(exc)
        finally:
            self._events.put(EOFError("Agent stdout 已关闭"))

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self._stderr.append(line[-2000:])
