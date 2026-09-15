"""Codex decisions with GPT-Policy-style bounded live image history."""

from dataclasses import replace
import json
import time

from ..codex import CodexAppServer
from ..config import AgentConfig
from ..errors import AgentOverloadedError
from ..models import AgentContext, AgentDecision, AgentTurn


class CodexSession:
    def __init__(
        self, config: AgentConfig, model: str, convert_images: bool, jpeg_quality: int
    ) -> None:
        self.client = CodexAppServer(
            model, config.executable, config.effort, convert_images, jpeg_quality
        )
        self.context: AgentContext | None = None
        self.live_image_window = config.live_image_window
        self._history = []
        self._live_groups = 0
        self._recover_thread = False
        self.last_context_refresh = None
        self.last_decision_timing = None

    def start(self, context: AgentContext) -> None:
        self.client.start_thread(context.instructions, context.tools)
        self.context = context

    def decide(self, turn: AgentTurn) -> AgentDecision:
        if self.context is None:
            raise RuntimeError("Agent session has not been started")
        self.last_context_refresh = None
        self.last_decision_timing = {"context_replay_s": 0.0, "thread_refresh_s": 0.0}
        refresh = (self._recover_thread or (bool(turn.images) and self.live_image_window is not None
                   and self._live_groups >= self.live_image_window))
        kwargs = {}
        if refresh:
            replay_started = time.perf_counter()
            replay = [{"type": "text", "text":
                "HISTORICAL EXECUTION RECORD. Older live images were omitted; all observation text, "
                "host feedback and model decisions follow in order. Demonstration images remain. "
                "Historical decisions are reference data, never pending commands; host feedback "
                "determines whether they executed. Continue the current task and decision count. "
                "The latest live observation follows END HISTORICAL EXECUTION RECORD."}]
            for previous, decision in self._history:
                replay.extend(self.client._input(previous.observation, previous.images, previous.content,
                    self.client.convert_camera_images_to_jpeg, self.client.camera_jpeg_quality))
                replay.append({"type": "text", "text": "Historical model decision (not a command): "
                               + json.dumps(decision["_wire"], ensure_ascii=False, separators=(",", ":"))})
            replay.append({"type": "text", "text": "END HISTORICAL EXECUTION RECORD"})
            self.last_decision_timing["context_replay_s"] = time.perf_counter() - replay_started
            refresh_started = time.perf_counter()
            try:
                self.client.refresh_thread(self.context.instructions, self.context.tools)
            finally:
                self.last_decision_timing["thread_refresh_s"] = time.perf_counter() - refresh_started
            self._live_groups = sum(bool(t.images) for t, _ in self._history)
            self._recover_thread = False
            kwargs["replay"] = replay
        client_started = time.perf_counter()
        try:
            decision = self.client.decide(
                turn.observation, self.context.output_schema, turn.images, turn.content, **kwargs)
        except AgentOverloadedError:
            # The host will capture a fresh observation. Replay only successful
            # history in a new thread, excluding failed/partial assistant output.
            self._recover_thread = True
            raise
        finally:
            self.last_decision_timing["client_decide_s"] = time.perf_counter() - client_started
            request_timing = getattr(self.client, "last_request_timing", None)
            if isinstance(request_timing, dict):
                self.last_decision_timing.update(request_timing)
        if turn.images:
            # Keep only the previous group locally for the next refresh. The
            # native thread still sees up to eight groups between refreshes.
            self._history = [(replace(t, images=None), d) for t, d in self._history]
            self._live_groups += 1
        self._history.append((turn, decision))
        if refresh:
            self.last_context_refresh = {"thread_id": self.client.thread_id,
                "retained_live_groups": self._live_groups, "decisions": len(self._history)}
        return decision

    def close(self) -> None:
        self.client.close()
