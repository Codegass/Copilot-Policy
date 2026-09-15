"""One persistent Claude process, with no native tool execution."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path

from ...config import AgentConfig
from ...errors import AgentProtocolError
from ...media import content_blocks
from ...models import AgentContext, AgentDecision, AgentTurn
from ...process import JsonProcess
from ...prompts import provider_instructions
from ...validation import parse_decision
from ...usage import model_call
from .codec import user_message

_CONFLICTING_ENVIRONMENT = (
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
    "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
    "CLAUDE_CODE_SUBAGENT_MODEL",
)


class ClaudeCodeSession:
    def __init__(
        self, config: AgentConfig, model: str, convert_images: bool, jpeg_quality: int,
    ) -> None:
        self.config, self.model = config, model
        self.convert_images, self.jpeg_quality = convert_images, jpeg_quality
        self.context: AgentContext | None = None
        self.process: JsonProcess | None = None
        self.workspace: tempfile.TemporaryDirectory | None = None

    def start(self, context: AgentContext) -> None:
        if self.process is not None:
            raise AgentProtocolError("Claude session 已启动")
        self.context = context
        self.workspace = tempfile.TemporaryDirectory(prefix="robot-claude-")
        command = [
            self.config.executable, "--print", "--verbose", "--safe-mode",
            "--input-format", "stream-json", "--output-format", "stream-json",
            "--model", self.model, "--system-prompt", provider_instructions(context, "Claude Code"),
            "--json-schema", json.dumps(context.output_schema, ensure_ascii=False),
            "--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--permission-mode", "dontAsk", "--disable-slash-commands",
            "--no-session-persistence", "--no-chrome",
        ]
        if self.config.effort is not None:
            command.extend(["--effort", self.config.effort])
        environment = os.environ.copy()
        for name in tuple(environment):
            if name.startswith("ANTHROPIC_") or name in _CONFLICTING_ENVIRONMENT:
                environment.pop(name, None)
        environment.update(self.config.environment)
        try:
            # User settings can override injected routing/auth even in safe mode.
            # Give every session its own config directory as well as its own cwd.
            config_dir = Path(self.workspace.name) / ".claude"
            config_dir.mkdir(mode=0o700)
            environment["CLAUDE_CONFIG_DIR"] = str(config_dir)
            self.process = JsonProcess(command, Path(self.workspace.name), env=environment)
        except BaseException:
            self.close()
            raise

    def decide(self, turn: AgentTurn) -> AgentDecision:
        with model_call("claude_code", self.model) as call:
            return self._decide(turn, call)

    def _decide(self, turn, call):
        if self.process is None or self.context is None:
            raise AgentProtocolError("Claude session 尚未启动")
        assert self.config.timeout_s is not None
        deadline = time.monotonic() + self.config.timeout_s
        messages = {}
        try:
            blocks = content_blocks(turn, self.convert_images, self.jpeg_quality)
            call["request_started"] = True
            self.process.send(user_message(blocks), deadline)
            while True:
                event = self.process.receive(deadline)
                kind = event.get("type")
                if kind == "control_request":
                    self.process.send({"type": "control_response", "response": {
                        "subtype": "error", "request_id": event.get("request_id"),
                        "error": "Only robot decisions are supported",
                    }}, deadline)
                    raise AgentProtocolError("Claude 请求了未启用的宿主交互")
                if kind == "assistant":
                    message = event.get("message", {})
                    if message.get("id") and isinstance(message.get("usage"), dict):
                        previous = messages.get(message["id"], {})
                        messages[message["id"]] = _max_usage(previous, message["usage"])
                        call["raw_usage"] = _sum_usage(messages.values())
                    for block in event.get("message", {}).get("content", []):
                        if block.get("type") == "tool_use" and block.get("name") != "StructuredOutput":
                            raise AgentProtocolError(f"Claude 请求了未启用的工具: {block.get('name')}")
                if kind == "result":
                    call["usage_final"] = True
                    # The final result is authoritative and includes retries;
                    # assistant message usage is only a fallback on interruption.
                    if isinstance(event.get("usage"), dict) and event["usage"]:
                        call["raw_usage"] = event["usage"]
                    call["session_id"] = event.get("session_id")
                    call["num_turns"] = event.get("num_turns")
                    reported = event.get("total_cost_usd")
                    if isinstance(reported, (int, float)) and not isinstance(reported, bool) and math.isfinite(reported) and reported >= 0:
                        call["provider_reported_cost_usd"] = reported
                    # CLI cost is also an estimate, especially through a gateway.
                    if event.get("is_error") or event.get("subtype") != "success":
                        raise AgentProtocolError(f"Claude turn failed: {event.get('errors', event.get('result'))}")
                    value = event.get("structured_output", event.get("result"))
                    return parse_decision(value, self.context)
        except BaseException:
            # A failed/incomplete turn must not contaminate a subsequent decision.
            self.close()
            raise

    def close(self) -> None:
        process, self.process = self.process, None
        try:
            if process is not None:
                process.close()
        finally:
            if self.workspace is not None:
                self.workspace.cleanup()
                self.workspace = None


def _sum_usage(values):
    result = {}
    for usage in values:
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                result[key] = result.get(key, 0) + value
            elif key == "cache_creation" and isinstance(value, dict):
                result[key] = _sum_usage([result.get(key, {}), value])
    return result


def _max_usage(previous, current):
    result = dict(previous)
    for key, value in current.items():
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[key] = max(result.get(key, 0), value)
        elif key == "cache_creation" and isinstance(value, dict):
            result[key] = _max_usage(result.get(key, {}), value)
    return result
