"""Persistent Gemini ACP session with images and host-validated decisions."""

from __future__ import annotations

import time

from ...config import AgentConfig
from ...errors import AgentProtocolError
from ...media import content_blocks
from ...models import AgentContext, AgentDecision, AgentTurn
from ...process import JsonProcess
from ...validation import parse_decision
from .acp import AcpConnection
from .codec import prompt_blocks
from .workspace import GeminiWorkspace


class GeminiCliSession:
    def __init__(
        self, config: AgentConfig, model: str, convert_images: bool, jpeg_quality: int,
    ) -> None:
        self.config, self.model = config, model
        self.convert_images, self.jpeg_quality = convert_images, jpeg_quality
        self.context: AgentContext | None = None
        self.process: JsonProcess | None = None
        self.connection: AcpConnection | None = None
        self.workspace: GeminiWorkspace | None = None
        self.session_id: str | None = None

    def start(self, context: AgentContext) -> None:
        if self.process is not None:
            raise AgentProtocolError("Gemini session 已启动")
        self.context = context
        self.workspace = GeminiWorkspace(context)
        assert self.config.timeout_s is not None
        deadline = time.monotonic() + self.config.timeout_s
        try:
            self.process = JsonProcess([
                self.config.executable, "--acp", "--model", self.model,
                "--extensions", "none", "--allowed-mcp-server-names", "__robot_no_mcp__",
                "--admin-policy", str(self.workspace.policy),
            ], self.workspace.path, self.workspace.environment())
            self.connection = AcpConnection(self.process)
            initialized = self.connection.request("initialize", {
                "protocolVersion": 1, "clientInfo": {"name": "gpt-policy", "version": "0.1.0"},
                "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False},
            }, deadline)
            if initialized.get("protocolVersion") != 1:
                raise AgentProtocolError("不支持的 Gemini ACP 协议版本")
            capabilities = initialized.get("agentCapabilities", {}).get("promptCapabilities", {})
            if not capabilities.get("image"):
                raise AgentProtocolError("Gemini ACP 不支持图片输入")
            created = self.connection.request("session/new", {
                "cwd": str(self.workspace.path), "mcpServers": [],
            }, deadline)
            session_id = created.get("sessionId")
            if not isinstance(session_id, str) or not session_id:
                raise AgentProtocolError("Gemini 未返回有效的 sessionId")
            self.session_id = session_id
        except BaseException:
            self.close()
            raise

    def decide(self, turn: AgentTurn) -> AgentDecision:
        if self.connection is None or self.session_id is None or self.context is None:
            raise AgentProtocolError("Gemini session 尚未启动")
        assert self.config.timeout_s is not None
        chunks: list[str] = []

        def update(event: dict) -> None:
            if event.get("method") != "session/update":
                return
            params = event.get("params", {})
            if params.get("sessionId") != self.session_id:
                return
            payload = params.get("update", {})
            kind = payload.get("sessionUpdate")
            if kind in {"tool_call", "tool_call_update"}:
                raise AgentProtocolError("Gemini 请求了未启用的原生工具")
            if kind == "agent_message_chunk":
                content = payload.get("content", {})
                if content.get("type") != "text" or not isinstance(content.get("text"), str):
                    raise AgentProtocolError("Gemini 决策输出必须是文字 JSON")
                chunks.append(content["text"])

        try:
            blocks = content_blocks(turn, self.convert_images, self.jpeg_quality)
            result = self.connection.request("session/prompt", {
                "sessionId": self.session_id, "prompt": prompt_blocks(blocks),
            }, time.monotonic() + self.config.timeout_s, update)
            if result.get("stopReason") != "end_turn":
                raise AgentProtocolError(f"Gemini turn failed: {result.get('stopReason')}")
            return parse_decision("".join(chunks), self.context)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        process, self.process = self.process, None
        self.connection = None
        self.session_id = None
        try:
            if process is not None:
                process.close()
        finally:
            if self.workspace is not None:
                self.workspace.close()
                self.workspace = None
