"""Synchronous ACP requests with explicit rejection of host-side operations."""

from collections.abc import Callable
from typing import Any

from ...errors import AgentProtocolError
from ...process import JsonProcess


class AcpConnection:
    def __init__(self, process: JsonProcess) -> None:
        self.process = process
        self._next_id = 0

    def request(
        self, method: str, params: dict, deadline: float,
        notification: Callable[[dict], None] | None = None,
    ) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self.process.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}, deadline)
        while True:
            event = self.process.receive(deadline)
            if "method" in event:
                if "id" in event:
                    self._reject(event, deadline)
                if notification is not None:
                    notification(event)
                continue
            if event.get("id") != request_id:
                raise AgentProtocolError("Gemini ACP 返回了不匹配的 request id")
            if "error" in event:
                raise AgentProtocolError(f"Gemini {method} failed: {event['error']}")
            result = event.get("result")
            if not isinstance(result, dict):
                raise AgentProtocolError(f"Gemini {method} 返回结果必须是对象")
            return result

    def _reject(self, event: dict, deadline: float) -> None:
        response: dict[str, Any] = {"jsonrpc": "2.0", "id": event["id"]}
        if event["method"] == "session/request_permission":
            response["result"] = {"outcome": {"outcome": "cancelled"}}
        else:
            response["error"] = {"code": -32601, "message": "Only robot decisions are supported"}
        self.process.send(response, deadline)
        raise AgentProtocolError(f"Gemini 请求了未启用的宿主操作: {event['method']}")
