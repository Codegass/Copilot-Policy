"""Strict host validation for providers without Codex's native output contract."""

from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from .errors import AgentProtocolError
from .models import AgentContext, AgentDecision


def parse_decision(value: Any, context: AgentContext) -> AgentDecision:
    try:
        selection = _json(value) if isinstance(value, str) else value
        if not isinstance(selection, dict):
            raise ValueError("工具选择必须是 JSON 对象")
        arguments = selection.get("arguments")
        if isinstance(arguments, str):
            arguments = _json(arguments)
        normalized = {**selection, "arguments": arguments}
        # jsonschema treats NaN as a number; reject non-finite values explicitly.
        json.dumps(normalized, allow_nan=False)
        Draft202012Validator(context.output_schema).validate(normalized)
        if not isinstance(arguments, dict) or not isinstance(selection.get("name"), str):
            raise ValueError("工具选择必须包含 name 字符串和 arguments 对象")
        definition = next((tool["function"] for tool in context.tools
                           if tool["function"]["name"] == selection["name"]), None)
        if definition is None:
            raise ValueError(f"未启用的工具: {selection['name']}")
        parameters = definition["parameters"]
        # Native structured output encodes optional parameters as null. Omit
        # only optional nulls; required nullable arm targets retain hold semantics.
        specific = {key: item for key, item in arguments.items()
                    if item is not None or key in parameters.get("required", [])}
        Draft202012Validator(parameters).validate(specific)
    except (ValueError, TypeError, ValidationError) as exc:
        message = exc.message if isinstance(exc, ValidationError) else str(exc)
        raise AgentProtocolError(f"Agent 工具选择无效: {message}") from exc
    return {"name": selection["name"], "arguments": arguments, "_wire": selection}


def _json(value: str) -> Any:
    text = value.strip()
    # Accept only a complete enclosing fence, never extract arbitrary fragments.
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0] in {"```", "```json"} and lines[-1] == "```":
        text = "\n".join(lines[1:-1])
    return json.loads(text)
