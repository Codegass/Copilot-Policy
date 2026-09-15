"""Adapt the existing robot prompt without changing the Codex prompt bytes."""

import json

from .models import AgentContext


def provider_instructions(context: AgentContext, provider: str) -> str:
    text = robot_instructions(context.instructions, provider)
    return text + "\n\nRobot tool catalog:\n" + json.dumps(context.tools, ensure_ascii=False)


def robot_instructions(text: str, provider: str) -> str:
    if provider == "codex":
        return text
    name = {"claude_code": "Claude Code", "gemini_cli": "Gemini CLI"}.get(provider, provider)
    return text.replace("native Codex harness", f"{name} harness")
