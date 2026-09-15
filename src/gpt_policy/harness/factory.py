"""Explicit provider registry; importing the application never starts an agent."""

from importlib import import_module
import shutil

from .config import AgentConfig
from .contract import AgentSession


_PROVIDERS = {
    "codex": (".providers.codex", "CodexSession"),
    "claude_code": (".providers.claude_code.session", "ClaudeCodeSession"),
    "gemini_cli": (".providers.gemini_cli.session", "GeminiCliSession"),
}


def preflight_agent(config: AgentConfig) -> None:
    """Check new-provider dependencies before opening robot hardware."""
    if config.type not in _PROVIDERS:
        raise ValueError(f"不支持的 agent.type: {config.type}")
    if config.type != "codex":
        if shutil.which(config.executable) is None:
            raise ValueError(f"找不到 {config.type} 可执行文件: {config.executable}")
        _provider(config.type)


def create_agent(
    config: AgentConfig, model: str, convert_images: bool = False, jpeg_quality: int = 85,
) -> AgentSession:
    return _provider(config.type)(config, model, convert_images, jpeg_quality)


def _provider(kind: str):
    if kind not in _PROVIDERS:
        raise ValueError(f"不支持的 agent.type: {kind}")
    module, name = _PROVIDERS[kind]
    return getattr(import_module(module, __package__), name)
