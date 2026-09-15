"""Small, provider-independent values exchanged with a robot decision agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, TypedDict

from ..input.manifest import ContentPart


class ImageInput(Protocol):
    """Structural interface already implemented by CapturedImage."""

    data: bytes
    mime_type: str
    width: int
    height: int
    rgb_data: bytes | None

    def data_url(self) -> str: ...


@dataclass(frozen=True)
class AgentContext:
    instructions: str
    tools: list[dict[str, Any]]
    output_schema: dict[str, Any]


@dataclass(frozen=True)
class AgentTurn:
    observation: str
    images: Mapping[str, ImageInput] | None = None
    content: tuple[ContentPart, ...] | None = None


class AgentDecision(TypedDict):
    name: str
    arguments: dict[str, Any]
    _wire: Any
