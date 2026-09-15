"""One session per run; only the host executes returned robot decisions."""

from typing import Protocol

from .models import AgentContext, AgentDecision, AgentTurn


class AgentSession(Protocol):
    def start(self, context: AgentContext) -> None: ...

    def decide(self, turn: AgentTurn) -> AgentDecision: ...

    def close(self) -> None: ...
