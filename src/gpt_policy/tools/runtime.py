"""Safe, explicit dispatch for configuration-selected robot tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .catalog import ToolCatalog


Handler = Callable[[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]], Any]


class ToolExecutor:
    """Bind configured handler IDs to a fixed set of host callbacks."""

    def __init__(
        self,
        catalog: ToolCatalog,
        arms: tuple[str, ...],
        robot: Any,
        localizer: Any,
    ) -> None:
        self.catalog = catalog
        self.arms = arms
        self._handlers: dict[str, Handler] = {
            "robot.move_to": lambda arguments, _state, _history: robot.execute(
                "move_to", arguments
            ),
            "robot.move_eef_chunk": lambda arguments, _state, _history: robot.execute(
                "move_eef_chunk", arguments
            ),
            "robot.check_path": lambda arguments, _state, _history: robot.execute(
                "check_path", arguments
            ),
            "robot.set_gripper": lambda arguments, _state, _history: robot.execute(
                "set_gripper", arguments
            ),
            "vision.locate_point": lambda arguments, state, history: localizer.locate(
                arguments, state, history
            ),
        }
        terminal_handlers = {"terminal.done", "terminal.give_up"}
        for tool in catalog.enabled_tools(arms):
            valid = tool.handler in (
                terminal_handlers if tool.terminal else self._handlers.keys()
            )
            if not valid:
                kind = "terminal" if tool.terminal else "executable"
                raise ValueError(
                    f"configured {kind} tool {tool.name} uses unknown handler: {tool.handler}"
                )

    def is_terminal(self, name: str) -> bool:
        try:
            return self.catalog.definition(name, self.arms).terminal
        except ValueError:
            # Let execute() turn an unknown model selection into the normal
            # recoverable tool error instead of aborting the run loop here.
            return False

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        state: dict[str, Any],
        history: dict[int, dict[str, Any]],
    ) -> Any:
        tool = self.catalog.definition(name, self.arms)
        if tool.terminal:
            raise ValueError(f"terminal tool must be handled by the run loop: {name}")
        return self._handlers[tool.handler](arguments, state, history)
