"""Load model-visible tools from JSON while keeping schema generation strict."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CATALOG = _PROJECT_ROOT / "configs" / "tools.json"
_MODES = frozenset({"single", "bimanual"})
_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    handler: str
    modes: frozenset[str]
    enabled: bool
    terminal: bool
    description: str
    prompt: str
    parameters: Mapping[str, Any] | None
    parameters_by_mode: Mapping[str, Mapping[str, Any]] | None


class ToolCatalog:
    """Validated, immutable-enough view of the configured model tool catalog."""

    def __init__(self, data: Mapping[str, Any], source: str = "inline") -> None:
        if data.get("version") != 1:
            raise ValueError(f"tool catalog {source} must have version 1")
        schemas = data.get("schemas")
        selection_description = data.get("selection_description")
        raw_tools = data.get("tools")
        if not isinstance(schemas, dict) or not schemas:
            raise ValueError(f"tool catalog {source} schemas must be a non-empty object")
        if not isinstance(selection_description, str) or not selection_description.strip():
            raise ValueError(f"tool catalog {source} selection_description must be text")
        if not isinstance(raw_tools, list) or not raw_tools:
            raise ValueError(f"tool catalog {source} tools must be a non-empty array")

        self.source = source
        self._schemas = deepcopy(schemas)
        self._selection_description = selection_description
        definitions: list[ToolDefinition] = []
        seen: set[str] = set()
        for index, value in enumerate(raw_tools):
            definition = self._parse_tool(value, index)
            if definition.name in seen:
                raise ValueError(f"duplicate configured tool: {definition.name}")
            seen.add(definition.name)
            definitions.append(definition)
        self._definitions = tuple(definitions)

    @staticmethod
    def _parse_tool(value: Any, index: int) -> ToolDefinition:
        if not isinstance(value, dict):
            raise ValueError(f"tool catalog tools[{index}] must be an object")
        name = value.get("name")
        handler = value.get("handler")
        modes_value = value.get("modes", ["single", "bimanual"])
        description = value.get("description")
        prompt = value.get("prompt")
        enabled = value.get("enabled", True)
        terminal = value.get("terminal", False)
        parameters = value.get("parameters")
        parameters_by_mode = value.get("parameters_by_mode")
        if not isinstance(name, str) or _NAME.fullmatch(name) is None:
            raise ValueError(f"tool catalog tools[{index}].name is invalid")
        if not isinstance(handler, str) or not handler:
            raise ValueError(f"configured tool {name} must name a handler")
        if not isinstance(modes_value, list) or not modes_value:
            raise ValueError(f"configured tool {name} modes must be a non-empty array")
        if not all(isinstance(mode, str) for mode in modes_value):
            raise ValueError(f"configured tool {name} has an invalid mode")
        modes = frozenset(modes_value)
        if not modes <= _MODES:
            raise ValueError(f"configured tool {name} has an invalid mode")
        if not isinstance(enabled, bool) or not isinstance(terminal, bool):
            raise ValueError(f"configured tool {name} enabled/terminal must be boolean")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"configured tool {name} must have a description")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"configured tool {name} must have prompt text")
        if (parameters is None) == (parameters_by_mode is None):
            raise ValueError(
                f"configured tool {name} must define exactly one of parameters or parameters_by_mode"
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError(f"configured tool {name} parameters must be an object")
        if parameters_by_mode is not None:
            if not isinstance(parameters_by_mode, dict):
                raise ValueError(
                    f"configured tool {name} parameters_by_mode must be an object"
                )
            missing = modes - set(parameters_by_mode)
            if missing or not all(
                isinstance(parameters_by_mode.get(mode), dict) for mode in modes
            ):
                raise ValueError(
                    f"configured tool {name} lacks parameters for {sorted(missing)[0] if missing else 'a mode'}"
                )
        return ToolDefinition(
            name=name,
            handler=handler,
            modes=modes,
            enabled=enabled,
            terminal=terminal,
            description=description,
            prompt=prompt,
            parameters=deepcopy(parameters),
            parameters_by_mode=deepcopy(parameters_by_mode),
        )

    def enabled_tools(self, arms: tuple[str, ...]) -> tuple[ToolDefinition, ...]:
        mode = _mode(arms)
        tools = tuple(
            definition
            for definition in self._definitions
            if definition.enabled and mode in definition.modes
        )
        if not tools:
            raise ValueError(f"tool catalog {self.source} enables no tools for {mode} mode")
        return tools

    def definition(self, name: str, arms: tuple[str, ...]) -> ToolDefinition:
        for definition in self.enabled_tools(arms):
            if definition.name == name:
                return definition
        raise ValueError(f"tool is not enabled for {_mode(arms)} mode: {name}")

    def output_schema(self, dof: int, arms: tuple[str, ...]) -> dict[str, Any]:
        tools = self.enabled_tools(arms)
        # Reuse each tool's contract instead of generating unrelated null fields.
        arguments = []
        for tool in self.function_schemas(dof, arms):
            parameters = _strict_parameters(tool["function"]["parameters"])
            if parameters not in arguments:
                arguments.append(parameters)
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": [tool.name for tool in tools]},
                "arguments": {"anyOf": arguments},
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
            "description": self._format(self._selection_description, dof),
        }

    def function_schemas(
        self, dof: int, arms: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        mode = _mode(arms)
        result = []
        for tool in self.enabled_tools(arms):
            parameters = (
                tool.parameters
                if tool.parameters is not None
                else tool.parameters_by_mode[mode]  # type: ignore[index]
            )
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": self._format(tool.description, dof),
                        "parameters": self._resolve(parameters, dof, arms),
                    },
                }
            )
        return result

    def prompt_catalog(
        self, dof: int, arms: tuple[str, ...], *, overrides: Mapping[str, str] | None = None
    ) -> str:
        return "\n".join(
            f"- {tool.name}: {self._format((overrides or {}).get(tool.name, tool.prompt), dof)}"
            for tool in self.enabled_tools(arms)
        )

    def _resolve(self, value: Any, dof: int, arms: tuple[str, ...]) -> Any:
        if isinstance(value, list):
            return [self._resolve(item, dof, arms) for item in value]
        if not isinstance(value, dict):
            return self._format(value, dof) if isinstance(value, str) else deepcopy(value)
        if set(value) == {"$schema"}:
            name = value["$schema"]
            if not isinstance(name, str) or name not in self._schemas:
                raise ValueError(f"tool catalog {self.source} references unknown schema: {name}")
            return self._resolve(self._schemas[name], dof, arms)
        if set(value) == {"$template"}:
            return self._dynamic_schema(value["$template"], dof, arms)
        return {
            key: self._resolve(item, dof, arms)
            for key, item in value.items()
        }

    def _dynamic_schema(self, name: Any, dof: int, arms: tuple[str, ...]) -> dict[str, Any]:
        if name == "target_pose":
            pose = self._resolve({"$schema": "pose"}, dof, arms)
            if len(arms) == 1:
                return pose
            return {
                "type": "object",
                "properties": {
                    arm: {"anyOf": [deepcopy(pose), {"type": "null"}]}
                    for arm in arms
                },
                "required": list(arms),
                "additionalProperties": False,
                "description": (
                    "One absolute fingertip TCP pose per selected arm; "
                    "null means hold that arm."
                ),
            }
        if name == "gripper_positions":
            nullable = self._resolve({"$schema": "nullable_number"}, dof, arms)
            return {
                "type": "object",
                "properties": {arm: deepcopy(nullable) for arm in arms},
                "required": list(arms),
                "additionalProperties": False,
            }
        if name == "camera_name":
            return {"type": "string", "enum": [*arms, "top"]}
        raise ValueError(f"tool catalog {self.source} references unknown template: {name}")

    @staticmethod
    def _format(value: str, dof: int) -> str:
        return value.replace("{dof}", str(dof))


def load_tool_catalog(settings: Mapping[str, Any] | None = None) -> ToolCatalog:
    """Load the default catalog, an alternate JSON path, or an inline catalog."""
    configured = (settings or {}).get("tool_catalog")
    if isinstance(configured, dict):
        return ToolCatalog(configured, "settings.tool_catalog")
    if configured is None:
        path = _DEFAULT_CATALOG
    elif isinstance(configured, str) and configured.strip():
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
    else:
        raise ValueError("tool_catalog must be a JSON path or an inline object")
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"tool catalog {path} root must be an object")
    return ToolCatalog(data, str(path))


def _strict_parameters(schema: Any) -> Any:
    """Codex requires all object fields; optional tool parameters become nullable."""
    if isinstance(schema, list):
        return [_strict_parameters(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {key: _strict_parameters(value) for key, value in schema.items()}
    if result.get("type") == "object":
        properties = result.get("properties", {})
        required = result.get("required", [])
        for name in properties:
            if name not in required:
                properties[name] = {"anyOf": [properties[name], {"type": "null"}]}
        result["required"] = list(properties)
        result["additionalProperties"] = False
    return result


def _mode(arms: tuple[str, ...]) -> str:
    if len(arms) == 1:
        return "single"
    if len(arms) == 2:
        return "bimanual"
    raise ValueError("tool catalog supports exactly one or two arms")
