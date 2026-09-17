"""Small, project-local configuration loader."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    """Provider-independent runtime options loaded from the main config."""

    robot_model: str
    interface: str
    right_interface: str
    gripper_open_readout: float
    camera_width: int
    camera_height: int
    convert_camera_images_to_jpeg: bool
    camera_jpeg_quality: int
    trajectory_hz: float
    task_name: str
    record_dir: Path | None
    camera_overrides: list[str] | None = None
    input_json: Path | None = None
    max_decisions: int = 100


def settings_path(path: Path | None = None, machine: str | None = None) -> Path:
    root = Path(__file__).resolve().parents[2]
    if path is not None:
        return path.expanduser().resolve()
    if machine is not None:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", machine):
            raise ValueError(f"unknown machine: {machine}")
        return root / "configs" / "machines" / f"{machine}.json"
    if configured := (os.environ.get("GPT_POLICY_CONFIG")
                      or os.environ.get("ROBOT_GPT_CONFIG")
                      or os.environ.get("ARX_GPT_CONFIG")):
        return Path(configured).expanduser().resolve()
    selected = root / "configs" / "machine.local.json"
    if selected.exists():
        return selected.resolve()
    local = root / "configs" / "default.local.json"
    return (local if local.exists() else root / "configs" / "default.json").resolve()


def load_settings(path: Path | None = None) -> dict[str, Any]:
    return _load_settings(settings_path(path), ())


def _load_settings(filename: Path, parents: tuple[Path, ...]) -> dict[str, Any]:
    filename = filename.resolve()
    if filename in parents:
        raise ValueError(f"configuration inheritance cycle: {filename}")
    with filename.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("配置文件根节点必须是对象")
    inherits = value.pop("extends", [])
    if isinstance(inherits, str):
        inherits = [inherits]
    if not isinstance(inherits, list) or not all(isinstance(p, str) for p in inherits):
        raise ValueError("extends must be a path or list of paths")
    merged: dict[str, Any] = {}
    for parent in inherits:
        merged = _merge(merged, _load_settings(filename.parent / parent, (*parents, filename)))
    # Resolve file-valued fields where they are defined, not in the child file.
    for key in ("agent_config_dir",):
        if key in value:
            value[key] = str((filename.parent / value[key]).resolve())
    for key in ("input_json", "record_dir"):
        if value.get("runtime", {}).get(key):
            value["runtime"][key] = str((filename.parent / value["runtime"][key]).resolve())
    return _merge(merged, value)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def runtime_config(
    settings: dict[str, Any],
    legacy: Any | None = None,
    base_dir: Path | None = None,
) -> RuntimeConfig:
    """Resolve runtime options from ``settings.runtime``.

    ``legacy`` is only used by older callers/tests that construct an argument
    namespace directly. New command-line parsing does not expose these fields.
    """
    values = settings.get("runtime", {})
    if not isinstance(values, dict):
        raise ValueError("配置中的 runtime 必须是对象")
    agent_fields = set(values) & {"model", "effort", "codex_bin"}
    if agent_fields:
        raise ValueError(
            f"agent 专属字段不能放在 runtime 中: {sorted(agent_fields)}; "
            "请移动到 configs/agents/<agent>.json"
        )

    def get(name: str, default: Any) -> Any:
        if name in values:
            return values[name]
        if legacy is not None and hasattr(legacy, name):
            return getattr(legacy, name)
        return default

    robot_model = _string(get("robot_model", "X5"), "runtime.robot_model")
    interface = _string(get("interface", "can1"), "runtime.interface")
    right_interface_value = get("right_interface", "can3")
    if right_interface_value is None:
        right_interface = ""
    elif isinstance(right_interface_value, str) and not right_interface_value.strip():
        right_interface = ""
    else:
        right_interface = _string(right_interface_value, "runtime.right_interface")
    task_name = _task_name(get("task_name", "task"))

    record_value = get("record_dir", None)
    record_dir = None if record_value in (None, "") else Path(str(record_value)).expanduser()

    if "input_json" in values:
        input_value = values["input_json"]
    elif legacy is not None and getattr(legacy, "input_json", None) is not None:
        input_value = legacy.input_json
    else:
        input_value = settings.get("input_json")
    input_json = _config_path(input_value, base_dir, "runtime.input_json")

    camera_value = get("camera_overrides", get("camera", None))
    if camera_value is None:
        camera_overrides = None
    elif isinstance(camera_value, list) and all(isinstance(item, str) for item in camera_value):
        camera_overrides = list(camera_value)
    else:
        raise ValueError("runtime.camera_overrides 必须是字符串数组")

    return RuntimeConfig(
        robot_model=robot_model,
        interface=interface,
        right_interface=right_interface,
        gripper_open_readout=_number(
            get("gripper_open_readout", -3.4), "runtime.gripper_open_readout"
        ),
        camera_width=_positive_int(get("camera_width", 640), "runtime.camera_width"),
        camera_height=_positive_int(get("camera_height", 480), "runtime.camera_height"),
        convert_camera_images_to_jpeg=_boolean(
            get("convert_camera_images_to_jpeg", False),
            "runtime.convert_camera_images_to_jpeg",
        ),
        camera_jpeg_quality=_bounded_int(
            get("camera_jpeg_quality", 85), "runtime.camera_jpeg_quality", 1, 95
        ),
        trajectory_hz=_positive_number(get("trajectory_hz", 100.0), "runtime.trajectory_hz"),
        task_name=task_name,
        record_dir=record_dir,
        camera_overrides=camera_overrides,
        input_json=input_json,
        max_decisions=_positive_int(get("max_decisions", 100), "runtime.max_decisions"),
    )


def camera_defaults(settings: dict[str, Any]) -> list[str]:
    cameras = settings.get("cameras", {})
    return [f"{name}={data.get('device', data.get('serial'))}" for name, data in cameras.items()]


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value.strip()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是数字")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字") from exc


def _positive_number(value: Any, name: str) -> float:
    result = _number(value, name)
    if result <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _number(value, name)
    if not result.is_integer() or result <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return int(result)


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    result = _number(value, name)
    if not result.is_integer() or not minimum <= result <= maximum:
        raise ValueError(f"{name} 必须是 {minimum} 到 {maximum} 之间的整数")
    return int(result)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是布尔值")
    return value


def _task_name(value: Any) -> str:
    task_name = _string(value, "runtime.task_name").lower()
    task_name = re.sub(r"[\s_]+", "-", task_name)
    if len(task_name) > 64 or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", task_name):
        raise ValueError("runtime.task_name 须为英文或数字，可用短横线分隔，最长 64 字符")
    return task_name


def _config_path(value: Any, base_dir: Path | None, name: str) -> Path | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} 必须是路径字符串")
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()
