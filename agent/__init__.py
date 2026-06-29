"""Agent package for CAD lighting ReAct workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "RoomAgentState",
    "AgentStateManager",
    "LightingTools",
    "LightingAgent",
    "build_lighting_agent",
    "process_image_to_agent_input",
    "run_agent_pipeline",
]

_EXPORT_MAP = {
    "RoomAgentState": (".state", "RoomAgentState"),
    "AgentStateManager": (".state", "AgentStateManager"),
    "LightingTools": (".tools", "LightingTools"),
    "LightingAgent": (".base", "LightingAgent"),
    "build_lighting_agent": (".factory", "build_lighting_agent"),
    "process_image_to_agent_input": (".image_process", "process_image_to_agent_input"),
    "run_agent_pipeline": (".run", "run_agent_pipeline"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
