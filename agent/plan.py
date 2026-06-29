from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class PlanPhase:
    name: str
    expected_tools: List[str]
    done: bool = False


@dataclass
class Reflection:
    phase: str
    tool: str
    ok: bool
    message: str
    next_hint: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AgentPlan:
    def __init__(self) -> None:
        self.phases = [
            PlanPhase("需求解析", ["tool_parse_user_requirement", "tool_query_design_standard", "tool_lookup_room_requirement"]),
            PlanPhase("选型计算", ["tool_estimate_component_count", "tool_calc_required_flux_per_lamp", "tool_retrieve_lamp_model"]),
            PlanPhase("布局", ["tool_place_components"]),
            PlanPhase("校验修正", ["tool_validate_layout", "tool_diagnose_layout_issue", "tool_apply_layout_edit"]),
            PlanPhase("布线", ["tool_generate_wiring"]),
            PlanPhase("合规报告", ["tool_check_standard_compliance", "tool_generate_report"]),
        ]

    def phase_for_tool(self, tool_name: str) -> str:
        for phase in self.phases:
            if tool_name in phase.expected_tools:
                return phase.name
        if tool_name == "finish":
            return "完成"
        return "未知"

    def reflect(self, tool_name: str, tool_output: Dict[str, Any]) -> Reflection:
        phase = self.phase_for_tool(tool_name)
        status = str(tool_output.get("status", "ok") if isinstance(tool_output, dict) else "ok")
        ok = status not in {"error", "failed"} and tool_name != "__unknown_tool__"
        if isinstance(tool_output, dict) and tool_output.get("violations"):
            ok = False
        next_hint = "继续推进下一阶段" if ok else "根据 Observation 修正后重试或回滚"
        message = f"{tool_name} executed in phase {phase}, ok={ok}"
        return Reflection(phase=phase, tool=tool_name, ok=ok, message=message, next_hint=next_hint)
