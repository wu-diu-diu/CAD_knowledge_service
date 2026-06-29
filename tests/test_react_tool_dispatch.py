from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from agent.base import LightingAgent
from agent.config import ProviderConfig
from agent.model_clients import BaseModelClient
from agent.state import RoomAgentState


class FakeModelClient(BaseModelClient):
    def __init__(self, actions: List[Dict[str, Any]]) -> None:
        self.actions = list(actions)
        super().__init__(
            ProviderConfig(
                name="fake",
                api_key="",
                base_url="",
                model="fake-model",
                api_type="fake",
            )
        )

    def complete_action(self, **_: Any) -> Dict[str, Any]:
        if self.actions:
            return self.actions.pop(0)
        return {"action": "finish", "reason": "done", "strategy": "test"}


def test_unified_agent_dispatches_all_declared_tools() -> None:
    actions = [
        {"thought": "parse", "action": "tool_parse_user_requirement", "args": {"user_input": "办公室 300lux 2盏灯"}},
        {
            "thought": "standard",
            "action": "tool_query_design_standard",
            "args": {"query": "办公室照度标准", "kg_store_dir": "/tmp/nonexistent-agent-kg"},
        },
        {"thought": "room", "action": "tool_lookup_room_requirement", "args": {"room_name": "办公室"}},
        {"thought": "count", "action": "tool_estimate_component_count", "args": {"lamp_count": 2}},
        {"thought": "flux", "action": "tool_calc_required_flux_per_lamp", "args": {"target_lux": 300, "lamp_count": 2}},
        {"thought": "model", "action": "tool_retrieve_lamp_model", "args": {"lamp_type": "筒灯"}},
        {"thought": "place", "action": "tool_place_components", "args": {"lamp_count": 2, "switch_count": 1}},
        {"thought": "validate", "action": "tool_validate_layout", "args": {}},
        {"thought": "diagnose", "action": "tool_diagnose_layout_issue", "args": {}},
        {"thought": "edit", "action": "tool_apply_layout_edit", "args": {"edits": []}},
        {"thought": "read", "action": "tool_read_matrix_state", "args": {}},
        {"thought": "wire", "action": "tool_generate_wiring", "args": {}},
        {"thought": "summary", "action": "tool_summarize_design", "args": {}},
        {
            "thought": "compliance",
            "action": "tool_check_standard_compliance",
            "args": {"kg_store_dir": "/tmp/nonexistent-agent-kg"},
        },
        {"thought": "report", "action": "tool_generate_report", "args": {}},
        {"thought": "finish", "action": "finish", "reason": "done", "strategy": "test"},
    ]
    matrix = np.ones((8, 8), dtype=np.int32)
    matrix[0, 3] = 2
    state = RoomAgentState(room_name="办公室", area_m2=20.0, matrix=matrix)
    agent = LightingAgent(model_client=FakeModelClient(actions), init_mode="llm")

    result = agent.run_for_room(state, max_steps=20, generate_wiring=True)
    called = {item["tool"] for item in state.tool_history}

    expected_tools = {
        "tool_parse_user_requirement",
        "tool_query_design_standard",
        "tool_lookup_room_requirement",
        "tool_estimate_component_count",
        "tool_calc_required_flux_per_lamp",
        "tool_retrieve_lamp_model",
        "tool_place_components",
        "tool_validate_layout",
        "tool_diagnose_layout_issue",
        "tool_apply_layout_edit",
        "tool_read_matrix_state",
        "tool_generate_wiring",
        "tool_summarize_design",
        "tool_check_standard_compliance",
        "tool_generate_report",
    }
    assert result["finish_reason"] == "done"
    assert expected_tools.issubset(called)
    assert not any(str(item["tool"]).startswith("unhandled") for item in state.tool_history)
