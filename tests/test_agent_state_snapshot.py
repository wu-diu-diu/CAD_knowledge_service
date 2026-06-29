from __future__ import annotations

import numpy as np

from agent.state import RoomAgentState


def test_snapshot_restore_recovers_layout_plan_and_cache() -> None:
    state = RoomAgentState(room_name="测试房间", area_m2=12.0, matrix=np.ones((4, 4), dtype=np.int32))
    state.placements["lamps"] = [[1, 1]]
    state.lamp_plan = {"lamp_count": 1}
    state.tool_cache["component_count_plan"] = {"lamp_count": 1}

    snapshot_id = state.snapshot("before_edit", "test")
    state.placements["lamps"] = [[2, 2], [3, 3]]
    state.lamp_plan = {"lamp_count": 2}
    state.tool_cache["component_count_plan"] = {"lamp_count": 2}

    assert state.restore(snapshot_id) is True

    assert state.placements["lamps"] == [[1, 1]]
    assert state.lamp_plan == {"lamp_count": 1}
    assert state.tool_cache["component_count_plan"] == {"lamp_count": 1}
    assert state.list_snapshots()[0]["snapshot_id"] == snapshot_id
