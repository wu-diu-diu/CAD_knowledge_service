from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .logger import AgentRunLogger
from .utils import describe_tool_result, rle_encode


@dataclass
class RoomAgentState:
    room_name: str
    area_m2: float
    matrix: np.ndarray
    placements: Dict[str, List[List[int]]] = field(
        default_factory=lambda: {"lamps": [], "switches": []}
    )
    selected_lamp_type: Optional[str] = None
    lamp_plan: Optional[Dict[str, Any]] = None
    tool_cache: Dict[str, Any] = field(default_factory=dict)
    tool_history: List[Dict[str, Any]] = field(default_factory=list)
    tool_result_history: List[Dict[str, Any]] = field(default_factory=list)
    thought_history: List[Dict[str, Any]] = field(default_factory=list)
    snapshots: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    snapshot_order: List[str] = field(default_factory=list)
    logger: Optional[AgentRunLogger] = None

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, np.ndarray):
            self.matrix = np.array(self.matrix, dtype=np.int32)
        self.matrix = self.matrix.astype(np.int32)

    def record(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Dict[str, Any],
        tool_result: Optional[str] = None,
    ) -> None:
        resolved_tool_result = tool_result or describe_tool_result(tool_name, tool_output)
        record_item = {
            "tool": tool_name,
            "input": tool_input,
            "output": tool_output,
            "tool_result": resolved_tool_result,
        }
        self.tool_history.append(record_item)
        self.tool_result_history.append(
            {
                "tool": tool_name,
                "tool_result": resolved_tool_result,
            }
        )
        self.tool_cache["last_tool_result"] = {
            "tool": tool_name,
            "tool_result": resolved_tool_result,
        }
        if self.logger:
            self.logger.tool_io(tool_name, tool_input, resolved_tool_result)

    def to_ascii_board(self, max_rows: int = 64, max_cols: int = 64, compress: bool = True) -> str:
        rows, cols = self.matrix.shape
        row_idx = np.linspace(0, rows - 1, min(rows, max_rows), dtype=int)
        col_idx = np.linspace(0, cols - 1, min(cols, max_cols), dtype=int)

        sampled = self.matrix[np.ix_(row_idx, col_idx)].copy()
        lamp_cells = {(int(p[0]), int(p[1])) for p in self.placements.get("lamps", []) if len(p) == 2}
        switch_cells = {(int(p[0]), int(p[1])) for p in self.placements.get("switches", []) if len(p) == 2}

        lines: List[str] = []
        for rr, r in enumerate(row_idx):
            chars: List[str] = []
            for cc, c in enumerate(col_idx):
                if (int(r), int(c)) in switch_cells:
                    ch = "S"
                elif (int(r), int(c)) in lamp_cells:
                    ch = "L"
                else:
                    value = int(sampled[rr, cc])
                    if value == 2:
                        ch = "D"
                    elif value == 1:
                        ch = "."
                    else:
                        ch = "#"
                chars.append(ch)
            row_text = "".join(chars)
            lines.append(rle_encode(row_text) if compress else row_text)
        return "\n".join(lines)

    def summary(self) -> Dict[str, Any]:
        return {
            "room_name": self.room_name,
            "area_m2": float(self.area_m2),
            "matrix_shape": [int(self.matrix.shape[0]), int(self.matrix.shape[1])],
            "selected_lamp_type": self.selected_lamp_type,
            "lamp_plan": self.lamp_plan,
            "tool_cache_keys": sorted(list(self.tool_cache.keys())),
            "placements": self.placements,
            "tool_calls": len(self.tool_history),
            "tool_result_calls": len(self.tool_result_history),
            "thought_calls": len(self.thought_history),
            "snapshot_count": len(self.snapshot_order),
        }

    @staticmethod
    def _copy_tool_cache(tool_cache: Dict[str, Any]) -> Dict[str, Any]:
        copied: Dict[str, Any] = {}
        for key, value in (tool_cache or {}).items():
            if str(key).startswith("_"):
                continue
            try:
                copied[key] = deepcopy(value)
            except Exception:
                copied[key] = value
        return copied

    def snapshot(self, label: str = "", reason: str = "") -> str:
        snapshot_id = f"snap_{len(self.snapshot_order) + 1:04d}"
        self.snapshots[snapshot_id] = {
            "snapshot_id": snapshot_id,
            "label": str(label or ""),
            "reason": str(reason or ""),
            "created_at": datetime.now().isoformat(),
            "placements": deepcopy(self.placements),
            "selected_lamp_type": self.selected_lamp_type,
            "lamp_plan": deepcopy(self.lamp_plan),
            "tool_cache": self._copy_tool_cache(self.tool_cache),
        }
        self.snapshot_order.append(snapshot_id)
        return snapshot_id

    def restore(self, snapshot_id: str) -> bool:
        snapshot = self.snapshots.get(snapshot_id)
        if not isinstance(snapshot, dict):
            return False
        self.placements = deepcopy(snapshot.get("placements", {"lamps": [], "switches": []}))
        self.selected_lamp_type = snapshot.get("selected_lamp_type")
        self.lamp_plan = deepcopy(snapshot.get("lamp_plan"))
        self.tool_cache = deepcopy(snapshot.get("tool_cache", {}))
        self.record(
            "restore_snapshot",
            {"snapshot_id": snapshot_id},
            {"status": "ok", "label": snapshot.get("label", "")},
        )
        return True

    def list_snapshots(self) -> List[Dict[str, Any]]:
        return [
            {
                "snapshot_id": item,
                "label": self.snapshots[item].get("label", ""),
                "reason": self.snapshots[item].get("reason", ""),
                "created_at": self.snapshots[item].get("created_at", ""),
            }
            for item in self.snapshot_order
            if item in self.snapshots
        ]


class AgentStateManager:
    """
    状态管理器:
    - 存储房间基础属性和当前布局
    - 按需序列化给 LLM
    - 记录工具调用与思考历史
    """

    def __init__(self) -> None:
        self.rooms: Dict[str, RoomAgentState] = {}

    def add_room(self, room_id: str, state: RoomAgentState) -> None:
        self.rooms[room_id] = state

    def get_room(self, room_id: str) -> RoomAgentState:
        if room_id not in self.rooms:
            raise KeyError(f"room_id not found: {room_id}")
        return self.rooms[room_id]

    def to_llm_payload(self, room_id: str, max_rows: int = 64, max_cols: int = 64) -> Dict[str, Any]:
        state = self.get_room(room_id)
        return {
            "state_summary": state.summary(),
            "ascii_board": state.to_ascii_board(max_rows=max_rows, max_cols=max_cols, compress=True),
            "recent_tool_results": state.tool_result_history[-8:],
            "recent_thoughts": state.thought_history[-8:],
        }
