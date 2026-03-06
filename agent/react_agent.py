"""
ReAct 智能体（Reasoning + Acting）示例实现。

目标:
1) 使用工具链完成房间照明初步方案生成;
2) 通过状态管理器维护网格矩阵与元件位置;
3) 在 while 循环中让模型进行“读状态 -> 决策 -> 行动 -> 再读状态”。
"""

from __future__ import annotations

import json
import math
import os
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from preprocess.lighting_layout import (
    _select_lamp_cells_regular_grid,
    _select_lamp_cells_rule_based,
)
from preprocess.wiring_layout import (
    _build_edge_candidates,
    _build_mst,
    _grid_path_to_pixel_path,
    _merge_unique_step_segments,
    _orient_edges_from_switch,
    _pixel_path_to_cad_path,
)


class AgentRunLogger:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    GRAY = "\033[90m"

    def __init__(self, log_dir: Optional[str] = None) -> None:
        root_dir = Path(__file__).resolve().parents[1]
        final_log_dir = Path(log_dir) if log_dir else (root_dir / "logs")
        final_log_dir.mkdir(parents=True, exist_ok=True)
        self.max_line_len = max(120, int(os.getenv("CAD_AGENT_LOG_MAX_LEN", "320")))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = str(final_log_dir / f"react_agent_{ts}.log")
        self._emit("SESSION", f"log file: {self.log_path}", self.GRAY)

    @staticmethod
    def _safe_json(data: Any) -> str:
        try:
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            return str(data)

    def _one_line(self, text: str, max_len: Optional[int] = None) -> str:
        limit = self.max_line_len if max_len is None else max(80, int(max_len))
        compact = re.sub(r"\s+", " ", str(text)).strip()
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    def _one_line_json(self, data: Any, max_len: Optional[int] = None) -> str:
        return self._one_line(self._safe_json(data), max_len=max_len)

    def _emit(self, tag: str, message: str, color: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        plain = f"[{ts}] [{tag}] {message}"
        colored = f"{color}{plain}{self.RESET}"
        print(colored + "\n\n", end="")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(plain + "\n\n")

    def room_start(self, room_name: str, area_m2: float, shape: Tuple[int, int]) -> None:
        self._emit(
            "ROOM",
            f"start room='{room_name}' area_m2={float(area_m2):.3f} matrix_shape={list(shape)}",
            self.CYAN + self.BOLD,
        )

    def room_end(self, room_name: str, finish_reason: str, tool_calls: int, score: Optional[int]) -> None:
        self._emit(
            "ROOM",
            f"end room='{room_name}' finish='{finish_reason}' tool_calls={tool_calls} score={score}",
            self.CYAN + self.BOLD,
        )

    def llm_response(self, content: str) -> None:
        self._emit("LLM", f"raw_response={self._one_line(content)}", self.YELLOW)

    def action(self, action_obj: Dict[str, Any]) -> None:
        action = str(action_obj.get("action", ""))
        args = self._one_line_json(action_obj.get("args", {}), max_len=180)
        reason = self._one_line(str(action_obj.get("reason", "")), max_len=100)
        msg = f"action={action} args={args}"
        if reason:
            msg += f" reason='{reason}'"
        self._emit("ACTION", msg, self.MAGENTA)

    def thought(self, text: str) -> None:
        self._emit("THOUGHT", self._one_line(text, max_len=180), self.YELLOW + self.BOLD)

    def tool_io(self, tool_name: str, tool_input: Dict[str, Any], tool_output: Dict[str, Any]) -> None:
        in_line = self._one_line_json(tool_input, max_len=180)
        out_line = self._one_line_json(tool_output, max_len=220)
        self._emit("TOOL Calling", f"{tool_name} input={in_line} output={out_line}", self.BLUE)

    def error(self, message: str) -> None:
        self._emit("ERROR", message, self.RED)


@dataclass
class LampSpec:
    lamp_type: str
    model: str
    flux_lm: float
    power_w: Optional[float] = None
    vendor: str = ""
    url: str = ""


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
    logger: Optional[AgentRunLogger] = None

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, np.ndarray):
            self.matrix = np.array(self.matrix, dtype=np.int32)
        self.matrix = self.matrix.astype(np.int32)

    def record(self, tool_name: str, tool_input: Dict[str, Any], tool_output: Dict[str, Any]) -> None:
        record_item = {"tool": tool_name, "input": tool_input, "output": tool_output}
        self.tool_history.append(record_item)
        if self.logger:
            self.logger.tool_io(tool_name, tool_input, tool_output)

    def to_ascii_board(self, max_rows: int = 64, max_cols: int = 64, compress: bool = True) -> str:
        """
        使用 ASCII 棋盘映射法输出网格:
        - '#': 不可用区域(0)
        - '.': 可用区域(1)
        - 'D': 门区域(2)
        - 'L': 灯具
        - 'S': 开关
        """
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
                    v = int(sampled[rr, cc])
                    if v == 2:
                        ch = "D"
                    elif v == 1:
                        ch = "."
                    else:
                        ch = "#"
                chars.append(ch)
            row_str = "".join(chars)
            lines.append(_rle(row_str) if compress else row_str)
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
        }


def _is_point(item: Any) -> bool:
    return (
        isinstance(item, (list, tuple))
        and len(item) == 2
        and isinstance(item[0], (int, float, np.integer, np.floating))
        and isinstance(item[1], (int, float, np.integer, np.floating))
    )


class AgentStateManager:
    """
    状态管理器:
    - 存储房间基础属性和当前布局;
    - 按需序列化给 LLM;
    - 记录工具调用历史。
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
            "recent_tool_history": state.tool_history[-8:],
        }


class LightingTools:
    def __init__(
        self,
        lamp_catalog: Optional[List[Dict[str, Any]]] = None,
        room_lux_requirements: Optional[Dict[str, int]] = None,
    ) -> None:
        self.catalog = _build_lamp_specs(lamp_catalog or DEFAULT_LAMP_CATALOG)
        self.room_lux_map = dict(DEFAULT_ROOM_LUX)
        if room_lux_requirements:
            self.room_lux_map.update(room_lux_requirements)

    def infer_default_lamp_type(self, room_name: str) -> Dict[str, Any]:
        """
        确定性规则匹配（非工具）:
        根据房间名返回默认灯具类型与目标照度。
        """
        normalized_name = (room_name or "").strip()
        target_lux = _get_target_lux(normalized_name, self.room_lux_map)

        lamp_type = "筒灯"
        if any(k in normalized_name for k in ("配电", "除尘", "高温")):
            lamp_type = "防爆灯"
        elif any(k in normalized_name for k in ("楼梯", "卫生间", "盟洗")):
            lamp_type = "感应式吸顶灯"
        elif target_lux >= 500:
            lamp_type = "双管格栅灯"
        elif target_lux >= 300:
            lamp_type = "双管荧光灯"

        return {
            "room_name": normalized_name,
            "target_lux": int(target_lux),
            "lamp_type": lamp_type,
        }

    def tool_lookup_room_requirement(
        self,
        state: RoomAgentState,
        room_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        元件匹配工具。

        输入:
        - room_name: 可选，房间名称；为空时使用 state.room_name。

        输出:
        - room_name: 归一化后的房间名称
        - target_lux: 该房间目标照度(lx)
        - lamp_type: 推荐灯具类型
        - constraints: 布置约束
        """
        normalized_name = (room_name or state.room_name or "").strip()
        inferred = self.infer_default_lamp_type(normalized_name)
        preferred_lamp = str(inferred.get("lamp_type", "筒灯"))
        target_lux = int(inferred.get("target_lux", 300))
        result = {
            "room_name": normalized_name,
            "target_lux": target_lux,
            "preferred_lamp_types": [preferred_lamp],
            "lamp_type": preferred_lamp,
            "constraints": {
                "switch_near_door": True,
                "avoid_door_cells": True,
                "rectilinear_layout_preferred": True,
            },
        }
        state.selected_lamp_type = preferred_lamp
        state.tool_cache["room_requirement"] = result
        state.record("tool_lookup_room_requirement", {"room_name": normalized_name}, result)
        return result

    def tool_estimate_component_count(
        self,
        state: RoomAgentState,
        is_regular: Optional[bool] = None,
        min_spacing_m: float = 2.0,
        max_spacing_m: float = 3.0,
        max_lamps: int = 64,
        switch_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        元件数量估计工具。

        输入:
        - is_regular: 可选，是否规则房间；为空时自动判断。
        - min_spacing_m/max_spacing_m: 期望间距范围(米)。
        - max_lamps: 灯具数量上限。
        - switch_count: 开关数量（可选覆盖）。

        输出:
        - lamp_count/switch_count: 建议灯具与开关数量
        - grid_rows/grid_cols: 推荐灯具阵列
        - component_count: {"lamps":int,"switches":int}
        - is_regular/fill_ratio/aspect_ratio: 房间几何估计信息
        """
        valid = np.argwhere(state.matrix > 0)
        if len(valid) == 0:
            valid = np.argwhere(np.ones_like(state.matrix, dtype=np.int32) > 0)

        r_min, c_min = valid.min(axis=0)
        r_max, c_max = valid.max(axis=0)
        h_cells = int(r_max - r_min + 1)
        w_cells = int(c_max - c_min + 1)
        bbox_area_cells = max(1, h_cells * w_cells)
        fill_ratio = float(len(valid)) / float(bbox_area_cells)

        if is_regular is None:
            is_regular = bool(fill_ratio >= 0.90)

        aspect = float(w_cells) / max(1.0, float(h_cells))
        area_m2 = max(0.01, float(state.area_m2))
        room_w_m = max(0.1, math.sqrt(area_m2 * max(aspect, 1e-3)))
        room_h_m = max(0.1, area_m2 / room_w_m)

        base_spacing = (max(0.5, float(min_spacing_m)) + max(0.5, float(max_spacing_m))) / 2.0
        preferred_spacing = base_spacing if is_regular else max(0.8, base_spacing * 0.85)
        rough_count = int(math.ceil(area_m2 / max(0.25, preferred_spacing * preferred_spacing)))
        rough_count = max(1, min(int(max_lamps), rough_count))

        grid_rows, grid_cols = _search_grid_shape(
            target_count=rough_count,
            aspect=aspect,
            room_w_m=room_w_m,
            room_h_m=room_h_m,
            preferred_spacing_m=preferred_spacing,
        )
        lamp_count = int(grid_rows * grid_cols)

        if switch_count is None:
            switch_count_val = 1 if len(valid) > 0 else 0
        else:
            switch_count_val = max(0, int(switch_count))

        result = {
            "room_name": state.room_name,
            "is_regular": bool(is_regular),
            "fill_ratio": float(fill_ratio),
            "aspect_ratio": float(aspect),
            "grid_rows": int(grid_rows),
            "grid_cols": int(grid_cols),
            "lamp_count": int(lamp_count),
            "switch_count": int(switch_count_val),
            "component_count": {
                "lamps": int(lamp_count),
                "switches": int(switch_count_val),
            },
            "preferred_spacing_m": float(preferred_spacing),
            "room_w_m": float(room_w_m),
            "room_h_m": float(room_h_m),
        }
        state.tool_cache["component_count_plan"] = result
        state.record(
            "tool_estimate_component_count",
            {
                "is_regular": is_regular,
                "min_spacing_m": min_spacing_m,
                "max_spacing_m": max_spacing_m,
                "max_lamps": max_lamps,
                "switch_count": switch_count,
            },
            result,
        )
        return result

    def tool_calc_required_flux_per_lamp(
        self,
        state: RoomAgentState,
        target_lux: Optional[int] = None,
        lamp_count: Optional[int] = None,
        uf: float = 0.6,
        mf: float = 0.8,
    ) -> Dict[str, Any]:
        """
        元件光通量工具。

        输入:
        - target_lux: 目标照度(lx)；为空时从 tool_lookup_room_requirement 结果回退。
        - lamp_count: 灯具数量；为空时从 tool_estimate_component_count 结果回退。
        - uf/mf: 利用系数与维护系数。

        输出:
        - required_flux_per_lamp_lm: 单灯所需光通量(lm)
        - 并返回参与计算的参数。
        """
        cached_req = state.tool_cache.get("room_requirement", {}) or {}
        cached_count = state.tool_cache.get("component_count_plan", {}) or {}
        resolved_target_lux = int(
            target_lux
            if target_lux is not None
            else cached_req.get("target_lux", 300)
        )
        resolved_lamp_count = int(
            lamp_count
            if lamp_count is not None
            else cached_count.get("lamp_count", 1)
        )

        count = max(1, int(resolved_lamp_count))
        area_m2 = max(0.01, float(state.area_m2))
        util = max(1e-6, float(uf) * float(mf))
        required_flux = (float(resolved_target_lux) * area_m2) / max(1e-6, util * count)
        result = {
            "room_name": state.room_name,
            "target_lux": int(resolved_target_lux),
            "area_m2": float(area_m2),
            "lamp_count": int(count),
            "uf": float(uf),
            "mf": float(mf),
            "required_flux_per_lamp_lm": float(required_flux),
        }
        state.tool_cache["flux_plan"] = result
        state.record(
            "tool_calc_required_flux_per_lamp",
            {
                "target_lux": target_lux,
                "lamp_count": lamp_count,
                "resolved_target_lux": resolved_target_lux,
                "resolved_lamp_count": resolved_lamp_count,
                "uf": uf,
                "mf": mf,
            },
            result,
        )
        return result

    def tool_retrieve_lamp_model(
        self,
        state: RoomAgentState,
        lamp_type: Optional[str] = None,
        required_flux_lm: Optional[float] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        元件检索工具。

        输入:
        - lamp_type: 灯具类型；为空时回退到 state.selected_lamp_type 或默认类型。
        - required_flux_lm: 目标光通量；为空时从 tool_calc_required_flux_per_lamp 回退。
        - top_k: 返回候选数量。

        输出:
        - selected_lamp: 最匹配型号
        - candidates: 备选型号列表
        - match_scope: 是否按类型检索或全库回退
        """
        cached_flux = state.tool_cache.get("flux_plan", {}) or {}
        normalized_type = (lamp_type or state.selected_lamp_type or "筒灯").strip()
        resolved_flux = float(
            required_flux_lm
            if required_flux_lm is not None
            else cached_flux.get("required_flux_per_lamp_lm", 1000.0)
        )
        candidates = [x for x in self.catalog if x.lamp_type == normalized_type]
        match_scope = "by_type"
        if not candidates:
            candidates = list(self.catalog)
            match_scope = "fallback_all_types"
        if not candidates:
            raise ValueError("lamp catalog is empty")

        ranked = sorted(candidates, key=lambda x: abs(float(x.flux_lm) - float(resolved_flux)))
        selected = ranked[0]
        top = ranked[: max(1, int(top_k))]
        result = {
            "room_name": state.room_name,
            "requested_lamp_type": normalized_type,
            "required_flux_lm": float(resolved_flux),
            "selected_lamp": _lamp_to_dict(selected),
            "candidates": [_lamp_to_dict(x) for x in top],
            "match_scope": match_scope,
        }
        state.selected_lamp_type = str(result["selected_lamp"].get("lamp_type", normalized_type))
        state.tool_cache["lamp_model_plan"] = result
        state.record(
            "tool_retrieve_lamp_model",
            {
                "lamp_type": lamp_type,
                "required_flux_lm": required_flux_lm,
                "resolved_lamp_type": normalized_type,
                "resolved_required_flux_lm": resolved_flux,
                "top_k": top_k,
            },
            result,
        )
        return result

    def tool_place_components(
        self,
        state: RoomAgentState,
        lamp_count: Optional[int] = None,
        switch_count: Optional[int] = None,
        is_regular: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        元件布置工具（规则算法）。

        输入:
        - lamp_count: 可选，目标灯具数量；为空时按 state.lamp_plan/现有布置回退。
        - switch_count: 可选，目标开关数量；为空时按 tool_estimate_component_count 回退。
        - is_regular: 可选，是否规则房间；为空时根据填充率自动估计。

        输出:
        - lamp_positions/switch_positions: 灯具与开关网格坐标
        - lamp_count/switch_count: 实际放置数量
        - lamp_algorithm: 灯具算法标识(step7_regular_grid/step7_irregular_rule)
        """
        planned_count = int(
            lamp_count
            if lamp_count is not None
            else ((state.lamp_plan or {}).get("lamp_count", 0) or len(state.placements.get("lamps", [])) or 1)
        )
        planned_count = max(1, planned_count)
        cached_component_plan = state.tool_cache.get("component_count_plan", {}) or {}
        planned_switch_count = int(
            switch_count
            if switch_count is not None
            else (
                (state.lamp_plan or {}).get("switch_count", None)
                if (state.lamp_plan or {}).get("switch_count", None) is not None
                else cached_component_plan.get("switch_count", 1)
            )
        )
        planned_switch_count = max(0, planned_switch_count)

        if is_regular is None:
            valid = np.argwhere(state.matrix > 0)
            if len(valid) == 0:
                fill_ratio = 1.0
            else:
                r_min, c_min = valid.min(axis=0)
                r_max, c_max = valid.max(axis=0)
                bbox_area = max(1, int(r_max - r_min + 1) * int(c_max - c_min + 1))
                fill_ratio = float(len(valid)) / float(bbox_area)
            is_regular = bool(fill_ratio >= 0.90)

        if bool(is_regular):
            picked = _select_lamp_cells_regular_grid(state.matrix, planned_count)
            algo = "step7_regular_grid"
        else:
            picked = _select_lamp_cells_rule_based(state.matrix, planned_count)
            algo = "step7_irregular_rule"

        state.placements["lamps"] = [[int(r), int(c)] for r, c in picked]
        rows, cols = state.matrix.shape
        door_cells = [tuple(map(int, p)) for p in np.argwhere(state.matrix == 2).tolist()]
        valid_cells = [tuple(map(int, p)) for p in np.argwhere(state.matrix == 1).tolist()]
        switch_candidates: List[Tuple[int, int]] = []
        if door_cells:
            neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in neigh:
                for r, c in door_cells:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and int(state.matrix[rr, cc]) == 1:
                        switch_candidates.append((rr, cc))
        if not switch_candidates:
            switch_candidates = valid_cells

        # 去重并优先靠门（再靠房间中心）
        uniq_candidates: List[Tuple[int, int]] = []
        for rc in switch_candidates:
            if rc not in uniq_candidates:
                uniq_candidates.append(rc)
        if valid_cells:
            center_r = float(np.mean([p[0] for p in valid_cells]))
            center_c = float(np.mean([p[1] for p in valid_cells]))
            uniq_candidates = sorted(
                uniq_candidates,
                key=lambda rc: (
                    min((abs(rc[0] - d[0]) + abs(rc[1] - d[1])) for d in door_cells) if door_cells else 0,
                    abs(rc[0] - center_r) + abs(rc[1] - center_c),
                ),
            )
        switch_picked = uniq_candidates[:planned_switch_count]
        state.placements["switches"] = [[int(r), int(c)] for r, c in switch_picked]

        result = {
            "component_type": "components",
            "lamp_positions": state.placements["lamps"],
            "switch_positions": state.placements["switches"],
            "lamp_count": len(state.placements["lamps"]),
            "switch_count": len(state.placements["switches"]),
            "requested_lamp_count": int(planned_count),
            "requested_switch_count": int(planned_switch_count),
            "is_regular": bool(is_regular),
            "lamp_algorithm": algo,
        }
        state.record(
            "tool_place_components",
            {
                "lamp_count": lamp_count,
                "switch_count": switch_count,
                "is_regular": is_regular,
            },
            result,
        )
        return result

    def tool_validate_layout(
        self,
        state: RoomAgentState,
        min_lamp_dist_cells: int = 2,
        max_switch_to_door_dist_cells: int = 3,
    ) -> Dict[str, Any]:
        """
        元件布置规范性工具。

        输入:
        - min_lamp_dist_cells: 灯具最小建议间距（网格单位）。
        - max_switch_to_door_dist_cells: 开关到门的最大建议曼哈顿距离。

        输出:
        - score/is_valid: 评分与是否通过
        - violations: 违规列表
        - stats/suggestions: 统计与修复建议
        """
        rows, cols = state.matrix.shape
        lamps = [tuple(map(int, p)) for p in state.placements.get("lamps", []) if len(p) == 2]
        switches = [tuple(map(int, p)) for p in state.placements.get("switches", []) if len(p) == 2]
        door_cells = [tuple(map(int, p)) for p in np.argwhere(state.matrix == 2).tolist()]

        violations: List[Dict[str, Any]] = []

        def add_violation(code: str, severity: str, message: str) -> None:
            violations.append({"code": code, "severity": severity, "message": message})

        # 基础合法性检查
        if len(lamps) == 0:
            add_violation("NO_LAMP", "critical", "未放置任何灯具")
        if len(switches) == 0:
            add_violation("NO_SWITCH", "warning", "未放置开关")

        for rr, cc in lamps:
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                add_violation("LAMP_OUT_OF_RANGE", "critical", f"灯具坐标越界: [{rr},{cc}]")
            elif int(state.matrix[rr, cc]) != 1:
                add_violation("LAMP_ON_INVALID_CELL", "critical", f"灯具未落在可布置格: [{rr},{cc}]")

        for rr, cc in switches:
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                add_violation("SWITCH_OUT_OF_RANGE", "critical", f"开关坐标越界: [{rr},{cc}]")
            elif int(state.matrix[rr, cc]) != 1:
                add_violation("SWITCH_ON_INVALID_CELL", "critical", f"开关未落在可布置格: [{rr},{cc}]")

        if len(set(lamps)) != len(lamps):
            add_violation("LAMP_DUPLICATE_POSITION", "warning", "存在灯具坐标重叠")
        if len(set(switches)) != len(switches):
            add_violation("SWITCH_DUPLICATE_POSITION", "warning", "存在开关坐标重叠")

        # 灯具间距检查
        min_lamp_dist = None
        if len(lamps) >= 2:
            min_lamp_dist = 10**9
            for i in range(len(lamps)):
                for j in range(i + 1, len(lamps)):
                    d = math.sqrt((lamps[i][0] - lamps[j][0]) ** 2 + (lamps[i][1] - lamps[j][1]) ** 2)
                    min_lamp_dist = min(min_lamp_dist, d)
            if min_lamp_dist < float(min_lamp_dist_cells):
                add_violation("LAMPS_TOO_CLOSE", "warning", f"部分灯具间距过近: min={min_lamp_dist:.2f}格")

        # 均匀性(覆盖 proxy): 可布置格到最近灯具距离的变异系数
        coverage_cv = None
        valid_cells = np.argwhere(state.matrix == 1)
        if len(valid_cells) > 0 and len(lamps) > 0:
            dists: List[float] = []
            for rr, cc in valid_cells.tolist():
                best = min(math.sqrt((rr - lr) ** 2 + (cc - lc) ** 2) for lr, lc in lamps)
                dists.append(float(best))
            if dists:
                mean_d = float(np.mean(dists))
                std_d = float(np.std(dists))
                coverage_cv = std_d / max(1e-6, mean_d)
                if coverage_cv > 0.65:
                    add_violation("UNEVEN_COVERAGE", "warning", f"照明分布不均匀: cv={coverage_cv:.2f}")

        # 开关靠门检查
        switch_to_door = None
        if switches and door_cells:
            min_sd = min(abs(sr - dr) + abs(sc - dc) for sr, sc in switches for dr, dc in door_cells)
            switch_to_door = float(min_sd)
            if min_sd > int(max_switch_to_door_dist_cells):
                add_violation(
                    "SWITCH_FAR_FROM_DOOR",
                    "warning",
                    f"开关距门过远: min_manhattan={min_sd}",
                )

        score = 100
        for v in violations:
            score -= 30 if v["severity"] == "critical" else 10
        score = max(0, int(score))
        critical_count = sum(1 for v in violations if v["severity"] == "critical")
        is_valid = bool(critical_count == 0 and score >= 70)

        suggestions: List[str] = []
        codes = {v["code"] for v in violations}
        if "NO_SWITCH" in codes or "SWITCH_FAR_FROM_DOOR" in codes:
            suggestions.append("将开关移动到靠近门格(D)的一侧边缘可布置格")
        if "LAMPS_TOO_CLOSE" in codes or "UNEVEN_COVERAGE" in codes:
            suggestions.append("拉大灯具间距并向房间几何中心对称分布")
        if "LAMP_ON_INVALID_CELL" in codes or "SWITCH_ON_INVALID_CELL" in codes:
            suggestions.append("将元件移动到值为1的可布置网格")

        result = {
            "score": score,
            "is_valid": is_valid,
            "violations": violations,
            "stats": {
                "lamp_count": len(lamps),
                "switch_count": len(switches),
                "min_lamp_dist_cells": None if min_lamp_dist is None else float(min_lamp_dist),
                "coverage_cv": None if coverage_cv is None else float(coverage_cv),
                "switch_to_door_manhattan": switch_to_door,
            },
            "suggestions": suggestions,
        }
        state.record(
            "tool_validate_layout",
            {
                "min_lamp_dist_cells": min_lamp_dist_cells,
                "max_switch_to_door_dist_cells": max_switch_to_door_dist_cells,
            },
            result,
        )
        return result

    def tool_apply_layout_edit(
        self,
        state: RoomAgentState,
        edits: Any,
    ) -> Dict[str, Any]:
        """
        元件位置调整工具（批量编辑）。

        输入:
        - edits: 列表，每项格式
          {"component_type":"lamps|switches","source":[r,c]或null,"target":[r,c]}

        输出:
        - ok/applied/errors: 执行状态
        - placements: 更新后的所有元件坐标
        """
        if isinstance(edits, dict):
            edit_list = [edits]
        elif isinstance(edits, list):
            edit_list = edits
        else:
            edit_list = []

        applied = 0
        errors: List[str] = []
        for i, e in enumerate(edit_list):
            comp = str((e or {}).get("component_type", "lamps"))
            src = (e or {}).get("source")
            tgt = (e or {}).get("target")
            ok, err = self._apply_single_move(state=state, component_type=comp, source=src, target=tgt)
            if ok:
                applied += 1
            else:
                errors.append(f"edit[{i}]: {err}")

        result = {
            "ok": len(errors) == 0,
            "applied": int(applied),
            "errors": errors,
            "placements": state.placements,
        }
        state.record("tool_apply_layout_edit", {"edits": edit_list}, result)
        return result

    def _apply_single_move(
        self,
        state: RoomAgentState,
        component_type: str,
        source: Optional[List[int]],
        target: Any,
    ) -> Tuple[bool, Optional[str]]:
        rows, cols = state.matrix.shape
        if component_type not in ("lamps", "switches"):
            return False, f"invalid component_type: {component_type}"
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            return False, "invalid target: must be [row, col]"

        tr, tc = int(target[0]), int(target[1])
        if tr < 0 or tr >= rows or tc < 0 or tc >= cols:
            return False, "target out of range"
        if int(state.matrix[tr, tc]) != 1:
            return False, "target cell is not placeable (must be 1)"

        lst = state.placements.setdefault(component_type, [])
        src_pair = [int(source[0]), int(source[1])] if source and len(source) == 2 else None
        tgt_pair = [tr, tc]

        if src_pair and src_pair in lst:
            idx = lst.index(src_pair)
            lst[idx] = tgt_pair
        else:
            lst.append(tgt_pair)
        return True, None

    def tool_generate_wiring(
        self,
        state: RoomAgentState,
        turn_penalty: Optional[float] = None,
        bbox_pixel: Optional[List[int]] = None,
        cell_size_px: int = 40,
        cad_params: Optional[Dict[str, float]] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        布线工具（MST + A*）。

        输入:
        - state: 必含 matrix + lamps + switches
        - turn_penalty: 转弯惩罚
        - bbox_pixel/cell_size_px: 用于导出像素线路
        - cad_params/image_width/image_height: 用于导出CAD线路

        输出:
        - routes: 每条线路的 grid/pixel/cad 路径
        - merged_segments_*: 去重后的线段集合
        - status/unreachable_nodes: 连通状态与不可达节点
        """
        matrix = np.array(state.matrix, dtype=np.int32)
        lamp_points: List[Tuple[int, int]] = []
        switch_points: List[Tuple[int, int]] = []

        for p in state.placements.get("lamps", []):
            if _is_point(p):
                lamp_points.append((int(p[0]), int(p[1])))
        for p in state.placements.get("switches", []):
            if _is_point(p):
                switch_points.append((int(p[0]), int(p[1])))

        if not lamp_points or not switch_points:
            result = {
                "room_name": state.room_name,
                "status": "skipped",
                "reason": "missing_switch_or_lamps",
                "route_count": 0,
                "routes": [],
            }
            state.record(
                "tool_generate_wiring",
                {
                    "turn_penalty": turn_penalty,
                    "bbox_pixel": bbox_pixel,
                    "cell_size_px": cell_size_px,
                    "has_cad": bool(cad_params),
                    "image_width": image_width,
                    "image_height": image_height,
                },
                result,
            )
            return result

        penalty = float(turn_penalty) if turn_penalty is not None else float(os.getenv("CAD_WIRING_TURN_PENALTY", "0.8"))
        switch_point = switch_points[0]
        nodes: List[Tuple[int, int]] = [switch_point] + lamp_points
        node_labels = ["switch"] + [f"lamp_{i+1}" for i in range(len(lamp_points))]

        edge_candidates = _build_edge_candidates(matrix, nodes, turn_penalty=penalty)
        mst_edges = _build_mst(edge_candidates, len(nodes))
        directed = _orient_edges_from_switch(mst_edges, nodes, switch_idx=0)

        can_export_pixel = bool(bbox_pixel and len(bbox_pixel) == 4)
        can_export_cad = bool(can_export_pixel and cad_params and image_width and image_height)

        routes: List[Dict[str, Any]] = []
        route_paths_grid: List[List[Tuple[int, int]]] = []
        total_cost = 0.0
        for parent_idx, child_idx, path_grid, cost in directed:
            route_paths_grid.append(path_grid)
            total_cost += float(cost)

            route_obj: Dict[str, Any] = {
                "from_node": node_labels[parent_idx],
                "to_node": node_labels[child_idx],
                "from_grid": [int(nodes[parent_idx][0]), int(nodes[parent_idx][1])],
                "to_grid": [int(nodes[child_idx][0]), int(nodes[child_idx][1])],
                "path_grid": [[int(r), int(c)] for r, c in path_grid],
                "cost": float(cost),
            }
            if can_export_pixel:
                pixel_path = _grid_path_to_pixel_path(path_grid, bbox_pixel, int(cell_size_px))
                route_obj["path_pixel"] = pixel_path
                if can_export_cad:
                    cad_path = _pixel_path_to_cad_path(
                        pixel_path=pixel_path,
                        cad_params=cad_params or {},
                        image_w=int(image_width),
                        image_h=int(image_height),
                    )
                    route_obj["path_cad"] = cad_path
            routes.append(route_obj)

        unreachable_nodes: List[str] = []
        reachable = {0}
        for _, child_idx, _, _ in directed:
            reachable.add(child_idx)
        for idx in range(1, len(nodes)):
            if idx not in reachable:
                unreachable_nodes.append(node_labels[idx])

        merged_segments_grid = _merge_unique_step_segments(route_paths_grid)
        merged_segments_pixel: List[List[List[float]]] = []
        merged_segments_cad: List[List[List[float]]] = []
        if can_export_pixel:
            for seg in merged_segments_grid:
                p0, p1 = tuple(seg[0]), tuple(seg[1])
                pix = _grid_path_to_pixel_path([p0, p1], bbox_pixel, int(cell_size_px))
                merged_segments_pixel.append(pix)
                if can_export_cad:
                    cad = _pixel_path_to_cad_path(
                        pixel_path=pix,
                        cad_params=cad_params or {},
                        image_w=int(image_width),
                        image_h=int(image_height),
                    )
                    merged_segments_cad.append(cad)

        status = "ok"
        if len(directed) < max(0, len(nodes) - 1):
            status = "partial"

        result = {
            "room_name": state.room_name,
            "status": status,
            "turn_penalty": float(penalty),
            "node_count": len(nodes),
            "route_count": len(routes),
            "switch_grid_position": [int(switch_point[0]), int(switch_point[1])],
            "lamp_grid_positions": [[int(r), int(c)] for r, c in lamp_points],
            "total_route_cost": float(total_cost),
            "unreachable_nodes": unreachable_nodes,
            "routes": routes,
            "merged_segments_grid": merged_segments_grid,
            "merged_segments_pixel": merged_segments_pixel,
            "merged_segments_cad": merged_segments_cad,
        }
        state.record(
            "tool_generate_wiring",
            {
                "turn_penalty": penalty,
                "bbox_pixel": bbox_pixel,
                "cell_size_px": cell_size_px,
                "has_cad": bool(cad_params),
                "image_width": image_width,
                "image_height": image_height,
            },
            {
                "status": status,
                "route_count": len(routes),
                "unreachable_nodes": len(unreachable_nodes),
            },
        )
        return result

    def tool_read_matrix_state(
        self,
        state: RoomAgentState,
        max_rows: int = 64,
        max_cols: int = 64,
        compress: bool = True,
    ) -> Dict[str, Any]:
        """
        元件布置结果查看工具。

        输入:
        - max_rows/max_cols: ASCII棋盘采样尺寸
        - compress: 是否RLE压缩棋盘字符串

        输出:
        - summary: 当前房间状态摘要
        - ascii_board: 可供LLM阅读的棋盘文本
        """
        board = state.to_ascii_board(max_rows=max_rows, max_cols=max_cols, compress=compress)
        result = {
            "summary": state.summary(),
            "ascii_board": board,
        }
        state.record(
            "tool_read_matrix_state",
            {"max_rows": max_rows, "max_cols": max_cols, "compress": compress},
            {"summary": result["summary"]},
        )
        return result


class ReActLightingAgent:
    """
    简化 ReAct 智能体:
    - init_mode=rule 时先执行需求/数量/光通量/选型/布置形成初稿;
    - 在 while 循环中持续读取状态、校验并按需微调;
    - 最终调用终止动作 finish 并返回总结。
    """

    def __init__(
        self,
        tools: Optional[LightingTools] = None,
        provider: str = "qwen",
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        init_mode: str = "llm",
        log_dir: Optional[str] = None,
    ) -> None:
        self.tools = tools or LightingTools()
        self.provider = provider.strip().lower()
        self.temperature = float(temperature)
        self.init_mode = (init_mode or "rule").strip().lower()
        if self.init_mode not in ("rule", "llm"):
            self.init_mode = "rule"
        self.api_key, self.base_url, self.model = _resolve_provider(self.provider, model_name)
        self.run_logger = AgentRunLogger(log_dir=log_dir)
        self.client: Optional[OpenAI] = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.run_logger._emit(
            "MODEL",
            f"provider={self.provider} model={self.model} base_url={self.base_url} init_mode={self.init_mode}",
            AgentRunLogger.GRAY,
        )

    @staticmethod
    def list_tools() -> List[Dict[str, Any]]:
        """
        返回可用原子工具清单（工具名、输入、输出）。
        """
        return deepcopy(REACT_TOOL_SPECS)

    def run_for_room(
        self,
        state: RoomAgentState,
        max_steps: int = 6,
    ) -> Dict[str, Any]:
        state.logger = self.run_logger
        self.run_logger.room_start(state.room_name, state.area_m2, state.matrix.shape)
        # 初始阶段:
        # - rule: 先生成规则初稿，再进入 ReAct 迭代
        # - llm : 从空布局开始，由模型自主调用工具完成布局
        if self.init_mode == "rule":
            init_stage = self._run_replan_design(state, args={})
            first_validation = init_stage["validation"]
        else:
            state.placements["lamps"] = []
            state.placements["switches"] = []
            state.lamp_plan = None
            first_validation = self.tools.tool_validate_layout(state)
            init_stage = {
                "req": None,
                "count_plan": None,
                "flux_plan": None,
                "model_plan": None,
                "coords_plan": None,
                "validation": first_validation,
            }
            state.record(
                "init_mode",
                {"mode": "llm"},
                {"status": "start_from_empty"},
            )
        latest_wiring: Optional[Dict[str, Any]] = None

        final_reason = "max_steps_reached"
        for _ in range(max(1, int(max_steps))):
            view = self.tools.tool_read_matrix_state(state)
            validation = self.tools.tool_validate_layout(state)
            action = self._decide_action(state, view, validation)
            self.run_logger.action(action if isinstance(action, dict) else {"action": "invalid", "raw": str(action)})
            if isinstance(action, dict) and action.get("thought"):
                self.run_logger.thought(str(action.get("thought")))
            if action.get("action") == "finish":
                final_reason = str(action.get("reason", "done"))
                state.record("finish", {}, {"reason": final_reason, "strategy": action.get("strategy", "")})
                break
            if action.get("action") == "tool_validate_layout":
                # 显式校验动作，无状态变更
                continue
            if action.get("action") == "tool_lookup_room_requirement":
                args = action.get("args", {}) or {}
                self.tools.tool_lookup_room_requirement(
                    state=state,
                    room_name=args.get("room_name"),
                )
                continue
            if action.get("action") == "tool_estimate_component_count":
                args = action.get("args", {}) or {}
                self.tools.tool_estimate_component_count(
                    state=state,
                    is_regular=args.get("is_regular"),
                    min_spacing_m=float(args.get("min_spacing_m", 2.0)),
                    max_spacing_m=float(args.get("max_spacing_m", 3.0)),
                    max_lamps=int(args.get("max_lamps", 64)),
                    switch_count=args.get("switch_count"),
                )
                continue
            if action.get("action") == "tool_calc_required_flux_per_lamp":
                args = action.get("args", {}) or {}
                self.tools.tool_calc_required_flux_per_lamp(
                    state=state,
                    target_lux=args.get("target_lux"),
                    lamp_count=args.get("lamp_count"),
                    uf=float(args.get("uf", 0.6)),
                    mf=float(args.get("mf", 0.8)),
                )
                continue
            if action.get("action") == "tool_retrieve_lamp_model":
                args = action.get("args", {}) or {}
                self.tools.tool_retrieve_lamp_model(
                    state=state,
                    lamp_type=args.get("lamp_type"),
                    required_flux_lm=args.get("required_flux_lm"),
                    top_k=int(args.get("top_k", 3)),
                )
                continue
            if action.get("action") == "tool_place_components":
                args = action.get("args", {}) or {}
                self.tools.tool_place_components(
                    state=state,
                    lamp_count=args.get("lamp_count"),
                    switch_count=args.get("switch_count"),
                    is_regular=args.get("is_regular"),
                )
                continue
            if action.get("action") == "tool_read_matrix_state":
                args = action.get("args", {}) or {}
                self.tools.tool_read_matrix_state(
                    state=state,
                    max_rows=int(args.get("max_rows", 64)),
                    max_cols=int(args.get("max_cols", 64)),
                    compress=bool(args.get("compress", True)),
                )
                continue
            if action.get("action") == "tool_generate_wiring":
                args = action.get("args", {}) or {}
                latest_wiring = self.tools.tool_generate_wiring(
                    state=state,
                    turn_penalty=args.get("turn_penalty"),
                    bbox_pixel=args.get("bbox_pixel"),
                    cell_size_px=int(args.get("cell_size_px", 40)),
                    cad_params=args.get("cad_params"),
                    image_width=args.get("image_width"),
                    image_height=args.get("image_height"),
                )
                continue
            if action.get("action") == "tool_apply_layout_edit":
                args = action.get("args", {}) or {}
                self.tools.tool_apply_layout_edit(
                    state=state,
                    edits=args.get("edits", []),
                )
            else:
                final_reason = f"unknown_action:{action.get('action')}"
                break

        final_validation = self.tools.tool_validate_layout(state)
        if latest_wiring is None:
            latest_wiring = self.tools.tool_generate_wiring(state=state)
        result = {
            "room_name": state.room_name,
            "selected_lamp_type": state.selected_lamp_type,
            "lamp_plan": state.lamp_plan,
            "placements": state.placements,
            "wiring_plan": latest_wiring,
            "log_file": self.run_logger.log_path,
            "tool_calls": len(state.tool_history),
            "finish_reason": final_reason,
            "strategy_summary": self._build_strategy_summary(state),
            "validation": final_validation,
            "stage_outputs": {
                "tool_lookup_room_requirement": init_stage["req"],
                "tool_estimate_component_count": init_stage["count_plan"],
                "tool_calc_required_flux_per_lamp": init_stage["flux_plan"],
                "tool_retrieve_lamp_model": init_stage["model_plan"],
                "tool_place_components": init_stage["coords_plan"],
                "initial_validation": first_validation,
            },
        }
        self.run_logger.room_end(
            room_name=state.room_name,
            finish_reason=final_reason,
            tool_calls=len(state.tool_history),
            score=(final_validation or {}).get("score"),
        )
        return result

    def _run_replan_design(
        self,
        state: RoomAgentState,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        重新进行完整方案设计。允许覆盖灯具类型、灯具/开关数量、规则性和照度参数。
        """
        args = args or {}
        req = self.tools.tool_lookup_room_requirement(state)

        count_plan = self.tools.tool_estimate_component_count(
            state=state,
            is_regular=args.get("is_regular"),
            min_spacing_m=float(args.get("min_spacing_m", 2.0)),
            max_spacing_m=float(args.get("max_spacing_m", 3.0)),
            max_lamps=int(args.get("max_lamps", 64)),
            switch_count=args.get("switch_count"),
        )
        if args.get("lamp_count") is not None:
            lamp_count = max(1, int(args["lamp_count"]))
            count_plan["lamp_count"] = lamp_count
        else:
            lamp_count = int(count_plan.get("lamp_count", 1))
        if args.get("switch_count") is not None:
            component_switch_count = max(0, int(args["switch_count"]))
            count_plan["switch_count"] = component_switch_count
        else:
            component_switch_count = int(count_plan.get("switch_count", 1))

        target_lux = int(args.get("target_lux", req.get("target_lux", 300)))
        uf = float(args.get("uf", 0.6))
        mf = float(args.get("mf", 0.8))
        flux_plan = self.tools.tool_calc_required_flux_per_lamp(
            state=state,
            target_lux=target_lux,
            lamp_count=lamp_count,
            uf=uf,
            mf=mf,
        )

        lamp_type = str(args.get("lamp_type", req.get("lamp_type", "筒灯")))
        model_plan = self.tools.tool_retrieve_lamp_model(
            state=state,
            lamp_type=lamp_type,
            required_flux_lm=float(flux_plan.get("required_flux_per_lamp_lm", 1000.0)),
            top_k=int(args.get("top_k", 3)),
        )

        # 如显式指定阵列，覆盖自动估计
        grid_rows = int(args.get("grid_rows", count_plan.get("grid_rows", 1)))
        grid_cols = int(args.get("grid_cols", count_plan.get("grid_cols", 1)))
        if args.get("grid_rows") is not None and args.get("grid_cols") is not None:
            count_plan["grid_rows"] = max(1, grid_rows)
            count_plan["grid_cols"] = max(1, grid_cols)
            count_plan["lamp_count"] = max(1, grid_rows * grid_cols)
            lamp_count = int(count_plan["lamp_count"])

        state.lamp_plan = {
            "room_name": state.room_name,
            "target_lux": int(target_lux),
            "grid_rows": int(count_plan.get("grid_rows", 1)),
            "grid_cols": int(count_plan.get("grid_cols", 1)),
            "lamp_count": int(lamp_count),
            "switch_count": int(component_switch_count),
            "required_flux_per_lamp_lm": float(flux_plan.get("required_flux_per_lamp_lm", 1000.0)),
            "selected_lamp": model_plan.get("selected_lamp", {}),
            "backup_options": model_plan.get("candidates", []),
            "spacing_m": float(count_plan.get("preferred_spacing_m", 2.4)),
            "uf": float(uf),
            "mf": float(mf),
            "is_regular": bool(count_plan.get("is_regular", True)),
        }
        state.tool_cache["lamp_plan"] = state.lamp_plan

        coords_plan = self.tools.tool_place_components(
            state=state,
            lamp_count=int(state.lamp_plan.get("lamp_count", 1)),
            switch_count=int(state.lamp_plan.get("switch_count", 1)),
            is_regular=bool(state.lamp_plan.get("is_regular", True)),
        )
        validation = self.tools.tool_validate_layout(state)
        state.record("internal_replan_design", {"args": args}, {"validation_score": validation.get("score")})
        return {
            "req": req,
            "count_plan": count_plan,
            "flux_plan": flux_plan,
            "model_plan": model_plan,
            "coords_plan": coords_plan,
            "validation": validation,
        }

    def _decide_action(
        self,
        state: RoomAgentState,
        view: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.client is None:
            return {"action": "finish", "reason": "no_llm_key", "strategy": "deterministic tools only"}

        system_prompt = REACT_SYSTEM_PROMPT
        user_prompt = json.dumps(
            {
                "room_state": view["summary"],
                "ascii_board": view["ascii_board"],
                "validation": validation,
                "recent_history": state.tool_history[-8:],
            },
            ensure_ascii=False,
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
            content = ((resp.choices or [{}])[0].message.content or "").strip()
            if state.logger:
                state.logger.llm_response(content)
            obj = _extract_json(content)
            if isinstance(obj, dict) and obj.get("action"):
                return obj
        except Exception as e:
            if state.logger:
                state.logger.error(f"llm request failed in _decide_action: {e}")
        return {"action": "finish", "reason": "llm_parse_failed", "strategy": "fallback finish"}

    @staticmethod
    def _build_strategy_summary(state: RoomAgentState) -> str:
        lamp_type = state.selected_lamp_type or "未知灯具"
        lamp_n = len(state.placements.get("lamps", []))
        switch_n = len(state.placements.get("switches", []))
        return f"房间[{state.room_name}] 采用[{lamp_type}]，灯具{lamp_n}个，开关{switch_n}个。"


def _search_grid_shape(
    target_count: int,
    aspect: float,
    room_w_m: float,
    room_h_m: float,
    preferred_spacing_m: float,
) -> Tuple[int, int]:
    best = (1, max(1, target_count))
    best_score = 1e18
    for rows in range(1, min(16, target_count) + 1):
        cols = int(math.ceil(target_count / rows))
        spacing_x = room_w_m / max(1, cols + 1)
        spacing_y = room_h_m / max(1, rows + 1)
        score = 0.0
        score += 1.0 * abs((cols / max(1, rows)) - max(1e-6, aspect))
        score += 0.6 * abs(spacing_x - preferred_spacing_m)
        score += 0.6 * abs(spacing_y - preferred_spacing_m)
        score += 0.2 * (rows * cols - target_count)
        if score < best_score:
            best_score = score
            best = (rows, cols)
    return best


def _rle(s: str) -> str:
    if not s:
        return s
    out: List[str] = []
    cnt = 1
    for i in range(1, len(s) + 1):
        if i < len(s) and s[i] == s[i - 1]:
            cnt += 1
        else:
            ch = s[i - 1]
            out.append(f"{ch}{cnt}" if cnt > 1 else ch)
            cnt = 1
    return "".join(out)


def _parse_flux_lm(value: Any, default: float = 1000.0) -> float:
    text = str(value or "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*lm", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    return float(default)


def _parse_power_w(value: Any) -> Optional[float]:
    text = str(value or "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*W", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    return None


def _build_lamp_specs(raw_catalog: List[Dict[str, Any]]) -> List[LampSpec]:
    specs: List[LampSpec] = []
    for item in raw_catalog:
        lamp_type = str(item.get("灯具类型", "")).strip()
        if not lamp_type:
            continue
        specs.append(
            LampSpec(
                lamp_type=lamp_type,
                model=str(item.get("型号", "")).strip(),
                flux_lm=_parse_flux_lm(item.get("光通量", "")),
                power_w=_parse_power_w(item.get("功率", "")),
                vendor=str(item.get("厂家", "")).strip(),
                url=str(item.get("购买链接", "")).strip(),
            )
        )
    return specs


def _lamp_to_dict(spec: LampSpec) -> Dict[str, Any]:
    return {
        "lamp_type": spec.lamp_type,
        "model": spec.model,
        "flux_lm": spec.flux_lm,
        "power_w": spec.power_w,
        "vendor": spec.vendor,
        "url": spec.url,
    }


def _get_target_lux(room_name: str, lux_map: Dict[str, int]) -> int:
    name = (room_name or "").strip()
    if name in lux_map:
        return int(lux_map[name])
    for k, v in lux_map.items():
        if k and k in name:
            return int(v)
    return 300


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    fence = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _resolve_provider(provider: str, model_name: Optional[str]) -> Tuple[str, str, str]:
    provider = (provider or "qwen").strip().lower()
    model_name = (model_name or "").strip()
    if provider in ("qwen", "dashscope"):
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = model_name or os.getenv("CAD_AGENT_QWEN_MODEL", "qwen-plus").strip()
        return api_key, base_url, model
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip() or "https://api.deepseek.com/v1"
        model = model_name or os.getenv("CAD_AGENT_DEEPSEEK_MODEL", "deepseek-chat").strip()
        return api_key, base_url, model
    raise ValueError(f"unsupported provider: {provider}. Use 'qwen' or 'deepseek'.")


REACT_TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "tool_lookup_room_requirement",
        "input": {
            "room_name": "str | optional",
        },
        "output": {
            "room_name": "str",
            "target_lux": "int",
            "lamp_type": "str",
            "constraints": "dict",
        },
    },
    {
        "name": "tool_estimate_component_count",
        "input": {
            "is_regular": "bool | optional",
            "min_spacing_m": "float | optional",
            "max_spacing_m": "float | optional",
            "max_lamps": "int | optional",
            "switch_count": "int | optional",
        },
        "output": {
            "lamp_count": "int",
            "switch_count": "int",
            "grid_rows": "int",
            "grid_cols": "int",
            "is_regular": "bool",
            "fill_ratio": "float",
            "component_count": "{lamps:int,switches:int}",
        },
    },
    {
        "name": "tool_calc_required_flux_per_lamp",
        "input": {
            "target_lux": "int | optional",
            "lamp_count": "int | optional",
            "uf": "float | optional",
            "mf": "float | optional",
        },
        "output": {
            "required_flux_per_lamp_lm": "float",
            "target_lux": "int",
            "lamp_count": "int",
        },
    },
    {
        "name": "tool_retrieve_lamp_model",
        "input": {
            "lamp_type": "str | optional",
            "required_flux_lm": "float | optional",
            "top_k": "int | optional",
        },
        "output": {
            "selected_lamp": "dict",
            "candidates": "list[dict]",
            "match_scope": "str",
        },
    },
    {
        "name": "tool_place_components",
        "input": {
            "lamp_count": "int | optional",
            "switch_count": "int | optional",
            "is_regular": "bool | optional",
        },
        "output": {
            "lamp_positions": "list[[row, col]]",
            "switch_positions": "list[[row, col]]",
            "lamp_count": "int",
            "switch_count": "int",
            "lamp_algorithm": "str",
        },
    },
    {
        "name": "tool_validate_layout",
        "input": {
            "min_lamp_dist_cells": "int | optional",
            "max_switch_to_door_dist_cells": "int | optional",
        },
        "output": {
            "score": "int",
            "is_valid": "bool",
            "violations": "list[dict]",
            "suggestions": "list[str]",
        },
    },
    {
        "name": "tool_generate_wiring",
        "input": {
            "turn_penalty": "float | optional",
            "bbox_pixel": "[min_x,min_y,max_x,max_y] | optional",
            "cell_size_px": "int | optional",
            "cad_params": "dict | optional",
            "image_width": "int | optional",
            "image_height": "int | optional",
        },
        "output": {
            "status": "str",
            "routes": "list[dict]",
            "merged_segments_grid": "list[[[r0,c0],[r1,c1]]]",
        },
    },
    {
        "name": "tool_apply_layout_edit",
        "input": {
            "edits": "list[{component_type:'lamps|switches',source:[r,c]|null,target:[r,c]}]",
        },
        "output": {
            "ok": "bool",
            "applied": "int",
            "errors": "list[str]",
            "placements": "dict",
        },
    },
    {
        "name": "tool_read_matrix_state",
        "input": {
            "max_rows": "int | optional",
            "max_cols": "int | optional",
            "compress": "bool | optional",
        },
        "output": {
            "summary": "dict",
            "ascii_board": "str",
        },
    },
]


REACT_SYSTEM_PROMPT = """
你是建筑照明自动化智能体。目标是在离散网格上完成:
需求匹配 -> 数量估计 -> 光通量计算 -> 型号检索 -> 布点 -> 校验修正 -> 布线。

你只能通过工具行动。不要伪造工具结果。

网格语义:
- matrix中: 0=障碍, 1=可布置, 2=门位
- ascii_board中: #=障碍, .=可布置, D=门, L=灯, S=开关

你每轮必须仅输出一个JSON对象，禁止输出Markdown、解释性文字或多段内容。
固定格式:
{
  "thought": "一句简短决策理由",
  "action": "tool_lookup_room_requirement | tool_estimate_component_count | tool_calc_required_flux_per_lamp | tool_retrieve_lamp_model | tool_place_components | tool_validate_layout | tool_generate_wiring | tool_apply_layout_edit | tool_read_matrix_state | finish",
  "args": {},
  "reason": "仅在action=finish时必填",
  "strategy": "仅在action=finish时必填"
}

工具参数约定:
1) tool_lookup_room_requirement
   args: {"room_name":"可选"}
2) tool_estimate_component_count
   args: {"is_regular":可选,"min_spacing_m":可选,"max_spacing_m":可选,"max_lamps":可选,"switch_count":可选}
3) tool_calc_required_flux_per_lamp
   args: {"target_lux":可选,"lamp_count":可选,"uf":可选,"mf":可选}
4) tool_retrieve_lamp_model
   args: {"lamp_type":"可选","required_flux_lm":可选,"top_k":可选}
5) tool_place_components
   args: {"lamp_count":可选,"switch_count":可选,"is_regular":可选}
6) tool_validate_layout
   args: {"min_lamp_dist_cells":可选,"max_switch_to_door_dist_cells":可选}
7) tool_generate_wiring
   args: {"turn_penalty":可选,"bbox_pixel":[min_x,min_y,max_x,max_y]可选,"cell_size_px":可选,"cad_params":可选,"image_width":可选,"image_height":可选}
8) tool_apply_layout_edit
   args: {"edits":[{"component_type":"lamps|switches","source":[r,c]或null,"target":[r,c]}]}
9) tool_read_matrix_state
   args: {"max_rows":可选,"max_cols":可选,"compress":可选}
10) finish
   args: {}

执行策略:
- 从空布局开始时，优先顺序:
  tool_lookup_room_requirement -> tool_estimate_component_count -> tool_calc_required_flux_per_lamp -> tool_retrieve_lamp_model -> tool_place_components -> tool_validate_layout。
- 若校验存在critical或明显warning，先tool_apply_layout_edit再tool_validate_layout。
- 布线前确保至少有1个开关和1个灯具。
- 布局可用后调用tool_generate_wiring。
- 仅当布局和布线都完成时调用finish。
- 每轮只调用一个动作，不可在同一轮做多动作。

硬约束:
- 任何target必须落在值为1的网格。
- 不能将元件放在0或2上。
- 优先横平竖直、均匀分布，避免灯具聚集。
"""


DEFAULT_LAMP_CATALOG: List[Dict[str, Any]] = [
    {
        "灯具类型": "感应式吸顶灯",
        "型号": "LPXDD 002",
        "光通量": "1000lm",
        "功率": "15W",
        "厂家": "Alibaba 供应商",
        "购买链接": "https://www.alibaba.com/product-detail/Modern-Intelligent-LED-Induction-Ceiling-Light_1601488477089.html",
    },
    {
        "灯具类型": "防爆灯",
        "型号": "BC9102S-L30",
        "光通量": "4200lm",
        "功率": "30W",
        "厂家": "通明电器 TORMIN",
        "购买链接": "https://i-item.jd.com/100021096200.html",
    },
    {
        "灯具类型": "双管格栅灯",
        "型号": "ML-XTD014E",
        "光通量": "3500lm",
        "功率": "36W",
        "厂家": "Moonlight",
        "购买链接": "https://www.alibaba.com/product-detail/Industrial-Grille-Light-36W-4FT-T8_1600967544017.html",
    },
    {
        "灯具类型": "双管荧光灯",
        "型号": "BAY51-S28XJWF1",
        "光通量": "5600lm",
        "功率": "56W",
        "厂家": "合隆 Helon",
        "购买链接": "https://test-www.mymro.cn:443/u-8W2652.html",
    },
    {
        "灯具类型": "筒灯",
        "型号": "tp2351q",
        "光通量": "540lm",
        "功率": "6W",
        "厂家": "tp",
        "购买链接": "https://www.alibaba.com/product-detail/Modern-Aluminum-Recessed-Downlight-Led-Spotlight_1601702044947.html",
    },
]


DEFAULT_ROOM_LUX: Dict[str, int] = {
    "办公室": 300,
    "办公室1": 300,
    "楼梯间": 100,
    "楼梯间1": 100,
    "配电室": 200,
    "煤样存放室": 100,
    "元素分析室": 500,
    "高温室": 300,
    "热量室准备间": 300,
    "工业分析室": 500,
    "天平室": 500,
    "接样室": 300,
    "男卫生间": 100,
    "备用间": 100,
    "盟洗室": 100,
    "存样室": 100,
    "女卫生间": 100,
    "除尘室": 300,
}
