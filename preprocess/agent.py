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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI


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
    tool_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, np.ndarray):
            self.matrix = np.array(self.matrix, dtype=np.int32)
        self.matrix = self.matrix.astype(np.int32)

    def record(self, tool_name: str, tool_input: Dict[str, Any], tool_output: Dict[str, Any]) -> None:
        self.tool_history.append(
            {
                "tool": tool_name,
                "input": tool_input,
                "output": tool_output,
            }
        )

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
            "placements": self.placements,
            "tool_calls": len(self.tool_history),
        }


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

    def tool1_match_lamp_type(self, state: RoomAgentState) -> Dict[str, Any]:
        room_name = (state.room_name or "").strip()
        target_lux = _get_target_lux(room_name, self.room_lux_map)

        lamp_type = "筒灯"
        if any(k in room_name for k in ("配电", "除尘", "高温")):
            lamp_type = "防爆灯"
        elif any(k in room_name for k in ("楼梯", "卫生间", "盟洗")):
            lamp_type = "感应式吸顶灯"
        elif target_lux >= 500:
            lamp_type = "双管格栅灯"
        elif target_lux >= 300:
            lamp_type = "双管荧光灯"

        state.selected_lamp_type = lamp_type
        result = {"room_name": room_name, "target_lux": target_lux, "lamp_type": lamp_type}
        state.record("tool1_match_lamp_type", {"room_name": room_name}, result)
        return result

    def tool2_select_spec_and_count(
        self,
        state: RoomAgentState,
        spacing_m: float = 2.4,
        uf: float = 0.6,
        mf: float = 0.8,
        max_lamps: int = 64,
    ) -> Dict[str, Any]:
        """
        先估算阵列(rows, cols)，再结合照度公式选择灯具规格并给出数量。
        """
        room_name = (state.room_name or "").strip()
        area_m2 = max(0.01, float(state.area_m2))
        target_lux = _get_target_lux(room_name, self.room_lux_map)

        if not state.selected_lamp_type:
            self.tool1_match_lamp_type(state)
        selected_type = state.selected_lamp_type or "筒灯"

        h_cells, w_cells = state.matrix.shape
        valid = np.argwhere(state.matrix > 0)
        if len(valid) > 0:
            r_min, c_min = valid.min(axis=0)
            r_max, c_max = valid.max(axis=0)
            h_cells = int(r_max - r_min + 1)
            w_cells = int(c_max - c_min + 1)
        aspect = float(w_cells) / max(1.0, float(h_cells))
        room_w_m = max(0.1, math.sqrt(area_m2 * max(aspect, 1e-3)))
        room_h_m = max(0.1, area_m2 / room_w_m)

        seed_rows = max(1, int(round(room_h_m / max(0.5, spacing_m))))
        seed_cols = max(1, int(round(room_w_m / max(0.5, spacing_m))))
        n_seed = seed_rows * seed_cols

        type_specs = [s for s in self.catalog if s.lamp_type == selected_type]
        if not type_specs:
            type_specs = self.catalog[:]
        if not type_specs:
            raise ValueError("lamp catalog is empty")

        rough_flux = float(np.mean([s.flux_lm for s in type_specs]))
        n_lux = int(math.ceil((target_lux * area_m2) / max(1e-6, uf * mf * rough_flux)))
        n_target = max(1, min(max_lamps, max(n_seed, n_lux)))

        rows, cols = _search_grid_shape(
            target_count=n_target,
            aspect=aspect,
            room_w_m=room_w_m,
            room_h_m=room_h_m,
            preferred_spacing_m=spacing_m,
        )
        n_final = int(rows * cols)
        required_flux_per_lamp = (target_lux * area_m2) / max(1e-6, uf * mf * n_final)

        primary = min(type_specs, key=lambda s: abs(s.flux_lm - required_flux_per_lamp))
        backups = sorted(
            self.catalog,
            key=lambda s: abs(s.flux_lm - required_flux_per_lamp),
        )[:3]

        plan = {
            "room_name": room_name,
            "target_lux": int(target_lux),
            "grid_rows": int(rows),
            "grid_cols": int(cols),
            "lamp_count": int(n_final),
            "required_flux_per_lamp_lm": float(required_flux_per_lamp),
            "selected_lamp": _lamp_to_dict(primary),
            "backup_options": [_lamp_to_dict(x) for x in backups],
            "spacing_m": float(spacing_m),
            "uf": float(uf),
            "mf": float(mf),
        }
        state.lamp_plan = plan
        state.record(
            "tool2_select_spec_and_count",
            {"spacing_m": spacing_m, "uf": uf, "mf": mf, "max_lamps": max_lamps},
            plan,
        )
        return plan

    def tool3_generate_uniform_coords(
        self,
        state: RoomAgentState,
        component_type: str = "lamps",
    ) -> Dict[str, Any]:
        """
        根据 tool2 阵列结果，在多边形离散矩阵(值为1)内生成尽可能均匀的绝对网格坐标。
        """
        if not state.lamp_plan:
            self.tool2_select_spec_and_count(state)
        rows = int(state.lamp_plan["grid_rows"])
        cols = int(state.lamp_plan["grid_cols"])

        valid = np.argwhere(state.matrix == 1)
        if len(valid) == 0:
            result = {"component_type": component_type, "positions": []}
            state.record("tool3_generate_uniform_coords", {"component_type": component_type}, result)
            return result

        r_min, c_min = valid.min(axis=0)
        r_max, c_max = valid.max(axis=0)
        target_points: List[Tuple[float, float]] = []
        for r in range(rows):
            rr = float(r_min) + (r + 0.5) * (float(r_max - r_min + 1) / rows)
            for c in range(cols):
                cc = float(c_min) + (c + 0.5) * (float(c_max - c_min + 1) / cols)
                target_points.append((rr, cc))

        free = {(int(p[0]), int(p[1])) for p in valid.tolist()}
        placed: List[Tuple[int, int]] = []
        for tp in target_points:
            pick = _nearest_free_cell(tp, free)
            if pick is not None:
                placed.append(pick)
                free.discard(pick)

        if len(placed) < rows * cols and free:
            remaining = list(free)
            while len(placed) < rows * cols and remaining:
                if not placed:
                    pick = remaining.pop(0)
                else:
                    idx = int(
                        np.argmax(
                            [
                                min((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 for y in placed)
                                for x in remaining
                            ]
                        )
                    )
                    pick = remaining.pop(idx)
                placed.append((int(pick[0]), int(pick[1])))

        placed = placed[: rows * cols]
        state.placements[component_type] = [[int(r), int(c)] for r, c in placed]
        result = {
            "component_type": component_type,
            "positions": state.placements[component_type],
            "count": len(state.placements[component_type]),
        }
        state.record("tool3_generate_uniform_coords", {"component_type": component_type}, result)
        return result

    def tool4_move_component(
        self,
        state: RoomAgentState,
        component_type: str,
        source: Optional[List[int]],
        target: List[int],
    ) -> Dict[str, Any]:
        rows, cols = state.matrix.shape
        if component_type not in ("lamps", "switches"):
            result = {"ok": False, "error": f"invalid component_type: {component_type}"}
            state.record("tool4_move_component", {"component_type": component_type, "source": source, "target": target}, result)
            return result

        tr, tc = int(target[0]), int(target[1])
        if tr < 0 or tr >= rows or tc < 0 or tc >= cols:
            result = {"ok": False, "error": "target out of range"}
            state.record("tool4_move_component", {"component_type": component_type, "source": source, "target": target}, result)
            return result
        if int(state.matrix[tr, tc]) != 1:
            result = {"ok": False, "error": "target cell is not placeable (must be 1)"}
            state.record("tool4_move_component", {"component_type": component_type, "source": source, "target": target}, result)
            return result

        lst = state.placements.setdefault(component_type, [])
        src_pair = [int(source[0]), int(source[1])] if source and len(source) == 2 else None
        tgt_pair = [tr, tc]

        if src_pair and src_pair in lst:
            idx = lst.index(src_pair)
            lst[idx] = tgt_pair
        else:
            lst.append(tgt_pair)

        result = {"ok": True, "component_type": component_type, "positions": lst}
        state.record(
            "tool4_move_component",
            {"component_type": component_type, "source": source, "target": target},
            result,
        )
        return result

    def tool5_read_matrix_state(
        self,
        state: RoomAgentState,
        max_rows: int = 64,
        max_cols: int = 64,
        compress: bool = True,
    ) -> Dict[str, Any]:
        board = state.to_ascii_board(max_rows=max_rows, max_cols=max_cols, compress=compress)
        result = {
            "summary": state.summary(),
            "ascii_board": board,
        }
        state.record(
            "tool5_read_matrix_state",
            {"max_rows": max_rows, "max_cols": max_cols, "compress": compress},
            {"summary": result["summary"]},
        )
        return result


class ReActLightingAgent:
    """
    简化 ReAct 智能体:
    - 先执行工具 1/2/3 形成初稿;
    - 在 while 循环中读取状态(tool5)并让模型决定是否调用 tool4 微调;
    - 最终调用终止动作 finish 并返回总结。
    """

    def __init__(
        self,
        tools: Optional[LightingTools] = None,
        provider: str = "qwen",
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self.tools = tools or LightingTools()
        self.provider = provider.strip().lower()
        self.temperature = float(temperature)
        self.api_key, self.base_url, self.model = _resolve_provider(self.provider, model_name)
        self.client: Optional[OpenAI] = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def run_for_room(
        self,
        state: RoomAgentState,
        max_steps: int = 6,
    ) -> Dict[str, Any]:
        # 初始工具链
        t1 = self.tools.tool1_match_lamp_type(state)
        t2 = self.tools.tool2_select_spec_and_count(state)
        t3 = self.tools.tool3_generate_uniform_coords(state, component_type="lamps")
        self._init_switch(state)

        final_reason = "max_steps_reached"
        for _ in range(max(1, int(max_steps))):
            view = self.tools.tool5_read_matrix_state(state)
            action = self._decide_action(state, view)
            if action.get("action") == "finish":
                final_reason = str(action.get("reason", "done"))
                state.record("finish", {}, {"reason": final_reason, "strategy": action.get("strategy", "")})
                break
            if action.get("action") == "tool4_move_component":
                args = action.get("args", {}) or {}
                self.tools.tool4_move_component(
                    state=state,
                    component_type=str(args.get("component_type", "switches")),
                    source=args.get("source"),
                    target=args.get("target", [0, 0]),
                )
            else:
                final_reason = f"unknown_action:{action.get('action')}"
                break

        return {
            "room_name": state.room_name,
            "selected_lamp_type": t1.get("lamp_type"),
            "lamp_plan": t2,
            "placements": state.placements,
            "tool_calls": len(state.tool_history),
            "finish_reason": final_reason,
            "strategy_summary": self._build_strategy_summary(state),
            "stage_outputs": {
                "tool1": t1,
                "tool2": t2,
                "tool3": t3,
            },
        }

    def _decide_action(self, state: RoomAgentState, view: Dict[str, Any]) -> Dict[str, Any]:
        if self.client is None:
            return {"action": "finish", "reason": "no_llm_key", "strategy": "deterministic tools only"}

        system_prompt = REACT_SYSTEM_PROMPT
        user_prompt = json.dumps(
            {
                "room_state": view["summary"],
                "ascii_board": view["ascii_board"],
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
            obj = _extract_json(content)
            if isinstance(obj, dict) and obj.get("action"):
                return obj
        except Exception:
            pass
        return {"action": "finish", "reason": "llm_parse_failed", "strategy": "fallback finish"}

    def _init_switch(self, state: RoomAgentState) -> None:
        if state.placements.get("switches"):
            return
        rows, cols = state.matrix.shape
        doors = np.argwhere(state.matrix == 2)
        if len(doors) > 0:
            best = None
            best_score = 10**9
            for d in doors:
                r, c = int(d[0]), int(d[1])
                cand = [
                    (r - 1, c),
                    (r + 1, c),
                    (r, c - 1),
                    (r, c + 1),
                ]
                for rr, cc in cand:
                    if 0 <= rr < rows and 0 <= cc < cols and int(state.matrix[rr, cc]) == 1:
                        score = min(rr, rows - 1 - rr, cc, cols - 1 - cc)
                        if score < best_score:
                            best_score = score
                            best = (rr, cc)
            if best is not None:
                state.placements["switches"] = [[int(best[0]), int(best[1])]]
                return

        valid = np.argwhere(state.matrix == 1)
        if len(valid) > 0:
            r, c = valid[0]
            state.placements["switches"] = [[int(r), int(c)]]

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


def _nearest_free_cell(target: Tuple[float, float], free: set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if not free:
        return None
    tr, tc = target
    best = None
    best_dist = 1e18
    for r, c in free:
        d = (float(r) - tr) ** 2 + (float(c) - tc) ** 2
        if d < best_dist:
            best_dist = d
            best = (r, c)
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
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        base_url = os.getenv("OPENROUTER_BASE_URL", "").strip() or "https://openrouter.ai/api/v1"
        model = model_name or os.getenv("CAD_AGENT_OPENROUTER_MODEL", "glm-5").strip()
        return api_key, base_url, model
    raise ValueError(f"unsupported provider: {provider}")


REACT_SYSTEM_PROMPT = """
你是房间照明布置 ReAct 智能体。你必须在每一步只输出一个 JSON 动作，不要输出解释。

可用动作:
1) {"action":"tool4_move_component","args":{"component_type":"lamps|switches","source":[r,c]或null,"target":[r,c]}}
2) {"action":"finish","reason":"终止原因","strategy":"本房间布局策略总结"}

原则:
- 灯具应尽量横平竖直、均匀分散，避免扎堆；
- 开关应靠门且贴边，不可落在门格(D)上；
- 任何 target 必须落在可布置格(值=1)。
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

