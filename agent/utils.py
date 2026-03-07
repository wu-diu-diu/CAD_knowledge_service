from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def is_point(item: Any) -> bool:
    return (
        isinstance(item, (list, tuple))
        and len(item) == 2
        and isinstance(item[0], (int, float, np.integer, np.floating))
        and isinstance(item[1], (int, float, np.integer, np.floating))
    )


def count_binary_components(mask: np.ndarray) -> int:
    """
    统计二值 mask 的 4 邻域连通块数量。
    """
    if mask is None:
        return 0
    arr = np.array(mask, dtype=np.uint8)
    if arr.ndim != 2 or arr.size == 0:
        return 0
    height, width = arr.shape
    visited = np.zeros((height, width), dtype=np.uint8)
    count = 0
    for row in range(height):
        for col in range(width):
            if arr[row, col] == 0 or visited[row, col] == 1:
                continue
            count += 1
            stack = [(row, col)]
            visited[row, col] = 1
            while stack:
                cur_row, cur_col = stack.pop()
                for next_row, next_col in (
                    (cur_row - 1, cur_col),
                    (cur_row + 1, cur_col),
                    (cur_row, cur_col - 1),
                    (cur_row, cur_col + 1),
                ):
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and arr[next_row, next_col] == 1
                        and visited[next_row, next_col] == 0
                    ):
                        visited[next_row, next_col] = 1
                        stack.append((next_row, next_col))
    return int(count)


def search_grid_shape(
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


def rle_encode(text: str) -> str:
    if not text:
        return text
    out: List[str] = []
    count = 1
    for idx in range(1, len(text) + 1):
        if idx < len(text) and text[idx] == text[idx - 1]:
            count += 1
        else:
            ch = text[idx - 1]
            out.append(f"{ch}{count}" if count > 1 else ch)
            count = 1
    return "".join(out)


def parse_flux_lm(value: Any, default: float = 1000.0) -> float:
    text = str(value or "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*lm", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return float(default)


def parse_power_w(value: Any) -> Optional[float]:
    text = str(value or "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*W", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None


def get_target_lux(room_name: str, lux_map: Dict[str, int]) -> int:
    name = (room_name or "").strip()
    if name in lux_map:
        return int(lux_map[name])
    for key, value in lux_map.items():
        if key and key in name:
            return int(value)
    return 300


def extract_json(text: str) -> Optional[Dict[str, Any]]:
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


def resolve_provider(provider: str, model_name: Optional[str]) -> Tuple[str, str, str]:
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


def describe_tool_result(tool_name: str, tool_output: Dict[str, Any]) -> str:
    if not isinstance(tool_output, dict):
        return f"{tool_name} 已执行。"

    if tool_name == "tool_lookup_room_requirement":
        return (
            f"房间{tool_output.get('room_name', '')}的目标照度为"
            f"{tool_output.get('target_lux', '?')}lx，推荐灯具类型为"
            f"{tool_output.get('lamp_type', '未知')}。"
        )
    if tool_name == "tool_estimate_component_count":
        return (
            f"估计灯具数量{tool_output.get('lamp_count', 0)}个，开关"
            f"{tool_output.get('switch_count', 0)}个，推荐阵列"
            f"{tool_output.get('grid_rows', '?')}x{tool_output.get('grid_cols', '?')}，"
            f"房间判定为{'规则' if tool_output.get('is_regular') else '不规则'}。"
        )
    if tool_name == "tool_calc_required_flux_per_lamp":
        return (
            f"按房间面积{float(tool_output.get('area_m2', 0.0)):.2f}m2、目标照度"
            f"{tool_output.get('target_lux', '?')}lx、灯具数量"
            f"{tool_output.get('lamp_count', '?')}个计算，单灯所需光通量约为"
            f"{float(tool_output.get('required_flux_per_lamp_lm', 0.0)):.1f}lm。"
        )
    if tool_name == "tool_retrieve_lamp_model":
        selected = tool_output.get("selected_lamp", {}) or {}
        return (
            f"已选择灯具类型{selected.get('lamp_type', '未知')}，型号"
            f"{selected.get('model', '未知')}，光通量"
            f"{selected.get('flux_lm', '?')}lm，功率{selected.get('power_w', '?')}W。"
        )
    if tool_name == "tool_place_components":
        return (
            f"已布置灯具{tool_output.get('lamp_count', 0)}个，开关"
            f"{tool_output.get('switch_count', 0)}个，灯具布置算法为"
            f"{tool_output.get('lamp_algorithm', 'unknown')}。"
        )
    if tool_name == "tool_validate_layout":
        violations = tool_output.get("violations", []) or []
        codes = [str(item.get("code", "")) for item in violations[:3] if isinstance(item, dict)]
        if len(violations) > 3:
            codes.append("...")
        code_text = ",".join(codes) if codes else "无"
        return (
            f"布局评分{tool_output.get('score', 0)}分，"
            f"{'通过' if tool_output.get('is_valid') else '未通过'}校验，"
            f"违规项为{code_text}。"
        )
    if tool_name == "tool_apply_layout_edit":
        return (
            f"位置调整已应用{tool_output.get('applied', 0)}项，"
            f"{'无错误' if tool_output.get('ok') else '存在错误'}。"
        )
    if tool_name == "tool_generate_wiring":
        return (
            f"布线状态为{tool_output.get('status', 'unknown')}，共生成"
            f"{tool_output.get('route_count', 0)}条路径，不可达节点"
            f"{len(tool_output.get('unreachable_nodes', []) or [])}个。"
        )
    if tool_name == "tool_read_matrix_state":
        summary = tool_output.get("summary", {}) or {}
        placements = summary.get("placements", {}) or {}
        shape = summary.get("matrix_shape", ["?", "?"])
        return (
            f"当前房间为{summary.get('room_name', '')}，矩阵尺寸"
            f"{shape[0]}x{shape[1]}，已放置灯具"
            f"{len(placements.get('lamps', []))}个，开关"
            f"{len(placements.get('switches', []))}个。"
        )
    if tool_name == "init_mode":
        return f"初始化模式为{tool_output.get('status', 'unknown')}。"
    if tool_name == "finish":
        return f"本轮设计结束，结束原因为{tool_output.get('reason', 'done')}。"
    if tool_name == "internal_replan_design":
        return f"已完成一轮规则初稿设计，初始评分为{tool_output.get('validation_score', '?')}分。"
    return f"{tool_name} 已执行。"
