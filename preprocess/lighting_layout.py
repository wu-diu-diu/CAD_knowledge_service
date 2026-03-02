"""
房间网格离散化与灯具布置模块
将房间像素轮廓离散为0/1/2网格（2表示门位），并输出灯具像素坐标与CAD坐标。
"""
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from openai import OpenAI

from .coordinate_converter import pixel_to_cad
from .prompt import lighting_layout_prompt, lighting_type_count_prompt


LAMP_CATALOG_DEFAULT: List[Dict[str, Any]] = [
    {
        "灯具类型": "感应式吸顶灯",
        "型号": "LPXDD 002",
        "光通量": "600lm",
        "功率": "6W",
        "厂家": "Alibaba 供应商",
        "购买链接": "https://www.alibaba.com/product-detail/Modern-Intelligent-LED-Induction-Ceiling-Light_1601488477089.html",
    },
    {
        "灯具类型": "防爆灯",
        "型号": "BC9102S-L30",
        "光通量": "6100lm",
        "功率": "60W",
        "厂家": "通明电器 TORMIN",
        "购买链接": "https://i-item.jd.com/100021096200.html",
    },
    {
        "灯具类型": "双管格栅灯",
        "型号": "ML-XTD014E",
        "光通量": "5000lm",
        "功率": "72W",
        "厂家": "Moonlight",
        "购买链接": "https://www.alibaba.com/product-detail/Industrial-Grille-Light-36W-4FT-T8_1600967544017.html",
    },
    {
        "灯具类型": "双管荧光灯",
        "型号": "BAY51-S28XJWF1",
        "光通量": "5000lm",
        "功率": "72W",
        "厂家": "合隆 Helon",
        "购买链接": "https://test-www.mymro.cn:443/u-8W2652.html",
    },
    {
        "灯具类型": "筒灯",
        "型号": "tp2351q",
        "光通量": "900lm",
        "功率": "10.5W",
        "厂家": "tp",
        "购买链接": "https://www.alibaba.com/product-detail/Modern-Aluminum-Recessed-Downlight-Led-Spotlight_1601702044947.html",
    },
]


ROOM_LUX_REQUIREMENTS: Dict[str, int] = {
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


def _is_point(item: Any) -> bool:
    return (
        isinstance(item, (list, tuple))
        and len(item) == 2
        and isinstance(item[0], (int, float, np.integer, np.floating))
        and isinstance(item[1], (int, float, np.integer, np.floating))
    )


def _normalize_shapes(room_shapes: Any) -> List[List[List[float]]]:
    """
    兼容以下结构:
    1) 单个多边形: [[x, y], ...]
    2) 多个多边形: [ [[x, y], ...], [[x, y], ...] ]
    """
    if room_shapes is None:
        return []
    if isinstance(room_shapes, np.ndarray):
        if room_shapes.ndim == 2 and room_shapes.shape[1] == 2:
            return [room_shapes.tolist()]
        if room_shapes.ndim == 3 and room_shapes.shape[2] == 2:
            return [shape.tolist() for shape in room_shapes]
    if not isinstance(room_shapes, (list, tuple)) or not room_shapes:
        return []

    first = room_shapes[0]
    if _is_point(first):
        return [list(room_shapes)]
    if (
        isinstance(first, (list, tuple))
        and first
        and _is_point(first[0])
    ):
        return [list(shape) for shape in room_shapes]
    return []


def _build_room_mask(room_shapes: Any) -> Tuple[Optional[np.ndarray], int, int, int, int]:
    polygons = _normalize_shapes(room_shapes)
    valid_polygons = []
    all_points: List[np.ndarray] = []
    for poly in polygons:
        arr = np.array(poly, dtype=np.int32).reshape(-1, 2)
        if len(arr) >= 3:
            valid_polygons.append(arr)
            all_points.append(arr)

    if not valid_polygons:
        return None, 0, 0, 0, 0

    merged = np.vstack(all_points)
    min_x = int(np.min(merged[:, 0]))
    min_y = int(np.min(merged[:, 1]))
    max_x = int(np.max(merged[:, 0]))
    max_y = int(np.max(merged[:, 1]))

    width = max_x - min_x + 1
    height = max_y - min_y + 1
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in valid_polygons:
        shifted = poly.copy()
        shifted[:, 0] -= min_x
        shifted[:, 1] -= min_y
        cv2.fillPoly(mask, [shifted], 1)
    return mask, min_x, min_y, width, height


def _compute_room_centroid(room_shapes: Any) -> Optional[Tuple[float, float]]:
    polygons = _normalize_shapes(room_shapes)
    if not polygons:
        return None
    pts = []
    for poly in polygons:
        arr = np.array(poly, dtype=np.float32).reshape(-1, 2)
        if len(arr) >= 3:
            pts.append(arr)
    if not pts:
        return None
    merged = np.vstack(pts)
    return float(np.mean(merged[:, 0])), float(np.mean(merged[:, 1]))


def _build_room_single_door_assignment(
    room_rectangles: Dict[str, Any],
    door_assignments: Optional[List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """
    为每个房间挑选1个门:
    - 候选门条件: 房间名在门的(primary_room, secondary_room)中
    - 若有多个候选门: 选门中心离房间几何中心最近的那个
    """
    if not door_assignments:
        return {}

    room_centroids: Dict[str, Tuple[float, float]] = {}
    for room_name, room_shapes in room_rectangles.items():
        centroid = _compute_room_centroid(room_shapes)
        if centroid is not None:
            room_centroids[room_name] = centroid

    result: Dict[str, Dict[str, Any]] = {}
    for room_name in room_rectangles.keys():
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        centroid = room_centroids.get(room_name)
        for door in door_assignments:
            primary = door.get("primary_room")
            secondary = door.get("secondary_room")
            if room_name != primary and room_name != secondary:
                continue

            center = door.get("center")
            if not _is_point(center):
                continue
            cx, cy = float(center[0]), float(center[1])

            if centroid is None:
                dist2 = 0.0
            else:
                dx = cx - centroid[0]
                dy = cy - centroid[1]
                dist2 = float(dx * dx + dy * dy)
            candidates.append((dist2, door))

        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        chosen = candidates[0][1]
        chosen_center = chosen.get("center", [0.0, 0.0])
        result[room_name] = {
            "center": [float(chosen_center[0]), float(chosen_center[1])],
            "bbox": [int(v) for v in chosen.get("bbox", [])] if isinstance(chosen.get("bbox"), list) else [],
            "primary_room": chosen.get("primary_room"),
            "secondary_room": chosen.get("secondary_room"),
        }
    return result


def _mark_door_edge_cells(
    grid: np.ndarray,
    min_x: int,
    min_y: int,
    room_w: int,
    room_h: int,
    cell_size_px: int,
    assigned_door: Dict[str, Any],
) -> Tuple[List[List[int]], Optional[str]]:
    """
    按门方位标注房间边缘网格为2:
    - 先根据门中心与房间四边距离确定方位(left/right/top/bottom)
    - 再把该边上与门bbox投影重叠的一段网格置为2
    """
    if grid.size == 0:
        return [], None
    bbox_x1 = min_x + room_w - 1
    bbox_y1 = min_y + room_h - 1
    bbox = assigned_door.get("bbox", [])
    if isinstance(bbox, list) and len(bbox) == 4:
        bx, by, bw, bh = [int(v) for v in bbox]
        door_x0 = bx
        door_y0 = by
        door_x1 = bx + max(0, bw) - 1
        door_y1 = by + max(0, bh) - 1
        cx = (door_x0 + door_x1) / 2.0
        cy = (door_y0 + door_y1) / 2.0
    else:
        center = assigned_door.get("center")
        if not _is_point(center):
            return [], None
        cx, cy = float(center[0]), float(center[1])
        door_x0 = door_x1 = int(round(cx))
        door_y0 = door_y1 = int(round(cy))

    # 支持门朝外开：门中心可以在房间外。用“门bbox到房间四边的最短距离”判定门方位。
    dist_left = min(abs(door_x0 - min_x), abs(door_x1 - min_x))
    dist_right = min(abs(door_x0 - bbox_x1), abs(door_x1 - bbox_x1))
    dist_top = min(abs(door_y0 - min_y), abs(door_y1 - min_y))
    dist_bottom = min(abs(door_y0 - bbox_y1), abs(door_y1 - bbox_y1))
    side, _ = min(
        [("left", dist_left), ("right", dist_right), ("top", dist_top), ("bottom", dist_bottom)],
        key=lambda x: x[1],
    )

    marked: List[List[int]] = []
    rows, cols = grid.shape
    if side in ("left", "right"):
        col = 0 if side == "left" else cols - 1
        r0 = int(np.clip((door_y0 - min_y) // cell_size_px, 0, rows - 1))
        r1 = int(np.clip((door_y1 - min_y) // cell_size_px, 0, rows - 1))
        if r1 < r0:
            r0, r1 = r1, r0
        for r in range(r0, r1 + 1):
            grid[r, col] = 2
            marked.append([int(r), int(col)])
    else:
        row = 0 if side == "top" else rows - 1
        c0 = int(np.clip((door_x0 - min_x) // cell_size_px, 0, cols - 1))
        c1 = int(np.clip((door_x1 - min_x) // cell_size_px, 0, cols - 1))
        if c1 < c0:
            c0, c1 = c1, c0
        for c in range(c0, c1 + 1):
            grid[row, c] = 2
            marked.append([int(row), int(c)])

    return marked, side


def _mask_to_grid(mask: np.ndarray, cell_size_px: int, occupancy_threshold: float) -> np.ndarray:
    if cell_size_px <= 0:
        raise ValueError(f"cell_size_px必须大于0，当前值: {cell_size_px}")
    h, w = mask.shape
    rows = int(np.ceil(h / float(cell_size_px)))
    cols = int(np.ceil(w / float(cell_size_px)))
    grid = np.zeros((rows, cols), dtype=np.uint8)

    for r in range(rows):
        y0 = r * cell_size_px
        y1 = min((r + 1) * cell_size_px, h)
        for c in range(cols):
            x0 = c * cell_size_px
            x1 = min((c + 1) * cell_size_px, w)
            patch = mask[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            if float(np.mean(patch)) >= occupancy_threshold:
                grid[r, c] = 1
    return grid


def _select_cells_heuristic(grid: np.ndarray, lamp_count: int) -> List[Tuple[int, int]]:
    """
    启发式选灯：
    - 尽量横平竖直（同一行或同一列）
    - 尽量均匀分布在房间主轴方向
    """
    if lamp_count <= 0:
        return []

    candidates = np.argwhere(grid == 1)
    if len(candidates) == 0:
        return []

    if len(candidates) == 1:
        one = (int(candidates[0][0]), int(candidates[0][1]))
        return [one for _ in range(lamp_count)]

    row_span = int(np.max(candidates[:, 0]) - np.min(candidates[:, 0]))
    col_span = int(np.max(candidates[:, 1]) - np.min(candidates[:, 1]))

    selected: List[Tuple[int, int]] = []
    need = int(lamp_count)
    if col_span >= row_span:
        target_row = int(round(float(np.median(candidates[:, 0]))))
        row_band = candidates[np.abs(candidates[:, 0] - target_row) <= 1]
        pool = row_band if len(row_band) > 0 else candidates
        pool = pool[np.argsort(pool[:, 1])]
    else:
        target_col = int(round(float(np.median(candidates[:, 1]))))
        col_band = candidates[np.abs(candidates[:, 1] - target_col) <= 1]
        pool = col_band if len(col_band) > 0 else candidates
        pool = pool[np.argsort(pool[:, 0])]

    if len(pool) == 0:
        pool = candidates

    idxs = np.linspace(0, len(pool) - 1, num=min(need, len(pool)), dtype=int)
    for idx in idxs.tolist():
        cell = (int(pool[idx][0]), int(pool[idx][1]))
        if cell not in selected:
            selected.append(cell)

    if len(selected) < need:
        # 回退：补齐剩余点，优先与已选点保持较大间距
        remaining = [tuple(map(int, rc)) for rc in candidates.tolist() if tuple(map(int, rc)) not in selected]
        while remaining and len(selected) < need:
            if not selected:
                selected.append(remaining.pop(0))
                continue
            best_i = 0
            best_score = -1.0
            for i, cand in enumerate(remaining):
                score = min((cand[0] - s[0]) ** 2 + (cand[1] - s[1]) ** 2 for s in selected)
                if score > best_score:
                    best_score = score
                    best_i = i
            selected.append(remaining.pop(best_i))

    while len(selected) < need:
        selected.append(selected[-1])
    return selected[:need]


def _extract_json_obj(text: str) -> Optional[dict]:
    if not text:
        return None
    fence = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _parse_luminous_flux_to_lm(raw_value: str, default_lm: float = 1000.0) -> float:
    if not raw_value:
        return float(default_lm)
    text = str(raw_value)
    m = re.search(r"(\d+(?:\.\d+)?)\s*lm", text, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m2 = re.search(r"(\d+(?:\.\d+)?)", text)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            pass
    return float(default_lm)


def _get_target_lux_for_room(room_name: str) -> int:
    name = (room_name or "").strip()
    if name in ROOM_LUX_REQUIREMENTS:
        return int(ROOM_LUX_REQUIREMENTS[name])
    for k, v in ROOM_LUX_REQUIREMENTS.items():
        if k and k in name:
            return int(v)
    return 300


def _tool_select_lamp_by_room(room_name: str, default_lamp_type: str = "筒灯") -> Dict[str, Any]:
    """
    工具函数：根据房间名称启发式选择灯具类型。
    """
    name = (room_name or "").strip()
    target_lux = _get_target_lux_for_room(name)

    selected_type = default_lamp_type
    if any(k in name for k in ("配电", "除尘", "高温")):
        selected_type = "防爆灯"
    elif any(k in name for k in ("楼梯", "卫生间", "盟洗")):
        selected_type = "感应式吸顶灯"
    elif target_lux >= 500:
        selected_type = "双管格栅灯"
    elif target_lux >= 300:
        selected_type = "双管荧光灯"
    elif target_lux <= 100:
        selected_type = "筒灯"

    selected = None
    for item in LAMP_CATALOG_DEFAULT:
        if str(item.get("灯具类型", "")).strip() == selected_type:
            selected = dict(item)
            break
    if selected is None and LAMP_CATALOG_DEFAULT:
        selected = dict(LAMP_CATALOG_DEFAULT[0])
    if selected is None:
        selected = {"灯具类型": default_lamp_type, "光通量": "1000lm"}

    phi_lm = _parse_luminous_flux_to_lm(str(selected.get("光通量", "")), default_lm=1000.0)
    selected["光通量_lm"] = float(phi_lm)

    return {
        "room_name": name,
        "target_lux": int(target_lux),
        "selected_lamp": selected,
    }


def _tool_calc_lamp_count(
    area_m2: float,
    phi_lm: float,
    target_lux: float,
    uf: float = 0.6,
    mf: float = 0.8,
) -> Dict[str, Any]:
    """
    工具函数：根据房间面积、灯具光通量和目标照度计算所需灯具数量。
    公式: N = (E * A) / (UF * MF * Φ)，其中:
    - E: 目标照度 (lux)
    - A: 房间面积 (m^2)
    - UF: 使用系数 (通常在0.5到0.8之间)
    - MF: 维护系数 (通常在0.7到0.9之间)
    - Φ: 灯具光通量 (lm)
    """
    area_m2 = max(0.0, float(area_m2))
    phi_lm = max(1e-6, float(phi_lm))
    target_lux = max(1.0, float(target_lux))
    uf = max(0.05, float(uf))
    mf = max(0.05, float(mf))


    n_raw = (target_lux * area_m2) / (uf * mf * phi_lm)
    lamp_count = max(1, min(64, int(math.ceil(n_raw))))
    return {
        "area_m2": area_m2,
        "phi_lm": phi_lm,
        "target_lux": target_lux,
        "uf": uf,
        "mf": mf,
        "lamp_count": int(lamp_count),
    }


def _resolve_llm_provider_config(
    provider: str,
    model_name: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    根据provider解析 API Key / Base URL / Model。
    支持: qwen, deepseek, openrouter（预留）。
    """
    provider = (provider or "qwen").strip().lower()
    model_name = (model_name or "").strip()

    if provider in ("qwen", "dashscope"):
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
        if not base_url:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = model_name or os.getenv("CAD_LIGHTING_QWEN_MODEL", "qwen-plus").strip()
        return api_key, base_url, model

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip()
        if not base_url:
            base_url = "https://api.deepseek.com/v1"
        model = model_name or os.getenv("CAD_LIGHTING_DEEPSEEK_MODEL", "deepseek-chat").strip()
        return api_key, base_url, model

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        base_url = os.getenv("OPENROUTER_BASE_URL", "").strip()
        if not base_url:
            base_url = "https://openrouter.ai/api/v1"
        model = model_name or os.getenv(
            "CAD_LIGHTING_OPENROUTER_MODEL",
            "glm-5",
        ).strip()
        return api_key, base_url, model

    raise ValueError(f"不支持的CAD_LIGHTING_LLM_PROVIDER: {provider}")


def _select_switch_heuristic(
    grid: np.ndarray,
    door_edge_cells: Optional[List[List[int]]] = None,
    door_side: Optional[str] = None,
) -> Optional[Tuple[int, int]]:
    rows, cols = grid.shape

    if door_edge_cells:
        valid = []
        for item in door_edge_cells:
            if _is_point(item):
                r, c = int(item[0]), int(item[1])
                if 0 <= r < rows and 0 <= c < cols:
                    valid.append((r, c))
        if valid:
            return valid[len(valid) // 2]

    edge_cells: List[Tuple[int, int]] = []
    for c in range(cols):
        if int(grid[0, c]) != 0:
            edge_cells.append((0, c))
        if rows > 1 and int(grid[rows - 1, c]) != 0:
            edge_cells.append((rows - 1, c))
    for r in range(1, rows - 1):
        if int(grid[r, 0]) != 0:
            edge_cells.append((r, 0))
        if cols > 1 and int(grid[r, cols - 1]) != 0:
            edge_cells.append((r, cols - 1))

    if not edge_cells:
        return None

    if door_side in ("left", "right"):
        target_col = 0 if door_side == "left" else cols - 1
        same_side = [rc for rc in edge_cells if rc[1] == target_col]
        if same_side:
            return same_side[len(same_side) // 2]
    elif door_side in ("top", "bottom"):
        target_row = 0 if door_side == "top" else rows - 1
        same_side = [rc for rc in edge_cells if rc[0] == target_row]
        if same_side:
            return same_side[len(same_side) // 2]

    return edge_cells[len(edge_cells) // 2]


def _estimate_lamp_count_from_area(
    room_name: str,
    room_area_m2: float,
    default_count: int,
) -> int:
    if room_area_m2 <= 0:
        return max(1, int(default_count))

    name = (room_name or "").strip()
    area_per_lamp = 10.0
    if any(k in name for k in ("厨房", "卫生间", "浴", "书房", "办公室", "工作")):
        area_per_lamp = 8.0
    elif any(k in name for k in ("走廊", "楼梯", "过道", "前厅")):
        area_per_lamp = 12.0

    estimated = int(math.ceil(float(room_area_m2) / area_per_lamp))
    return max(1, min(24, estimated))


def _polygon_area(points: List[List[float]]) -> float:
    if not points or len(points) < 3:
        return 0.0
    arr = np.array(points, dtype=np.float64).reshape(-1, 2)
    x = arr[:, 0]
    y = arr[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _compute_room_area_mm2_from_cad(
    room_shapes: Any,
    cad_params: Dict[str, float],
    image_w: int,
    image_h: int,
) -> float:
    """
    使用房间轮廓点(像素) -> CAD坐标(mm) 后计算面积，返回 mm^2。
    """
    polygons = _normalize_shapes(room_shapes)
    total_area_mm2 = 0.0
    for poly in polygons:
        if not poly or len(poly) < 3:
            continue
        cad_poly: List[List[float]] = []
        for px, py in poly:
            x_cad, y_cad = pixel_to_cad(
                px=float(px),
                py=float(py),
                Xmin=cad_params["Xmin"],
                Ymin=cad_params["Ymin"],
                Xmax=cad_params["Xmax"],
                Ymax=cad_params["Ymax"],
                width=image_w,
                height=image_h,
            )
            cad_poly.append([float(x_cad), float(y_cad)])
        total_area_mm2 += _polygon_area(cad_poly)
    return float(total_area_mm2)


def _complete_lamp_cells(
    selected: List[Tuple[int, int]],
    grid: np.ndarray,
    target_count: int,
) -> List[Tuple[int, int]]:
    target_count = max(0, int(target_count))
    unique_selected: List[Tuple[int, int]] = []
    for cell in selected:
        rc = (int(cell[0]), int(cell[1]))
        if rc not in unique_selected:
            unique_selected.append(rc)

    if target_count == 0:
        return []
    if len(unique_selected) >= target_count:
        return unique_selected[:target_count]

    fallback_cells = _select_cells_heuristic(grid, target_count)
    for rc in fallback_cells:
        rc = (int(rc[0]), int(rc[1]))
        if rc not in unique_selected:
            unique_selected.append(rc)
        if len(unique_selected) >= target_count:
            break

    while len(unique_selected) < target_count:
        unique_selected.append(unique_selected[-1] if unique_selected else (0, 0))
    return unique_selected[:target_count]




def _select_cells_with_llm(
    room_name: str,
    room_area_m2: float,
    grid: np.ndarray,
    suggested_lamp_count: int,
    default_lamp_type: str,
    door_side: Optional[str] = None,
    door_edge_cells: Optional[List[List[int]]] = None,
    stage1_provider: str = "deepseek",
    stage1_model: Optional[str] = None,
    stage2_provider: str = "qwen",
    stage2_model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    use_llm = os.getenv("CAD_LIGHTING_USE_LLM", "0").lower() in ("1", "true", "yes")
    if not use_llm:
        return None

    stage1_provider = (stage1_provider or "deepseek").strip().lower()
    stage2_provider = (stage2_provider or "qwen").strip().lower()

    # 第一阶段: LLM选择灯具类型 + 计算灯具数量
    planned_lamp_type = default_lamp_type
    planned_lamp_count = max(1, int(suggested_lamp_count))
    resolved_stage1_model = stage1_model

    lamp_options: List[str] = []
    lamp_flux_lm_map: Dict[str, float] = {}
    for item in LAMP_CATALOG_DEFAULT:
        lamp_type = str(item.get("灯具类型", "")).strip()
        if not lamp_type:
            continue
        lamp_options.append(lamp_type)
        lamp_flux_lm_map[lamp_type] = _parse_luminous_flux_to_lm(
            str(item.get("光通量", "")),
            default_lm=1000.0,
        )

    target_lux = float(_get_target_lux_for_room(room_name))
    plan_prompt = lighting_type_count_prompt(
        room_name=room_name,
        room_area_m2=float(room_area_m2),
        target_lux=target_lux,
        lamp_flux_lm_map=lamp_flux_lm_map,
    )

    try:
        s1_api_key, s1_base_url, resolved_stage1_model = _resolve_llm_provider_config(
            provider=stage1_provider,
            model_name=stage1_model,
        )
        if not s1_api_key or not resolved_stage1_model:
            raise ValueError("stage1 api key/model missing")

        s1_client = OpenAI(api_key=s1_api_key, base_url=s1_base_url or None)
        meta_completion = s1_client.chat.completions.create(
            model=resolved_stage1_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你必须严格只输出JSON，不要输出思考过程和解释。"
                        "输出格式固定为: {\"lamp_type\":str,\"lamp_count\":int}"
                    ),
                },
                {"role": "user", "content": plan_prompt},
            ],
            temperature=0,
        )
        meta_content = meta_completion.choices[0].message.content if meta_completion.choices else ""
        meta_obj = _extract_json_obj(meta_content or "")
        if isinstance(meta_obj, dict):
            lamp_type_raw = str(meta_obj.get("lamp_type", "")).strip()
            if lamp_type_raw in lamp_options:
                planned_lamp_type = lamp_type_raw
            lamp_count_raw = meta_obj.get("lamp_count")
            if isinstance(lamp_count_raw, int):
                planned_lamp_count = int(lamp_count_raw)
    except Exception:
        pass

    planned_lamp_count = max(1, min(64, int(planned_lamp_count)))

    available = np.argwhere(grid == 1)  ## availabel 表示可放灯的位置
    switch_count = 1 if (door_edge_cells and len(door_edge_cells) > 0) else 0
    if len(available) == 0:  ## 如果没有可放灯的位置，直接返回结果（仅包含灯具类型和数量，位置留空）
        return {
            "lamp_type": planned_lamp_type,
            "lamp_count": 0,
            "lamps": [],
            "switch": _select_switch_heuristic(grid, door_edge_cells, door_side) if switch_count > 0 else None,
            "lamp_lm": lamp_flux_lm_map.get(planned_lamp_type, 1000.0),
        }

    # 第二阶段: LLM网格布点
    resolved_stage2_model = stage2_model
    obj = None
    try:
        s2_api_key, s2_base_url, resolved_stage2_model = _resolve_llm_provider_config(
            provider=stage2_provider,
            model_name=stage2_model,
        )
        if not s2_api_key or not resolved_stage2_model:
            raise ValueError("stage2 api key/model missing")

        s2_client = OpenAI(api_key=s2_api_key, base_url=s2_base_url or None)

        matrix_data = grid.tolist()
        user_prompt = lighting_layout_prompt(
            lamp_count=int(planned_lamp_count),
            switch_count=int(switch_count),
            matrix_col=int(grid.shape[1]),
            matrix_row=int(grid.shape[0]),
            matrix=matrix_data,
        )
        system_prompt = (
            "你必须严格只输出JSON，不要输出思考过程和解释。"
            "输出格式固定为: "
            "{\"switches\":[[row,col],...],\"lights\":[[row,col],...]}。"
            f"其中 lights 数量必须等于 {int(planned_lamp_count)}，"
            f"switches 数量必须等于 {int(switch_count)}。"
        )

        completion = s2_client.chat.completions.create(
            model=resolved_stage2_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = completion.choices[0].message.content if completion.choices else ""
        obj = _extract_json_obj(content or "")
    except Exception:
        pass

    valid = {(int(r), int(c)) for r, c in available.tolist()}
    picked: List[Tuple[int, int]] = []
    if isinstance(obj, dict):
        lamp_items = obj.get("lights")
        if not isinstance(lamp_items, list):
            lamp_items = obj.get("lamps", [])
        if not isinstance(lamp_items, list):
            lamp_items = obj.get("positions", [])
        if not isinstance(lamp_items, list):
            lamp_items = []

        for p in lamp_items:
            row = None
            col = None
            if isinstance(p, dict):
                row = p.get("row")
                col = p.get("col")
            elif _is_point(p):
                row = int(p[0])
                col = int(p[1])
            if isinstance(row, int) and isinstance(col, int) and (row, col) in valid:
                picked.append((row, col))

    if not picked and len(valid) > 0:
        picked = _select_cells_heuristic(grid, planned_lamp_count)
    picked = _complete_lamp_cells(picked, grid, planned_lamp_count)

    switch_cell: Optional[Tuple[int, int]] = None
    if switch_count > 0 and isinstance(obj, dict):
        rows, cols = grid.shape
        switch_items = obj.get("switches")
        if not isinstance(switch_items, list):
            switch_items = []
        if switch_items:
            sw = switch_items[0]
            sr = None
            sc = None
            if isinstance(sw, dict):
                sr = sw.get("row")
                sc = sw.get("col")
            elif _is_point(sw):
                sr = int(sw[0])
                sc = int(sw[1])
            if isinstance(sr, int) and isinstance(sc, int):
                if 0 <= sr < rows and 0 <= sc < cols:
                    if (sr in (0, rows - 1) or sc in (0, cols - 1)) and int(grid[sr, sc]) != 0:
                        switch_cell = (sr, sc)

    if switch_count > 0 and switch_cell is None:
        switch_cell = _select_switch_heuristic(grid, door_edge_cells, door_side)

    return {
        "lamp_type": planned_lamp_type,
        "lamp_count": int(len(picked)),
        "lamps": picked,
        "switch": switch_cell,
        "lamp_lm": lamp_flux_lm_map.get(planned_lamp_type, 1000.0),
    }


def _grid_cell_to_pixel(
    row: int,
    col: int,
    min_x: int,
    min_y: int,
    room_w: int,
    room_h: int,
    cell_size_px: int,
) -> Tuple[float, float]:
    x0 = col * cell_size_px
    x1 = min((col + 1) * cell_size_px, room_w)
    y0 = row * cell_size_px
    y1 = min((row + 1) * cell_size_px, room_h)

    center_x_local = (x0 + x1 - 1) / 2.0
    center_y_local = (y0 + y1 - 1) / 2.0
    return min_x + center_x_local, min_y + center_y_local


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r"[\\/:*?\"<>|]", "_", name)
    safe = re.sub(r"\s+", "_", safe).strip("_")
    return safe or "room"


def _is_regular_room_shape(room_shapes: Any) -> bool:
    polygons = _normalize_shapes(room_shapes)
    if len(polygons) != 1:
        return False
    return len(polygons[0]) == 4


def _draw_room_grid_on_overlay(
    overlay: np.ndarray,
    room_name: str,
    room_info: Dict[str, Any],
    offset_x: int = 0,
    offset_y: int = 0,
    draw_label: bool = True,
) -> None:
    matrix = np.array(room_info.get("matrix", []), dtype=np.uint8)
    bbox = room_info.get("bbox_pixel", [])
    cell_size_px = int(room_info.get("cell_size_px", 40))
    if matrix.size == 0 or len(bbox) != 4:
        return

    color_placeable = (60, 180, 75)   # BGR, 1
    color_blocked = (60, 60, 220)     # BGR, 0
    color_door = (255, 0, 0)          # BGR, 2
    color_lamp = (0, 0, 0)            # BGR, lamp cell
    color_switch = (128, 128, 128)    # BGR, switch cell
    color_grid = (160, 160, 160)
    color_bbox = (255, 255, 255)

    min_x, min_y, max_x, max_y = [int(v) for v in bbox]
    min_x -= offset_x
    max_x -= offset_x
    min_y -= offset_y
    max_y -= offset_y

    lamp_positions = room_info.get("lamp_grid_positions", []) or []
    switch_positions = room_info.get("switch_grid_positions", []) or []
    lamp_cells = set()
    switch_cells = set()
    for pos in lamp_positions:
        if _is_point(pos):
            lamp_cells.add((int(pos[0]), int(pos[1])))
    for pos in switch_positions:
        if _is_point(pos):
            switch_cells.add((int(pos[0]), int(pos[1])))

    rows, cols = matrix.shape
    for r in range(rows):
        y0 = min_y + r * cell_size_px
        y1 = min(min_y + (r + 1) * cell_size_px, max_y + 1)
        for c in range(cols):
            x0 = min_x + c * cell_size_px
            x1 = min(min_x + (c + 1) * cell_size_px, max_x + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            if (r, c) in switch_cells:
                color = color_switch
            elif (r, c) in lamp_cells:
                color = color_lamp
            else:
                cell_val = int(matrix[r, c])
                if cell_val == 1:
                    color = color_placeable
                elif cell_val == 2:
                    color = color_door
                else:
                    color = color_blocked
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, -1)
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color_grid, 1)

    cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), color_bbox, 2)
    if draw_label:
        label = f"{room_name} | cell={cell_size_px}px | {rows}x{cols}"
        cv2.putText(
            overlay,
            label,
            (min_x + 4, max(20, min_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color_bbox,
            1,
            cv2.LINE_AA,
        )

    # 灯具与开关位置标注（红色英文）
    text_color = (0, 0, 255)
    for i, pos in enumerate(lamp_positions, start=1):
        if not _is_point(pos):
            continue
        r, c = int(pos[0]), int(pos[1])
        if r < 0 or c < 0 or r >= rows or c >= cols:
            continue
        x0 = min_x + c * cell_size_px
        x1 = min(min_x + (c + 1) * cell_size_px, max_x + 1)
        y0 = min_y + r * cell_size_px
        y1 = min(min_y + (r + 1) * cell_size_px, max_y + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        tx = x0 + 2
        ty = min(y1 - 3, y0 + 14)
        cv2.putText(overlay, f"L{i}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    for i, pos in enumerate(switch_positions, start=1):
        if not _is_point(pos):
            continue
        r, c = int(pos[0]), int(pos[1])
        if r < 0 or c < 0 or r >= rows or c >= cols:
            continue
        x0 = min_x + c * cell_size_px
        x1 = min(min_x + (c + 1) * cell_size_px, max_x + 1)
        y0 = min_y + r * cell_size_px
        y1 = min(min_y + (r + 1) * cell_size_px, max_y + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        tx = x0 + 2
        ty = min(y1 - 3, y0 + 14)
        cv2.putText(overlay, f"S{i}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)


def _draw_legend(vis: np.ndarray) -> None:
    color_placeable = (60, 180, 75)   # BGR, 1
    color_blocked = (60, 60, 220)     # BGR, 0
    color_door = (255, 0, 0)          # BGR, 2
    color_lamp = (0, 0, 0)            # BGR, lamp cell
    color_switch = (128, 128, 128)    # BGR, switch cell
    legend_x = 20
    legend_y = 20
    cv2.rectangle(vis, (legend_x, legend_y), (legend_x + 18, legend_y + 18), color_placeable, -1)
    cv2.putText(
        vis,
        "1: placeable",
        (legend_x + 24, legend_y + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(vis, (legend_x, legend_y + 26), (legend_x + 18, legend_y + 44), color_blocked, -1)
    cv2.putText(
        vis,
        "0: blocked",
        (legend_x + 24, legend_y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(vis, (legend_x, legend_y + 52), (legend_x + 18, legend_y + 70), color_door, -1)
    cv2.putText(
        vis,
        "2: door",
        (legend_x + 24, legend_y + 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(vis, (legend_x, legend_y + 78), (legend_x + 18, legend_y + 96), color_lamp, -1)
    cv2.putText(
        vis,
        "L: lamp cell",
        (legend_x + 24, legend_y + 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(vis, (legend_x, legend_y + 104), (legend_x + 18, legend_y + 122), color_switch, -1)
    cv2.putText(
        vis,
        "S: switch cell",
        (legend_x + 24, legend_y + 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _save_grid_visualization_overlay(
    image_path: str,
    grid_dump: Dict[str, Dict[str, Any]],
    output_dir: str,
    alpha: float = 0.35,
) -> Optional[str]:
    """
    将步骤7离散网格可视化按房间拆分保存：
    - 1(可放置): 绿色
    - 0(不可放置): 红色
    - 2(门位置): 蓝色
    - 灯具/开关: 红色英文标注(L1, L2, S1 ...)
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    alpha = max(0.05, min(0.95, float(alpha)))
    vis_dir = os.path.join(output_dir, "step7_room_grid_visualization")
    os.makedirs(vis_dir, exist_ok=True)

    # 所有房间都按单房间输出，便于逐个查看
    for room_name, room_info in grid_dump.items():
        bbox = room_info.get("bbox_pixel", [])
        if len(bbox) != 4:
            continue
        min_x, min_y, max_x, max_y = [int(v) for v in bbox]
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img.shape[1] - 1, max_x)
        max_y = min(img.shape[0] - 1, max_y)
        if max_x <= min_x or max_y <= min_y:
            continue

        crop = img[min_y:max_y + 1, min_x:max_x + 1].copy()
        overlay_crop = crop.copy()
        _draw_room_grid_on_overlay(
            overlay=overlay_crop,
            room_name=room_name,
            room_info=room_info,
            offset_x=min_x,
            offset_y=min_y,
            draw_label=True,
        )
        vis_crop = cv2.addWeighted(overlay_crop, alpha, crop, 1.0 - alpha, 0.0)
        _draw_legend(vis_crop)
        lamp_type = str(room_info.get("lamp_type", "")).strip()
        file_base = f"{room_name}-{lamp_type}" if lamp_type else room_name
        safe_name = _sanitize_filename(file_base)
        crop_output_path = os.path.join(vis_dir, f"{safe_name}.png")
        cv2.imwrite(crop_output_path, vis_crop)

    return vis_dir


def process_room_lighting_layout(
    room_rectangles: Dict[str, Any],
    image_path: str,
    cad_params: Dict[str, float],
    door_assignments: Optional[List[Dict[str, Any]]] = None,
    save_to_file: bool = True,
    ) -> Dict[str, Any]:
    """
    步骤7:
    1) 将每个房间离散为0/1网格，并把门所在方位的边缘格标记为2
    2) 调用LLM(可选)基于房间名+面积选择灯具类型、数量、网格位置与开关位置
    3) 反解像素坐标 -> CAD坐标
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    image_h, image_w = img.shape[:2]
    cell_size_px = max(1, int(os.getenv("CAD_GRID_CELL_SIZE", "40")))
    occupancy_threshold = float(os.getenv("CAD_GRID_OCCUPANCY_THRESHOLD", "0.35"))
    occupancy_threshold = max(0.0, min(1.0, occupancy_threshold))
    default_lamp_count = int(os.getenv("CAD_TEST_LAMP_COUNT", "2"))
    default_lamp_type = os.getenv("CAD_TEST_LAMP_TYPE", "筒灯")
    stage1_provider = os.getenv("CAD_LIGHTING_STAGE1_PROVIDER", "deepseek").strip().lower()
    stage2_provider = os.getenv("CAD_LIGHTING_STAGE2_PROVIDER", "qwen").strip().lower()
    stage1_model = os.getenv("CAD_LIGHTING_STAGE1_MODEL", "").strip() or None
    stage2_model = os.getenv("CAD_LIGHTING_STAGE2_MODEL", "").strip() or None

    room_assigned_doors = _build_room_single_door_assignment(room_rectangles, door_assignments)

    lighting_rooms: Dict[str, Dict[str, Any]] = {}
    grid_dump: Dict[str, Dict[str, Any]] = {}

    for room_name, room_shapes in room_rectangles.items():
        mask, min_x, min_y, room_w, room_h = _build_room_mask(room_shapes)
        if mask is None:
            continue

        is_regular = _is_regular_room_shape(room_shapes)
        grid = _mask_to_grid(mask, cell_size_px, occupancy_threshold)
        room_area_px = float(np.count_nonzero(mask))
        room_area_mm2 = _compute_room_area_mm2_from_cad(
            room_shapes=room_shapes,
            cad_params=cad_params,
            image_w=image_w,
            image_h=image_h,
        )
        room_area_m2 = float(room_area_mm2 / 1_000_000.0)
        estimated_lamp_count = _estimate_lamp_count_from_area(
            room_name=room_name,
            room_area_m2=room_area_m2,
            default_count=default_lamp_count,
        )
        assigned_door = room_assigned_doors.get(room_name)
        door_grid_position = None
        door_edge_cells: List[List[int]] = []
        door_side = None
        if assigned_door and grid.size > 0:
            door_edge_cells, door_side = _mark_door_edge_cells(
                grid=grid,
                min_x=min_x,
                min_y=min_y,
                room_w=room_w,
                room_h=room_h,
                cell_size_px=cell_size_px,
                assigned_door=assigned_door,
            )
            if door_edge_cells:
                door_grid_position = door_edge_cells[0]

        llm_selection = _select_cells_with_llm(
            room_name=room_name,
            room_area_m2=room_area_m2,
            grid=grid,
            suggested_lamp_count=estimated_lamp_count,
            default_lamp_type=default_lamp_type,
            door_side=door_side,
            door_edge_cells=door_edge_cells,
            stage1_provider=stage1_provider,
            stage1_model=stage1_model,
            stage2_provider=stage2_provider,
            stage2_model=stage2_model,
        )
        placement_method = "llm"
        if llm_selection is None:
            selected_lamp_type = default_lamp_type
            selected_lamp_cells = _select_cells_heuristic(grid, estimated_lamp_count)
            switch_cell = _select_switch_heuristic(
                grid=grid,
                door_edge_cells=door_edge_cells,
                door_side=door_side,
            )
            placement_method = "heuristic"
        else:
            selected_lamp_type = str(llm_selection.get("lamp_type", default_lamp_type))
            selected_lamp_cells = llm_selection.get("lamps", []) or []
            switch_cell = llm_selection.get("switch")
        print(
            f"[step7] room='{room_name}' "
            f"stage1={stage1_provider}/{stage1_model} "
            f"stage2={stage2_provider}/{stage2_model} "
            f"lamp_type={selected_lamp_type} lamp_count={len(selected_lamp_cells)} switch={'yes' if switch_cell else 'no'} lamp_lm={llm_selection.get('lamp_lm')} "
        )

        lamp_grid_positions: List[List[int]] = []
        lamp_pixel_positions: List[List[float]] = []
        lamp_cad_positions: List[List[float]] = []
        for row, col in selected_lamp_cells:
            px, py = _grid_cell_to_pixel(
                row=row,
                col=col,
                min_x=min_x,
                min_y=min_y,
                room_w=room_w,
                room_h=room_h,
                cell_size_px=cell_size_px,
            )
            x_cad, y_cad = pixel_to_cad(
                px=px,
                py=py,
                Xmin=cad_params["Xmin"],
                Ymin=cad_params["Ymin"],
                Xmax=cad_params["Xmax"],
                Ymax=cad_params["Ymax"],
                width=image_w,
                height=image_h,
            )
            lamp_grid_positions.append([int(row), int(col)])
            lamp_pixel_positions.append([float(px), float(py)])
            lamp_cad_positions.append([float(x_cad), float(y_cad)])

        lamps = {
            "lamp_type": selected_lamp_type,
            "count": int(len(lamp_grid_positions)),
            "grid_positions": lamp_grid_positions,
            "pixel_positions": lamp_pixel_positions,
            "cad_positions": lamp_cad_positions,
        }

        switch_info = None
        if (
            isinstance(switch_cell, (list, tuple))
            and len(switch_cell) == 2
            and isinstance(switch_cell[0], (int, np.integer))
            and isinstance(switch_cell[1], (int, np.integer))
        ):
            srow, scol = int(switch_cell[0]), int(switch_cell[1])
            if 0 <= srow < grid.shape[0] and 0 <= scol < grid.shape[1]:
                spx, spy = _grid_cell_to_pixel(
                    row=srow,
                    col=scol,
                    min_x=min_x,
                    min_y=min_y,
                    room_w=room_w,
                    room_h=room_h,
                    cell_size_px=cell_size_px,
                )
                sx_cad, sy_cad = pixel_to_cad(
                    px=spx,
                    py=spy,
                    Xmin=cad_params["Xmin"],
                    Ymin=cad_params["Ymin"],
                    Xmax=cad_params["Xmax"],
                    Ymax=cad_params["Ymax"],
                    width=image_w,
                    height=image_h,
                )
                switch_info = {
                    "switch_type": "单联开关",
                    "grid_position": [srow, scol],
                    "pixel_position": [float(spx), float(spy)],
                    "cad_position": [float(sx_cad), float(sy_cad)],
                }

        lighting_rooms[room_name] = {
            "room_name": room_name,
            "grid_rows": int(grid.shape[0]),
            "grid_cols": int(grid.shape[1]),
            "cell_size_px": cell_size_px,
            "bbox_pixel": [min_x, min_y, min_x + room_w - 1, min_y + room_h - 1],
            "lamp_count": int(len(lamp_grid_positions)),
            "room_area_px": room_area_px,
            "room_area_mm2": room_area_mm2,
            "room_area_m2": room_area_m2,
            "estimated_lamp_count": int(estimated_lamp_count),
            "is_regular": bool(is_regular),
            "assigned_door": assigned_door,
            "door_grid_position": door_grid_position,
            "door_side": door_side,
            "door_edge_cells": door_edge_cells,
            "placement_method": placement_method,
            "stage1_provider": stage1_provider,
            "stage1_model": stage1_model,
            "stage2_provider": stage2_provider,
            "stage2_model": stage2_model,
            "switch": switch_info,
            "switch_count": 1 if switch_info else 0,
            "switches": [switch_info] if switch_info else [],
            "lamps": lamps,
        }
        grid_dump[room_name] = {
            "room_name": room_name,
            "lamp_type": selected_lamp_type,
            "grid_rows": int(grid.shape[0]),
            "grid_cols": int(grid.shape[1]),
            "cell_size_px": cell_size_px,
            "bbox_pixel": [min_x, min_y, min_x + room_w - 1, min_y + room_h - 1],
            "room_area_px": room_area_px,
            "room_area_mm2": room_area_mm2,
            "room_area_m2": room_area_m2,
            "estimated_lamp_count": int(estimated_lamp_count),
            "is_regular": bool(is_regular),
            "assigned_door": assigned_door,
            "door_grid_position": door_grid_position,
            "door_side": door_side,
            "door_edge_cells": door_edge_cells,
            "lamp_grid_positions": lamp_grid_positions,
            "switch_grid_positions": [switch_info["grid_position"]] if switch_info else [],
            "matrix": grid.tolist(),
        }

    payload = {
        "image_width": int(image_w),
        "image_height": int(image_h),
        "cell_size_px": cell_size_px,
        "rooms": lighting_rooms,
    }

    if save_to_file:
        output_dir = os.getenv("CAD_STEP_OUTPUT_DIR", "images/output")
        os.makedirs(output_dir, exist_ok=True)

        grid_file = os.path.join(output_dir, "step7_room_grid_matrices.json")
        with open(grid_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image_width": int(image_w),
                    "image_height": int(image_h),
                    "cell_size_px": cell_size_px,
                    "occupancy_threshold": occupancy_threshold,
                    "rooms": grid_dump,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        plan_file = os.path.join(output_dir, "step7_lighting_layout.json")
        with open(plan_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        vis_alpha = float(os.getenv("CAD_GRID_VIS_ALPHA", "0.35"))
        vis_path = _save_grid_visualization_overlay(
            image_path=image_path,
            grid_dump=grid_dump,
            output_dir=output_dir,
            alpha=vis_alpha,
        )

        print(f"步骤7网格矩阵已保存: {grid_file}")
        print(f"步骤7灯具布置已保存: {plan_file}")
        if vis_path:
            print(f"步骤7网格可视化已保存: {vis_path}")

    return payload
