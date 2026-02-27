"""
房间网格离散化与灯具布置模块
将房间像素轮廓离散为0/1网格，并输出灯具像素坐标与CAD坐标。
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from openai import OpenAI

from .coordinate_converter import pixel_to_cad


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
    启发式选点：优先选择彼此距离较远、覆盖房间不同区域、避免集中在同一角落的点。
    1) 先选一个点作为基准（如最接近中心的可放置点）
    2) 再选一个点使其与第一个点距离最远
    3) 若需要更多点，继续在剩余候选点中选择与已选点平均距离最大的点，直到满足数量
    """
    candidates = np.argwhere(grid == 1)
    if len(candidates) == 0:
        return []

    if len(candidates) == 1:
        one = (int(candidates[0][0]), int(candidates[0][1]))
        return [one for _ in range(lamp_count)]

    centroid = np.mean(candidates, axis=0)
    dist_to_centroid = np.sum((candidates - centroid) ** 2, axis=1)
    first_idx = int(np.argmin(dist_to_centroid))
    first = candidates[first_idx]

    d_first = np.sum((candidates - first) ** 2, axis=1)
    second_idx = int(np.argmax(d_first))
    second = candidates[second_idx]

    selected = [
        (int(first[0]), int(first[1])),
        (int(second[0]), int(second[1])),
    ]
    while len(selected) < lamp_count:
        selected.append(selected[-1])
    return selected[:lamp_count]


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
            "minimax/minimax-m2.5",
        ).strip()
        return api_key, base_url, model

    raise ValueError(f"不支持的CAD_LIGHTING_LLM_PROVIDER: {provider}")


def _select_cells_with_llm(
    room_name: str,
    grid: np.ndarray,
    lamp_count: int,
    provider: str = "qwen",
    model_name: Optional[str] = None,
) -> Optional[List[Tuple[int, int]]]:
    use_llm = os.getenv("CAD_LIGHTING_USE_LLM", "0").lower() in ("1", "true", "yes")
    if not use_llm:
        return None

    try:
        api_key, base_url, resolved_model_name = _resolve_llm_provider_config(
            provider=provider,
            model_name=model_name,
        )
    except Exception as exc:
        print(f"LLM provider配置错误({exc})，回退启发式选点。")
        return None

    if not api_key:
        print(f"CAD_LIGHTING_USE_LLM=1 但provider={provider}未配置API Key，回退启发式选点。")
        return None
    if not resolved_model_name:
        print(f"provider={provider}未配置模型名，回退启发式选点。")
        return None

    available = np.argwhere(grid == 1)
    if len(available) == 0:
        return []

    payload = {
        "room_name": room_name,
        "lamp_count": lamp_count,
        "grid_rows": int(grid.shape[0]),
        "grid_cols": int(grid.shape[1]),
        "matrix": grid.tolist(),
    }
    system_prompt = (
        "你是照明布置助手。输入是房间可布置矩阵(1可放置,0不可放置)。"
        "请输出JSON: {\"positions\":[{\"row\":int,\"col\":int},...] }。"
        "目标是让照明分布尽量均匀：优先选择彼此距离较远、覆盖房间不同区域、避免集中在同一角落的点；"
        "若房间近似对称，优先给出对称分布。"
        "必须输出恰好lamp_count个位置，且位置必须落在值为1的网格。"
        "不要输出任何解释文字，只输出JSON。"
    )

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
        )
        request_kwargs: Dict[str, Any] = {
            "model": resolved_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0,
        }
        # OpenRouter: 预留推理模式开关，默认关闭。
        if provider == "openrouter":
            reasoning_enabled = os.getenv(
                "CAD_LIGHTING_OPENROUTER_REASONING_ENABLED", "1"
            ).lower() in ("1", "true", "yes")
            request_kwargs["extra_body"] = {"reasoning": {"enabled": reasoning_enabled}}

        completion = client.chat.completions.create(**request_kwargs)
        content = completion.choices[0].message.content if completion.choices else ""
    except Exception as exc:
        print(f"LLM调用失败({exc})，回退启发式选点。")
        return None
    obj = _extract_json_obj(content or "")
    if not obj or "positions" not in obj or not isinstance(obj["positions"], list):
        print("LLM返回格式异常，回退启发式选点。")
        return None

    valid = {(int(r), int(c)) for r, c in available.tolist()}
    picked: List[Tuple[int, int]] = []
    for p in obj["positions"]:
        if not isinstance(p, dict):
            continue
        row = p.get("row")
        col = p.get("col")
        if not isinstance(row, int) or not isinstance(col, int):
            continue
        if (row, col) in valid:
            picked.append((row, col))

    if not picked:
        return None
    while len(picked) < lamp_count:
        picked.append(picked[-1])
    return picked[:lamp_count]


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
    color_grid = (160, 160, 160)
    color_bbox = (255, 255, 255)

    min_x, min_y, max_x, max_y = [int(v) for v in bbox]
    min_x -= offset_x
    max_x -= offset_x
    min_y -= offset_y
    max_y -= offset_y

    rows, cols = matrix.shape
    for r in range(rows):
        y0 = min_y + r * cell_size_px
        y1 = min(min_y + (r + 1) * cell_size_px, max_y + 1)
        for c in range(cols):
            x0 = min_x + c * cell_size_px
            x1 = min(min_x + (c + 1) * cell_size_px, max_x + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            color = color_placeable if int(matrix[r, c]) == 1 else color_blocked
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


def _draw_legend(vis: np.ndarray) -> None:
    color_placeable = (60, 180, 75)   # BGR, 1
    color_blocked = (60, 60, 220)     # BGR, 0
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


def _save_grid_visualization_overlay(
    image_path: str,
    grid_dump: Dict[str, Dict[str, Any]],
    output_dir: str,
    alpha: float = 0.35,
) -> Optional[str]:
    """
    将步骤7离散网格可视化叠加在原图上：
    - 1(可放置): 绿色
    - 0(不可放置): 红色
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    alpha = max(0.05, min(0.95, float(alpha)))
    vis_dir = os.path.join(output_dir, "step7_room_grid_visualization")
    os.makedirs(vis_dir, exist_ok=True)

    regular_rooms = {
        name: info for name, info in grid_dump.items()
        if bool(info.get("is_regular", True))
    }
    irregular_rooms = {
        name: info for name, info in grid_dump.items()
        if not bool(info.get("is_regular", True))
    }

    # 规则房间：合并到同一张图
    if regular_rooms:
        overlay_regular = img.copy()
        for room_name, room_info in regular_rooms.items():
            _draw_room_grid_on_overlay(
                overlay=overlay_regular,
                room_name=room_name,
                room_info=room_info,
                offset_x=0,
                offset_y=0,
                draw_label=True,
            )
        vis_regular = cv2.addWeighted(overlay_regular, alpha, img, 1.0 - alpha, 0.0)
        _draw_legend(vis_regular)
        regular_output_path = os.path.join(vis_dir, "regular_rooms_grid.png")
        cv2.imwrite(regular_output_path, vis_regular)
    else:
        regular_output_path = None

    # 不规则房间：逐个从原图裁剪后单独可视化
    for idx, (room_name, room_info) in enumerate(irregular_rooms.items(), start=1):
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
        safe_name = _sanitize_filename(room_name)
        crop_output_path = os.path.join(vis_dir, f"irregular_{idx:02d}_{safe_name}.png")
        cv2.imwrite(crop_output_path, vis_crop)

    return vis_dir


def process_room_lighting_layout(
    room_rectangles: Dict[str, Any],
    image_path: str,
    cad_params: Dict[str, float],
    save_to_file: bool = True,
) -> Dict[str, Any]:
    """
    步骤7:
    1) 将每个房间离散为0/1网格
    2) 调用LLM(可选)或启发式算法生成2个灯具位置
    3) 反解像素坐标 -> CAD坐标
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    image_h, image_w = img.shape[:2]
    cell_size_px = max(1, int(os.getenv("CAD_GRID_CELL_SIZE", "40")))
    occupancy_threshold = float(os.getenv("CAD_GRID_OCCUPANCY_THRESHOLD", "0.35"))
    occupancy_threshold = max(0.0, min(1.0, occupancy_threshold))
    lamp_count = int(os.getenv("CAD_TEST_LAMP_COUNT", "2"))
    lamp_type = os.getenv("CAD_TEST_LAMP_TYPE", "筒灯")
    llm_provider = os.getenv("CAD_LIGHTING_LLM_PROVIDER", "qwen").strip().lower()
    llm_model = os.getenv("CAD_LIGHTING_LLM_MODEL", "").strip() or None

    lighting_rooms: Dict[str, Dict[str, Any]] = {}
    grid_dump: Dict[str, Dict[str, Any]] = {}

    for room_name, room_shapes in room_rectangles.items():
        mask, min_x, min_y, room_w, room_h = _build_room_mask(room_shapes)
        if mask is None:
            continue

        is_regular = _is_regular_room_shape(room_shapes)
        grid = _mask_to_grid(mask, cell_size_px, occupancy_threshold)
        selected_cells = _select_cells_with_llm(
            room_name=room_name,
            grid=grid,
            lamp_count=lamp_count,
            provider=llm_provider,
            model_name=llm_model,
        )
        placement_method = "llm"
        if selected_cells is None:
            selected_cells = _select_cells_heuristic(grid, lamp_count)
            placement_method = "heuristic"

        if placement_method == "llm":
            model_display = llm_model or "provider_default"
            print(
                f"[step7] room='{room_name}' placement=LLM "
                f"provider={llm_provider} model={model_display} "
                f"lamps={len(selected_cells)}"
            )
        else:
            print(
                f"[step7] room='{room_name}' placement=heuristic "
                f"lamps={len(selected_cells)}"
            )

        lamps = []
        for row, col in selected_cells:
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
            lamps.append(
                {
                    "lamp_type": lamp_type,
                    "grid_position": [int(row), int(col)],
                    "pixel_position": [float(px), float(py)],
                    "cad_position": [float(x_cad), float(y_cad)],
                }
            )

        lighting_rooms[room_name] = {
            "room_name": room_name,
            "grid_rows": int(grid.shape[0]),
            "grid_cols": int(grid.shape[1]),
            "cell_size_px": cell_size_px,
            "bbox_pixel": [min_x, min_y, min_x + room_w - 1, min_y + room_h - 1],
            "lamp_count": len(lamps),
            "is_regular": bool(is_regular),
            "placement_method": placement_method,
            "lamps": lamps,
        }
        grid_dump[room_name] = {
            "room_name": room_name,
            "grid_rows": int(grid.shape[0]),
            "grid_cols": int(grid.shape[1]),
            "cell_size_px": cell_size_px,
            "bbox_pixel": [min_x, min_y, min_x + room_w - 1, min_y + room_h - 1],
            "is_regular": bool(is_regular),
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
