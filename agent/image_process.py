"""
Agent 图像预处理入口。

目标:
1) 复用 find_all 的 step1-step5（输出格式与内容保持一致）;
2) 生成离散网格(0/1/2)作为 Agent 初始输入。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import sys
import cv2
import numpy as np
from .config import REPO_ROOT

from preprocess.bounding_rectangle import process_room_bounding_rectangles
from preprocess.contour_detector import find_all_inner_contours
from preprocess.coordinate_converter import DEFAULT_CAD_PARAMS
from preprocess.door_point_exclusion import process_all_rooms
from preprocess.door_window_detector import find_door_and_window
from preprocess.lighting_layout import (
    _build_room_mask,
    _build_room_single_door_assignment,
    _compute_room_area_mm2_from_cad,
    _is_regular_room_shape,
    _mark_door_edge_cells,
    _mask_to_grid,
)
from preprocess.ocr_extractor import extract_text_boxes


def process_image_to_agent_input(
    image_path: Path,
    cad_params: Optional[Dict[str, float]] = None,
    output_dir: Optional[Path] = None,
    save_to_file: bool = True,
) -> Dict[str, Any]:
    """
    预处理阶段:
    - step1~step5 与 find_all 保持一致;
    - 额外生成 Agent 初始网格输入(0/1/2)。
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    if cad_params is None:
        cad_params = dict(DEFAULT_CAD_PARAMS)

    if output_dir is None:
        output_dir = REPO_ROOT / "agent_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[agent-image] input: {image_path}")
    print(f"[agent-image] output_dir: {output_dir}")

    prev_output_dir = os.environ.get("CAD_STEP_OUTPUT_DIR")
    os.environ["CAD_STEP_OUTPUT_DIR"] = str(output_dir)

    try:
        # step1
        print("[agent-image] step1: extract_text_boxes")
        room_dict = extract_text_boxes(str(image_path))

        # step2
        print("[agent-image] step2: find_all_inner_contours")
        _, approx_points, room_door_candidates = find_all_inner_contours(
            str(image_path),
            room_dict,
        )

        # step3
        print("[agent-image] step3: find_door_and_window")
        doors_and_windows = find_door_and_window(
            str(image_path),
            room_contours_by_name=approx_points,
            room_door_candidates=room_door_candidates,
        )

        # step4
        print("[agent-image] step4: process_all_rooms")
        processed_rooms = process_all_rooms(
            approx_points,
            doors_and_windows,
            str(image_path),
            manhattan_tolerance=4,
            min_distance=5.0,
            collinear_angle_tolerance_deg=10.0,
            dp_epsilon_ratio=0.004,
        )

        # step5
        print("[agent-image] step5: process_room_bounding_rectangles")
        room_rectangles = process_room_bounding_rectangles(processed_rooms, str(image_path))

        # 离散化(Agent初始输入)
        print("[agent-image] build initial grid matrix(0/1/2)")
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {image_path}")
        image_h, image_w = img.shape[:2]

        cell_size_px = max(1, int(os.getenv("CAD_GRID_CELL_SIZE", "40")))
        occupancy_threshold = float(os.getenv("CAD_GRID_OCCUPANCY_THRESHOLD", "0.35"))
        occupancy_threshold = max(0.0, min(1.0, occupancy_threshold))

        room_assigned_doors = _build_room_single_door_assignment(
            room_rectangles,
            doors_and_windows.get("door_assignments", []),
        )

        rooms: Dict[str, Dict[str, Any]] = {}
        for room_name, room_shapes in room_rectangles.items():
            mask, min_x, min_y, room_w, room_h = _build_room_mask(room_shapes)
            if mask is None:
                continue

            grid = _mask_to_grid(mask, cell_size_px, occupancy_threshold)
            assigned_door = room_assigned_doors.get(room_name)
            door_edge_cells = []
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

            room_area_px = float(np.count_nonzero(mask))
            room_area_mm2 = _compute_room_area_mm2_from_cad(
                room_shapes=room_shapes,
                cad_params=cad_params,
                image_w=image_w,
                image_h=image_h,
            )
            room_area_m2 = float(room_area_mm2 / 1_000_000.0)

            room_payload = {
                "room_name": room_name,
                "grid_rows": int(grid.shape[0]),
                "grid_cols": int(grid.shape[1]),
                "cell_size_px": int(cell_size_px),
                "bbox_pixel": [int(min_x), int(min_y), int(min_x + room_w - 1), int(min_y + room_h - 1)],
                "matrix": grid.astype(np.uint8).tolist(),
                "is_regular": bool(_is_regular_room_shape(room_shapes)),
                "door_side": door_side,
                "door_edge_cells": [[int(r), int(c)] for r, c in door_edge_cells],
                "assigned_door": assigned_door,
                "room_area_px": float(room_area_px),
                "room_area_mm2": float(room_area_mm2),
                "room_area_m2": float(room_area_m2),
            }
            rooms[room_name] = room_payload

        payload = {
            "image_path": str(image_path),
            "image_width": int(image_w),
            "image_height": int(image_h),
            "cell_size_px": int(cell_size_px),
            "occupancy_threshold": float(occupancy_threshold),
            "rooms": rooms,
        }

        if save_to_file:
            output_json = output_dir / "agent_initial_input.json"
            with output_json.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[agent-image] saved: {output_json}")

        return payload
    finally:
        if prev_output_dir is None:
            os.environ.pop("CAD_STEP_OUTPUT_DIR", None)
        else:
            os.environ["CAD_STEP_OUTPUT_DIR"] = prev_output_dir
