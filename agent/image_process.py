"""
Agent 图像处理与执行入口。

目标:
1) 复用 find_all 的 step1-step5（输出格式与内容保持一致）;
2) 生成离散网格(0/1/2)作为 Agent 初始输入;
3) 调用 ReAct Agent 完成灯具/开关布局，并按 step8 同格式生成布线可视化;
4) 将 Agent 最终绘制结果作为 step6 输出到根目录 `agent_output/`。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    _save_grid_visualization_overlay,
)
from preprocess.ocr_extractor import extract_text_boxes
from preprocess.wiring_layout import process_room_wiring_layout

try:
    from .react_agent import ReActLightingAgent, RoomAgentState
except ImportError:
    from agent.react_agent import ReActLightingAgent, RoomAgentState


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
        grid_dump: Dict[str, Dict[str, Any]] = {}
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

            grid_dump[room_name] = {
                "room_name": room_name,
                "lamp_type": "",
                "grid_rows": room_payload["grid_rows"],
                "grid_cols": room_payload["grid_cols"],
                "cell_size_px": room_payload["cell_size_px"],
                "bbox_pixel": room_payload["bbox_pixel"],
                "room_area_px": room_payload["room_area_px"],
                "room_area_mm2": room_payload["room_area_mm2"],
                "room_area_m2": room_payload["room_area_m2"],
                "is_regular": room_payload["is_regular"],
                "assigned_door": room_payload["assigned_door"],
                "door_side": room_payload["door_side"],
                "door_edge_cells": room_payload["door_edge_cells"],
                "lamp_grid_positions": [],
                "switch_grid_positions": [],
                "matrix": room_payload["matrix"],
            }

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

            # 网格可视化（仅用于初始输入检查）
            vis_alpha = float(os.getenv("CAD_GRID_VIS_ALPHA", "0.35"))
            vis_dir = _save_grid_visualization_overlay(
                image_path=str(image_path),
                grid_dump=grid_dump,
                output_dir=str(output_dir),
                alpha=vis_alpha,
            )
            if vis_dir:
                src_dir = Path(vis_dir)
                dst_dir = output_dir / "agent_initial_grid_visualization"
                if src_dir.exists():
                    if dst_dir.exists():
                        shutil.rmtree(dst_dir)
                    src_dir.rename(dst_dir)
                print(f"[agent-image] saved visualization: {dst_dir}")

        return payload
    finally:
        if prev_output_dir is None:
            os.environ.pop("CAD_STEP_OUTPUT_DIR", None)
        else:
            os.environ["CAD_STEP_OUTPUT_DIR"] = prev_output_dir


def run_agent_design(
    initial_input: Dict[str, Any],
    image_path: Path,
    cad_params: Dict[str, float],
    output_dir: Path,
    provider: str = "qwen",
    model_name: Optional[str] = None,
    max_steps: int = 8,
) -> Dict[str, Any]:
    """
    执行 Agent 布局并输出最终 step6 结果:
    - 每个房间运行 ReAct agent 得到灯具/开关网格坐标
    - 调用 step8 布线逻辑生成线路
    - 将 step8 同格式结果落盘为 step6 命名
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rooms = initial_input.get("rooms", {}) or {}
    image_w = int(initial_input.get("image_width", 0))
    image_h = int(initial_input.get("image_height", 0))

    agent = ReActLightingAgent(
        provider=provider,
        model_name=model_name,
        log_dir=str(Path("logs")),
    )

    room_plans: Dict[str, Dict[str, Any]] = {}
    rooms_internal: Dict[str, Dict[str, Any]] = {}

    for room_name, room_info in rooms.items():
        matrix = np.array(room_info.get("matrix", []), dtype=np.int32)
        if matrix.size == 0:
            continue

        state = RoomAgentState(
            room_name=room_name,
            area_m2=float(room_info.get("room_area_m2", 0.0)),
            matrix=matrix,
        )
        result = agent.run_for_room(state=state, max_steps=max_steps)

        lamps_grid = []
        for p in (result.get("placements", {}) or {}).get("lamps", []):
            if isinstance(p, (list, tuple)) and len(p) == 2:
                lamps_grid.append([int(p[0]), int(p[1])])
        switches_grid = []
        for p in (result.get("placements", {}) or {}).get("switches", []):
            if isinstance(p, (list, tuple)) and len(p) == 2:
                switches_grid.append([int(p[0]), int(p[1])])

        lamp_type = str(
            ((result.get("lamp_plan", {}) or {}).get("selected_lamp", {}) or {}).get(
                "lamp_type",
                result.get("selected_lamp_type", "筒灯"),
            )
        )
        primary_switch = switches_grid[0] if switches_grid else None

        room_plans[room_name] = {
            "room_name": room_name,
            "selected_lamp_type": lamp_type,
            "lamp_count": len(lamps_grid),
            "switch_count": len(switches_grid),
            "lamp_grid_positions": lamps_grid,
            "switch_grid_positions": switches_grid,
            "finish_reason": result.get("finish_reason"),
            "validation": result.get("validation"),
            "log_file": result.get("log_file"),
        }

        rooms_internal[room_name] = {
            "room_name": room_name,
            "grid_rows": int(matrix.shape[0]),
            "grid_cols": int(matrix.shape[1]),
            "cell_size_px": int(room_info.get("cell_size_px", initial_input.get("cell_size_px", 40))),
            "bbox_pixel": list(room_info.get("bbox_pixel", [0, 0, 0, 0])),
            "matrix": room_info.get("matrix", []),
            "lamps": {
                "lamp_type": lamp_type,
                "count": len(lamps_grid),
                "grid_positions": lamps_grid,
            },
            "switch": (
                {
                    "switch_type": "开关",
                    "grid_position": primary_switch,
                }
                if primary_switch
                else None
            ),
            "switch_count": len(switches_grid),
            "switches": [
                {
                    "switch_type": "开关",
                    "grid_position": pos,
                }
                for pos in switches_grid
            ],
        }

    lighting_payload = {
        "image_width": image_w,
        "image_height": image_h,
        "rooms_internal": rooms_internal,
    }

    prev_output_dir = os.environ.get("CAD_STEP_OUTPUT_DIR")
    os.environ["CAD_STEP_OUTPUT_DIR"] = str(output_dir)
    try:
        wiring_payload = process_room_wiring_layout(
            lighting_payload=lighting_payload,
            image_path=str(image_path),
            cad_params=cad_params,
            save_to_file=True,
        )
    finally:
        if prev_output_dir is None:
            os.environ.pop("CAD_STEP_OUTPUT_DIR", None)
        else:
            os.environ["CAD_STEP_OUTPUT_DIR"] = prev_output_dir

    # 将 step8 输出重命名为 step6（格式保持 step8 一致）
    step8_json = output_dir / "step8_wiring_layout.json"
    step6_json = output_dir / "step6_agent_wiring_layout.json"
    if step8_json.exists():
        if step6_json.exists():
            step6_json.unlink()
        step8_json.rename(step6_json)

    step8_vis_dir = output_dir / "step8_room_wiring_visualization"
    step6_vis_dir = output_dir / "step6_room_wiring_visualization"
    if step8_vis_dir.exists():
        if step6_vis_dir.exists():
            shutil.rmtree(step6_vis_dir)
        step8_vis_dir.rename(step6_vis_dir)

    final_payload = {
        "image_path": str(image_path),
        "image_width": image_w,
        "image_height": image_h,
        "rooms": room_plans,
        "wiring_rooms": wiring_payload.get("rooms", {}),
    }
    final_json = output_dir / "step6_agent_layout.json"
    with final_json.open("w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)
    print(f"[agent-image] saved final step6: {final_json}")
    print(f"[agent-image] saved final step6 visualization: {step6_vis_dir}")

    return final_payload


def run_agent_pipeline(
    image_path: Path,
    cad_params: Optional[Dict[str, float]] = None,
    output_dir: Optional[Path] = None,
    provider: str = "qwen",
    model_name: Optional[str] = None,
    max_steps: int = 8,
) -> Dict[str, Any]:
    if cad_params is None:
        cad_params = dict(DEFAULT_CAD_PARAMS)
    if output_dir is None:
        output_dir = REPO_ROOT / "agent_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_input = process_image_to_agent_input(
        image_path=image_path,
        cad_params=cad_params,
        output_dir=output_dir,
        save_to_file=True,
    )
    final_payload = run_agent_design(
        initial_input=initial_input,
        image_path=image_path,
        cad_params=cad_params,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        max_steps=max_steps,
    )
    return {
        "initial_input": initial_input,
        "final_result": final_payload,
    }


def main() -> None:
    default_image = REPO_ROOT / "test_files" / "agent_test.png"
    provider = os.getenv("CAD_AGENT_PROVIDER", "qwen").strip().lower()
    model_name = os.getenv("CAD_AGENT_MODEL", "").strip() or None
    max_steps = int(os.getenv("CAD_AGENT_MAX_STEPS", "8"))

    result = run_agent_pipeline(
        image_path=default_image,
        cad_params=dict(DEFAULT_CAD_PARAMS),
        output_dir=REPO_ROOT / "agent_output",
        provider=provider,
        model_name=model_name,
        max_steps=max_steps,
    )
    print(
        f"[agent-image] done. rooms={len((result.get('initial_input', {}) or {}).get('rooms', {}))}, "
        f"final_rooms={len((result.get('final_result', {}) or {}).get('rooms', {}))}"
    )


if __name__ == "__main__":
    main()
