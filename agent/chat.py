from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from preprocess.coordinate_converter import DEFAULT_CAD_PARAMS, pixel_to_cad
from preprocess.lighting_layout import _grid_cell_to_pixel, _sanitize_filename, _save_grid_visualization_overlay
from preprocess.wiring_layout import process_room_wiring_layout

try:
    from .config import REPO_ROOT
    from .image_process import process_image_to_agent_input
    from .mini_model import MiniMaxLightingAgent
    from .react_agent import ReActLightingAgent
    from .state import RoomAgentState
except ImportError:
    from agent.config import REPO_ROOT
    from agent.image_process import process_image_to_agent_input
    from agent.mini_model import MiniMaxLightingAgent
    from agent.react_agent import ReActLightingAgent
    from agent.state import RoomAgentState


def _create_timestamped_output_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / ts
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir
    idx = 1
    while True:
        candidate = base_dir / f"{ts}_{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        idx += 1


def _build_agent(
    agent_type: str,
    provider: str,
    model_name: Optional[str],
    log_dir: Path,
):
    resolved_agent_type = (agent_type or "react").strip().lower()
    if resolved_agent_type == "minimax":
        return MiniMaxLightingAgent(
            model_name=model_name,
            log_dir=str(log_dir),
        )
    if resolved_agent_type == "react":
        return ReActLightingAgent(
            provider=provider,
            model_name=model_name,
            log_dir=str(log_dir),
        )
    raise ValueError(f"unsupported agent_type: {agent_type}. Use 'react' or 'minimax'.")


def _build_room_states(initial_input: Dict[str, Any]) -> Dict[str, RoomAgentState]:
    states: Dict[str, RoomAgentState] = {}
    rooms = initial_input.get("rooms", {}) or {}
    for room_name, room_info in rooms.items():
        matrix = np.array(room_info.get("matrix", []), dtype=np.int32)
        if matrix.size == 0:
            continue
        states[room_name] = RoomAgentState(
            room_name=room_name,
            area_m2=float(room_info.get("room_area_m2", 0.0)),
            matrix=matrix,
        )
    return states


def _match_room_name(user_text: str, room_names: List[str], last_room_name: Optional[str]) -> Optional[str]:
    text = (user_text or "").strip()
    matched = [name for name in room_names if name and name in text]
    if matched:
        matched.sort(key=len, reverse=True)
        return matched[0]
    return last_room_name


def _should_reset_layout(user_text: str, state: RoomAgentState) -> bool:
    reset_keywords = ("重绘", "重新", "重做", "重来", "从头", "reset", "清空", "清除")
    if any(keyword in user_text for keyword in reset_keywords):
        return True
    return not bool(state.tool_history or state.placements.get("lamps") or state.placements.get("switches"))


def _lamp_type_from_result(result: Dict[str, Any], state: RoomAgentState) -> str:
    return str(
        ((result.get("lamp_plan", {}) or {}).get("selected_lamp", {}) or {}).get(
            "lamp_type",
            result.get("selected_lamp_type") or state.selected_lamp_type or "筒灯",
        )
    )


def _grid_positions_to_cad_positions(
    grid_positions: List[List[int]],
    bbox_pixel: List[int],
    cell_size_px: int,
    cad_params: Dict[str, float],
    image_width: int,
    image_height: int,
) -> List[List[float]]:
    cad_positions: List[List[float]] = []
    for row, col in grid_positions:
        px, py = _grid_cell_to_pixel(
            int(row),
            int(col),
            int(bbox_pixel[0]),
            int(bbox_pixel[1]),
            int(cell_size_px),
        )
        x_cad, y_cad = pixel_to_cad(
            px,
            py,
            cad_params["Xmin"],
            cad_params["Ymin"],
            cad_params["Xmax"],
            cad_params["Ymax"],
            int(image_width),
            int(image_height),
        )
        cad_positions.append([float(x_cad), float(y_cad)])
    return cad_positions


def _build_room_visual_payloads(
    room_name: str,
    room_info: Dict[str, Any],
    state: RoomAgentState,
    result: Dict[str, Any],
    cad_params: Dict[str, float],
    image_width: int,
    image_height: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    bbox_pixel = list(room_info.get("bbox_pixel", [0, 0, 0, 0]))
    cell_size_px = int(room_info.get("cell_size_px", 40))
    lamp_grid_positions = [[int(row), int(col)] for row, col in (state.placements.get("lamps", []) or [])]
    switch_grid_positions = [[int(row), int(col)] for row, col in (state.placements.get("switches", []) or [])]
    lamp_type = _lamp_type_from_result(result, state)
    lamp_cad_positions = _grid_positions_to_cad_positions(
        lamp_grid_positions,
        bbox_pixel,
        cell_size_px,
        cad_params,
        image_width,
        image_height,
    )
    switch_cad_positions = _grid_positions_to_cad_positions(
        switch_grid_positions,
        bbox_pixel,
        cell_size_px,
        cad_params,
        image_width,
        image_height,
    )

    grid_dump = {
        room_name: {
            "room_name": room_name,
            "lamp_type": lamp_type,
            "grid_rows": int(np.array(room_info.get("matrix", []), dtype=np.int32).shape[0]),
            "grid_cols": int(np.array(room_info.get("matrix", []), dtype=np.int32).shape[1]),
            "cell_size_px": cell_size_px,
            "bbox_pixel": bbox_pixel,
            "room_area_px": float(room_info.get("room_area_px", 0.0)),
            "room_area_mm2": float(room_info.get("room_area_mm2", 0.0)),
            "room_area_m2": float(room_info.get("room_area_m2", 0.0)),
            "estimated_lamp_count": int((result.get("lamp_plan", {}) or {}).get("lamp_count", len(lamp_grid_positions))),
            "is_regular": bool(room_info.get("is_regular", False)),
            "assigned_door": room_info.get("assigned_door"),
            "door_side": room_info.get("door_side"),
            "door_edge_cells": room_info.get("door_edge_cells", []),
            "lamp_grid_positions": lamp_grid_positions,
            "switch_grid_positions": switch_grid_positions,
            "matrix": deepcopy(room_info.get("matrix", [])),
        }
    }

    compact_switch = None
    if switch_cad_positions:
        compact_switch = {
            "switch_type": "开关",
            "cad_position": switch_cad_positions[0],
        }

    lighting_rooms = {
        room_name: {
            "room_name": room_name,
            "lamp_count": int(len(lamp_grid_positions)),
            "room_area_m2": float(room_info.get("room_area_m2", 0.0)),
            "switch": compact_switch,
            "switch_count": int(len(switch_grid_positions)),
            "lamps": {
                "lamp_type": lamp_type,
                "count": int(len(lamp_cad_positions)),
                "cad_positions": lamp_cad_positions,
            },
        }
    }

    switches_internal = []
    for idx, pos in enumerate(switch_grid_positions):
        cad_position = switch_cad_positions[idx] if idx < len(switch_cad_positions) else []
        switches_internal.append(
            {
                "switch_type": "开关",
                "grid_position": pos,
                "cad_position": cad_position,
            }
        )

    lighting_internal = {
        room_name: {
            "room_name": room_name,
            "grid_rows": int(np.array(room_info.get("matrix", []), dtype=np.int32).shape[0]),
            "grid_cols": int(np.array(room_info.get("matrix", []), dtype=np.int32).shape[1]),
            "cell_size_px": cell_size_px,
            "bbox_pixel": bbox_pixel,
            "matrix": deepcopy(room_info.get("matrix", [])),
            "room_area_px": float(room_info.get("room_area_px", 0.0)),
            "room_area_mm2": float(room_info.get("room_area_mm2", 0.0)),
            "room_area_m2": float(room_info.get("room_area_m2", 0.0)),
            "is_regular": bool(room_info.get("is_regular", False)),
            "door_side": room_info.get("door_side"),
            "door_edge_cells": room_info.get("door_edge_cells", []),
            "switch": switches_internal[0] if switches_internal else None,
            "switch_count": len(switch_grid_positions),
            "switches": switches_internal,
            "lamps": {
                "lamp_type": lamp_type,
                "count": len(lamp_grid_positions),
                "grid_positions": lamp_grid_positions,
                "cad_positions": lamp_cad_positions,
            },
        }
    }
    return grid_dump, lighting_rooms, lighting_internal


def _save_turn_outputs(
    turn_dir: Path,
    image_path: Path,
    room_name: str,
    room_info: Dict[str, Any],
    state: RoomAgentState,
    result: Dict[str, Any],
    cad_params: Dict[str, float],
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    turn_dir.mkdir(parents=True, exist_ok=True)
    grid_dump, lighting_rooms, lighting_internal = _build_room_visual_payloads(
        room_name=room_name,
        room_info=room_info,
        state=state,
        result=result,
        cad_params=cad_params,
        image_width=image_width,
        image_height=image_height,
    )

    step7_grid_file = turn_dir / "step7_room_grid_matrices.json"
    with step7_grid_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "image_width": int(image_width),
                "image_height": int(image_height),
                "cell_size_px": int(room_info.get("cell_size_px", 40)),
                "rooms": grid_dump,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    step7_layout_file = turn_dir / "step7_lighting_layout.json"
    with step7_layout_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "image_width": int(image_width),
                "image_height": int(image_height),
                "rooms": lighting_rooms,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    vis_alpha = float(os.getenv("CAD_GRID_VIS_ALPHA", "0.35"))
    step7_vis_dir = _save_grid_visualization_overlay(
        image_path=str(image_path),
        grid_dump=grid_dump,
        output_dir=str(turn_dir),
        alpha=vis_alpha,
    )

    lighting_payload = {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "rooms_internal": lighting_internal,
    }
    prev_output_dir = os.environ.get("CAD_STEP_OUTPUT_DIR")
    os.environ["CAD_STEP_OUTPUT_DIR"] = str(turn_dir)
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

    return {
        "step7_grid_file": str(step7_grid_file),
        "step7_layout_file": str(step7_layout_file),
        "step7_vis_dir": step7_vis_dir,
        "step8_file": str(turn_dir / "step8_wiring_layout.json"),
        "step8_vis_dir": str(turn_dir / "step8_room_wiring_visualization"),
        "wiring_payload": wiring_payload,
        "lighting_payload": {
            "image_width": int(image_width),
            "image_height": int(image_height),
            "rooms": lighting_rooms,
        },
    }


def _save_chat_history(session_dir: Path, history: List[Dict[str, Any]]) -> None:
    history_file = session_dir / "chat_history.json"
    with history_file.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _print_room_list(room_names: List[str]) -> None:
    print("可用房间:")
    for idx, room_name in enumerate(room_names, 1):
        print(f"{idx}. {room_name}")
    print("")


def main() -> None:
    default_image = REPO_ROOT / "test_files" / "agent_test.png"
    image_path = Path(os.getenv("CAD_AGENT_CHAT_IMAGE", str(default_image))).expanduser()
    agent_type = os.getenv("CAD_AGENT_TYPE", "react").strip().lower()
    provider = os.getenv("CAD_AGENT_PROVIDER", "qwen").strip().lower()
    model_name = os.getenv("CAD_AGENT_MODEL", "").strip() or None
    max_steps = int(os.getenv("CAD_AGENT_MAX_STEPS", "8"))
    session_dir = _create_timestamped_output_dir(REPO_ROOT / "chat_output")
    preprocess_dir = session_dir / "preprocess"
    logs_dir = session_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[chat] session_dir: {session_dir}")
    initial_input = process_image_to_agent_input(
        image_path=image_path,
        cad_params=dict(DEFAULT_CAD_PARAMS),
        output_dir=preprocess_dir,
        save_to_file=True,
    )

    room_names = sorted(list((initial_input.get("rooms", {}) or {}).keys()))
    if not room_names:
        raise RuntimeError("预处理完成，但没有检测到任何房间。")

    agent = _build_agent(
        agent_type=agent_type,
        provider=provider,
        model_name=model_name,
        log_dir=logs_dir,
    )
    states = _build_room_states(initial_input)
    last_room_name: Optional[str] = None
    history: List[Dict[str, Any]] = []

    session_meta = {
        "session_dir": str(session_dir),
        "image_path": str(image_path),
        "agent_type": agent_type,
        "provider": provider,
        "model_name": model_name,
        "room_names": room_names,
    }
    with (session_dir / "session_meta.json").open("w", encoding="utf-8") as f:
        json.dump(session_meta, f, ensure_ascii=False, indent=2)

    print(f"[chat] agent_type={agent_type} provider={provider} model={model_name or 'default'}")
    _print_room_list(room_names)
    print("输入房间指令开始设计，例如：帮我绘制除尘室，使用防爆灯，放6个灯具。")
    print("输入 `rooms` 查看房间列表，输入 `exit` 退出。\n")

    turn_index = 0
    while True:
        try:
            user_text = input("user> ").strip()
        except EOFError:
            print("")
            break

        if not user_text:
            continue
        lowered = user_text.lower()
        if lowered in {"exit", "quit"}:
            break
        if lowered in {"rooms", "list", "ls"}:
            _print_room_list(room_names)
            continue

        room_name = _match_room_name(user_text, room_names, last_room_name)
        if room_name is None:
            print("未识别到目标房间，请在指令中明确写出房间名。\n")
            _print_room_list(room_names)
            continue
        if room_name not in states:
            print(f"房间 `{room_name}` 当前不可用。\n")
            continue

        room_info = (initial_input.get("rooms", {}) or {}).get(room_name, {})
        state = states[room_name]
        reset_layout = _should_reset_layout(user_text, state)
        turn_index += 1
        turn_dir = session_dir / f"turn_{turn_index:02d}_{_sanitize_filename(room_name)}"
        turn_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[chat] turn={turn_index} room={room_name} reset_layout={reset_layout} "
            f"output_dir={turn_dir}"
        )

        result = agent.run_for_room(
            state=state,
            max_steps=max_steps,
            user_goal=user_text,
            reset_layout=reset_layout,
        )
        turn_outputs = _save_turn_outputs(
            turn_dir=turn_dir,
            image_path=image_path,
            room_name=room_name,
            room_info=room_info,
            state=state,
            result=result,
            cad_params=dict(DEFAULT_CAD_PARAMS),
            image_width=int(initial_input.get("image_width", 0)),
            image_height=int(initial_input.get("image_height", 0)),
        )

        turn_record = {
            "turn_index": turn_index,
            "room_name": room_name,
            "user_text": user_text,
            "reset_layout": reset_layout,
            "result": {
                "selected_lamp_type": _lamp_type_from_result(result, state),
                "lamp_count": len(state.placements.get("lamps", [])),
                "switch_count": len(state.placements.get("switches", [])),
                "finish_reason": result.get("finish_reason"),
                "validation": result.get("validation"),
                "strategy_summary": result.get("strategy_summary"),
                "log_file": result.get("log_file"),
            },
            "turn_dir": str(turn_dir),
            "step_outputs": turn_outputs,
        }
        history.append(turn_record)
        _save_chat_history(session_dir, history)
        with (turn_dir / "turn_result.json").open("w", encoding="utf-8") as f:
            json.dump(turn_record, f, ensure_ascii=False, indent=2)

        print(
            f"assistant> 房间={room_name} 灯具类型={turn_record['result']['selected_lamp_type']} "
            f"灯具={turn_record['result']['lamp_count']} 开关={turn_record['result']['switch_count']} "
            f"finish={turn_record['result']['finish_reason']}"
        )
        print(f"assistant> 可视化输出已保存到: {turn_dir}\n")
        last_room_name = room_name

    print(f"[chat] session finished. outputs saved in: {session_dir}")


if __name__ == "__main__":
    main()
