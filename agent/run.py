from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from preprocess.coordinate_converter import DEFAULT_CAD_PARAMS
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
    """
    在给定输出根目录下创建一次运行专属子目录，避免覆盖历史结果。
    子目录命名: YYYYMMDD_HHMMSS(_N)
    """
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


def run_agent_design(
    initial_input: Dict[str, Any],
    image_path: Path,
    cad_params: Dict[str, float],
    output_dir: Path,
    agent_type: str = "react",
    provider: str = "qwen",
    model_name: Optional[str] = None,
    max_steps: int = 8,
) -> Dict[str, Any]:
    """
    执行 Agent 布局并输出最终 step6 结果。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rooms = initial_input.get("rooms", {}) or {}
    image_w = int(initial_input.get("image_width", 0))
    image_h = int(initial_input.get("image_height", 0))

    resolved_agent_type = (agent_type or "react").strip().lower()
    if resolved_agent_type == "minimax":
        agent = MiniMaxLightingAgent(
            model_name=model_name,
            log_dir=str(REPO_ROOT / "logs"),
        )
    elif resolved_agent_type == "react":
        agent = ReActLightingAgent(
            provider=provider,
            model_name=model_name,
            log_dir=str(REPO_ROOT / "logs"),
        )
    else:
        raise ValueError(f"unsupported agent_type: {agent_type}. Use 'react' or 'minimax'.")

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
        for item in (result.get("placements", {}) or {}).get("lamps", []):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                lamps_grid.append([int(item[0]), int(item[1])])

        switches_grid = []
        for item in (result.get("placements", {}) or {}).get("switches", []):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                switches_grid.append([int(item[0]), int(item[1])])

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
        "agent_type": resolved_agent_type,
        "image_path": str(image_path),
        "image_width": image_w,
        "image_height": image_h,
        "rooms": room_plans,
        "wiring_rooms": wiring_payload.get("rooms", {}),
    }
    final_json = output_dir / "step6_agent_layout.json"
    with final_json.open("w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)
    print(f"[agent-run] saved final step6: {final_json}")
    print(f"[agent-run] saved final step6 visualization: {step6_vis_dir}")

    return final_payload


def run_agent_pipeline(
    image_path: Path,
    output_dir: Optional[Path] = None,
    agent_type: str = "react",
    provider: str = "qwen",
    model_name: Optional[str] = None,
    max_steps: int = 8,
) -> Dict[str, Any]:
    if cad_params is None:
        cad_params = dict(DEFAULT_CAD_PARAMS)
    if output_dir is None:
        output_dir = REPO_ROOT / "agent_output"

    run_output_dir = _create_timestamped_output_dir(Path(output_dir))
    print(f"[agent-run] run_output_dir: {run_output_dir}")

    initial_input = process_image_to_agent_input(
        image_path=image_path,
        cad_params=cad_params,
        output_dir=run_output_dir,
        save_to_file=True,
    )
    final_payload = run_agent_design(
        initial_input=initial_input,
        image_path=image_path,
        output_dir=run_output_dir,
        agent_type=agent_type,
        provider=provider,
        model_name=model_name,
        max_steps=max_steps,
    )
    return {
        "output_dir": str(run_output_dir),
        "initial_input": initial_input,
        "final_result": final_payload,
    }


def main() -> None:
    default_image = REPO_ROOT / "test_files" / "agent_test.png"
    agent_type = os.getenv("CAD_AGENT_TYPE", "react").strip().lower()
    provider = os.getenv("CAD_AGENT_PROVIDER", "qwen").strip().lower()
    model_name = os.getenv("CAD_AGENT_MODEL", "").strip() or None
    max_steps = int(os.getenv("CAD_AGENT_MAX_STEPS", "8"))

    result = run_agent_pipeline(
        image_path=default_image,
        output_dir=REPO_ROOT / "agent_output",
        agent_type=agent_type,
        provider=provider,
        model_name=model_name,
        max_steps=max_steps,
    )
    print(
        f"[agent-run] done. agent_type={agent_type}, rooms={len((result.get('initial_input', {}) or {}).get('rooms', {}))}, "
        f"final_rooms={len((result.get('final_result', {}) or {}).get('rooms', {}))}, "
        f"output_dir={result.get('output_dir')}"
    )


if __name__ == "__main__":
    main()
