"""
对 article_visu 下的三个房间 JSON 分别用 RL 和 MST+A* 完成布线，
将可视化结果保存到 RL/article_visu/output/{rl,mst}/ 子目录。

用法：
  cd /home/chen/punchy/CAD_knowledge_service
  .venv/bin/python RL/wiring_article_visu.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.wiring_env import WiringEnv, WiringEnvConfig, compute_mst_baseline
from RL.wiring_model import WiringPolicyNet, build_lamp_coords_tensor

# ------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------

ARTICLE_VISU_DIR = Path(__file__).resolve().parent / "article_visu"
MODEL_DIR        = Path(__file__).resolve().parent / "output_wiring" / "20260412_182713"
OUTPUT_DIR       = ARTICLE_VISU_DIR / "output"
RL_OUT_DIR       = OUTPUT_DIR / "rl"
MST_OUT_DIR      = OUTPUT_DIR / "mst"

CELL_SIZE = 40
PADDING   = 12

# ------------------------------------------------------------------
# 可视化核心：将路径列表画在房间底图上
# ------------------------------------------------------------------

def _draw_wiring(
    matrix: np.ndarray,
    paths: list[list[tuple[int, int]]],
    entry_points: list[tuple[int, int]],
    total_cost: float,
    method_label: str,
    room_name: str,
) -> np.ndarray:
    from RL.visualize import render_room_grid, _load_font
    from PIL import Image, ImageDraw

    img = render_room_grid(matrix, cell_size=CELL_SIZE, room_name=None)

    x0, y0 = PADDING, PADDING
    color_wire  = (0, 200, 200)   # 青色线路
    color_entry = (0, 120, 255)   # 橙色接入点

    for path in paths:
        pts = [
            (x0 + c * CELL_SIZE + CELL_SIZE // 2,
             y0 + r * CELL_SIZE + CELL_SIZE // 2)
            for r, c in path
        ]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], color_wire, 6, cv2.LINE_AA)


    rows, cols = matrix.shape
    text_y = y0 + rows * CELL_SIZE + 6
    stats = f"{method_label}  room={room_name}  cost={total_cost:.1f}  paths={len(paths)}"

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _load_font(18)
    draw.text((x0, text_y), stats, fill=(30, 30, 30), font=font)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------------
# RL 布线
# ------------------------------------------------------------------

def run_rl(
    room_data: dict[str, Any],
    model: WiringPolicyNet,
    env_cfg: WiringEnvConfig,
    device: torch.device,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]], float]:
    env = WiringEnv(room_data, config=env_cfg)
    obs = env.reset()
    lamp_coords_t = build_lamp_coords_tensor(
        env.lamp_positions, env.row_offset, env.col_offset, device
    )

    paths: list[list[tuple[int, int]]] = []
    entry_points: list[tuple[int, int]] = []
    total_cost = 0.0
    done = False

    while not done:
        obs_t  = torch.from_numpy(obs).unsqueeze(0).to(device, dtype=torch.float32)
        mask_t = torch.from_numpy(env.action_mask()).unsqueeze(0).to(device)
        out    = model.act(obs_t, lamp_coords_t, mask_t, deterministic=True)
        obs, _, done, info = env.step(int(out["action"].item()))
        if env.step_paths:
            paths.append(env.step_paths[-1])
            entry_points.append(info.get("entry_point", env.switch_pos))
        total_cost += info.get("route_cost", 0.0)

    return paths, entry_points, total_cost


# ------------------------------------------------------------------
# MST+A* 布线
# ------------------------------------------------------------------

def run_mst(
    room_data: dict[str, Any],
    turn_penalty: float,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]], float]:
    result = compute_mst_baseline(room_data, turn_penalty=turn_penalty)
    routes  = result["routes"]          # list[list[GridPoint]]
    total_cost = result["total_cost"]

    # MST 返回的路径已经是从父节点到子节点的方向，接入点取每条路径的末端
    entry_points = [path[-1] for path in routes if path]

    return routes, entry_points, total_cost


# ------------------------------------------------------------------
# 保存布线结果到 JSON
# ------------------------------------------------------------------

def save_wiring_result(
    room_data: dict[str, Any],
    paths: list[list[tuple[int, int]]],
    total_cost: float,
    method: str,
    out_path: Path,
) -> None:
    result = {
        "room_name": room_data.get("room_name", "unknown"),
        "method": method,
        "total_cost": total_cost,
        "n_paths": len(paths),
        "paths": [[[r, c] for r, c in p] for p in paths],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def main() -> None:
    # 输出目录
    RL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MST_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载环境配置
    import yaml
    config_path = MODEL_DIR / "config_wiring.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg_payload = yaml.safe_load(f) or {}
    env_raw = cfg_payload.get("wiring_environment", {})
    env_cfg = WiringEnvConfig(
        padded_size=env_raw.get("padded_size", 48),
        turn_penalty=env_raw.get("turn_penalty", 0.2),
        step_cost_coef=env_raw.get("step_cost_coef", 0.1),
        invalid_action_penalty=env_raw.get("invalid_action_penalty", 1.0),
        terminal_length_coef=env_raw.get("terminal_length_coef", 1.0),
        terminal_sharing_coef=env_raw.get("terminal_sharing_coef", 0.5),
        terminal_depth_coef=env_raw.get("terminal_depth_coef", 0.3),
    )

    # 加载 RL 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WiringPolicyNet(in_channels=6)
    ckpt = torch.load(MODEL_DIR / "wiring_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"[info] loaded model from {MODEL_DIR / 'wiring_best.pt'}, device={device}")

    # 遍历三个房间（排除上次生成的 wiring_result JSON）
    json_files = sorted(f for f in ARTICLE_VISU_DIR.glob("*.json")
                        if "_wiring_" not in f.name)
    print(f"[info] found {len(json_files)} rooms: {[f.name for f in json_files]}")

    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        room_data = payload if "matrix" in payload else next(iter(payload.values()))
        matrix    = np.asarray(room_data["matrix"], dtype=np.int32)
        room_name = room_data.get("room_name", json_path.stem)
        stem      = json_path.stem
        n_lamps   = int((matrix == 4).sum())
        print(f"\n[room] {stem}  shape={matrix.shape}  lamps={n_lamps}")

        # ── RL ──
        with torch.no_grad():
            rl_paths, rl_entries, rl_cost = run_rl(room_data, model, env_cfg, device)
        print(f"  RL  cost={rl_cost:.1f}  paths={len(rl_paths)}")

        rl_img = _draw_wiring(matrix, rl_paths, rl_entries, rl_cost, "RL", stem)
        rl_vis_path = RL_OUT_DIR / f"{stem}.png"
        cv2.imwrite(str(rl_vis_path), rl_img)

        save_wiring_result(
            room_data, rl_paths, rl_cost, "rl",
            ARTICLE_VISU_DIR / f"{stem}_wiring_rl.json",
        )
        print(f"  RL  vis -> {rl_vis_path}")

        # ── MST+A* ──
        mst_paths, mst_entries, mst_cost = run_mst(room_data, env_cfg.turn_penalty)
        print(f"  MST cost={mst_cost:.1f}  paths={len(mst_paths)}")

        mst_img = _draw_wiring(matrix, mst_paths, mst_entries, mst_cost, "MST+A*", stem)
        mst_vis_path = MST_OUT_DIR / f"{stem}.png"
        cv2.imwrite(str(mst_vis_path), mst_img)

        save_wiring_result(
            room_data, mst_paths, mst_cost, "mst",
            ARTICLE_VISU_DIR / f"{stem}_wiring_mst.json",
        )
        print(f"  MST vis -> {mst_vis_path}")

        # 对比
        ratio = rl_cost / max(mst_cost, 1.0)
        print(f"  RL/MST ratio={ratio:.3f}  ({'RL better' if ratio < 1 else 'MST better'})")

    print(f"\n[done] vis saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
