"""
RL vs GA 对比评估脚本

指标：
  - 均匀度：1 - 当前势能 / 初始势能，归一化到 [0,1]，越高越好
  - 对齐度：灯具行列软对齐分数 [0,1]，越高越好
  - 时间（ms）：单房间推理/优化耗时

用法：
  cd /home/chen/punchy/CAD_knowledge_service/RL
  ../.venv/bin/python eval_compare.py \\
    --room_dir ../RL/room_gen/all/json \\
    --split_dir ../RL/room_gen/all/split \\
    --model_path ../RL/output_multiroom/<timestamp>/ppo_multi_room_best.pt \\
    --config ../RL/config.yaml \\
    --output_dir ../RL/eval_results \\
    [--save_images]
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from env import EnvironmentConfig, SingleRoomLightingEnv
from ga_layout import GAConfig, GALayoutOptimizer
from model import LightingActorCritic
from reward import RewardConfig
from train_multi import load_room_dataset, load_split, split_by_shape_stratified
from visualize import save_room_grid_image


# ── helpers ───────────────────────────────────────────────────────────────────

def load_yaml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _mean(lst: list[float]) -> float:
    valid = [x for x in lst if x == x]  # filter NaN
    return float(np.mean(valid)) if valid else float("nan")


def _compute_uniformity(env: SingleRoomLightingEnv, state) -> tuple[float, float]:
    """Return (uniformity_normalized, potential_raw)."""
    potential_raw = float(env.reward_calculator.potential(state))
    initial_potential = float(env.reward_calculator.initial_potential(state))
    if initial_potential <= 0.0:
        return 1.0, potential_raw
    uniformity = float(np.clip(1.0 - potential_raw / initial_potential, 0.0, 1.0))
    return uniformity, potential_raw


# ── RL inference ──────────────────────────────────────────────────────────────

def run_rl(
    room_data: dict,
    model: LightingActorCritic,
    base_env_cfg: EnvironmentConfig,
    device: torch.device,
    images_dir: Path | None,
    room_idx: int,
) -> dict:
    lamp_count = room_data["lamp_count"]
    cfg = copy.deepcopy(base_env_cfg)
    cfg.target_lamp_count = lamp_count
    cfg.reward_config = copy.deepcopy(base_env_cfg.reward_config)
    cfg.reward_config.target_lamp_count = lamp_count

    env = SingleRoomLightingEnv(room_data, config=cfg)
    obs = env.reset()
    done = False

    t0 = time.perf_counter()
    with torch.no_grad():
        while not done:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
            action = int(model.act(obs_t, deterministic=True)["action"].item())
            obs, _, done, _ = env.step(action)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    final_state = env.current_room_state()
    uniformity, potential_raw = _compute_uniformity(env, final_state)

    bd = env.last_breakdown
    alignment = float(bd.alignment_normalized) if bd else 0.0

    result = {
        "uniformity": uniformity,
        "potential_raw": potential_raw,
        "alignment": alignment,
        "time_ms": elapsed_ms,
    }

    if images_dir is not None:
        room_name = room_data.get("room_name", f"room_{room_idx:04d}")
        title = f"RL {room_name} | u={uniformity:.2f} a={alignment:.2f}"
        save_room_grid_image(
            env.current_encoded_matrix(),
            images_dir / f"rl_{room_idx:04d}_{lamp_count}lamp.png",
            cell_size=16,
            room_name=title,
        )

    return result


# ── GA inference ──────────────────────────────────────────────────────────────

def run_ga(
    room_data: dict,
    ga_cfg: GAConfig,
    base_env_cfg: EnvironmentConfig,
    images_dir: Path | None,
    room_idx: int,
    tmp_dir: Path,
) -> dict:
    lamp_count = room_data["lamp_count"]
    cfg = copy.deepcopy(base_env_cfg)
    cfg.target_lamp_count = lamp_count
    cfg.reward_config = copy.deepcopy(base_env_cfg.reward_config)
    cfg.reward_config.target_lamp_count = lamp_count

    env = SingleRoomLightingEnv(room_data, config=cfg)

    t0 = time.perf_counter()
    optimizer = GALayoutOptimizer(env=env, ga_config=ga_cfg, output_dir=tmp_dir)
    summary = optimizer.run()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    best_genome = tuple(summary["best_genome"])
    best_state = optimizer._build_state(best_genome)
    uniformity, potential_raw = _compute_uniformity(env, best_state)

    diag = summary["best_diagnostics"]
    alignment = float(diag["alignment_score"])

    result = {
        "uniformity": uniformity,
        "potential_raw": potential_raw,
        "alignment": alignment,
        "time_ms": elapsed_ms,
    }

    if images_dir is not None:
        room_name = room_data.get("room_name", f"room_{room_idx:04d}")
        best_positions = summary["best_positions"]
        encoded = env.original_matrix.copy()
        for r, c in best_positions:
            encoded[r, c] = 4
        title = f"GA {room_name} | u={uniformity:.2f} a={alignment:.2f}"
        save_room_grid_image(
            encoded,
            images_dir / f"ga_{room_idx:04d}_{lamp_count}lamp.png",
            cell_size=16,
            room_name=title,
        )

    return result


# ── reporting ─────────────────────────────────────────────────────────────────

METRICS = [
    ("uniformity", "均匀度（↑）",  True),   # (key, label, higher_is_better)
    ("alignment",  "对齐度（↑）",  True),
    ("time_ms",    "时间 ms（↓）", False),
]


def _build_table(
    rl_records: list[dict],
    ga_records: list[dict],
) -> tuple[str, str]:
    """Return (terminal_str, markdown_str) for the comparison table."""
    lamp_counts = sorted({r["lamp_count"] for r in rl_records})

    # ── terminal ──────────────────────────────────────────────────────────────
    col_w = 14
    header = f"{'lamp':>5}  {'指标':<16}  {'RL':>{col_w}}  {'GA':>{col_w}}  {'差值(RL-GA)':>{col_w}}"
    sep = "=" * len(header)
    thin = "-" * len(header)
    lines = ["\n" + sep, header, sep]

    # ── markdown ──────────────────────────────────────────────────────────────
    md_lines: list[str] = []

    def _add_group(label: str, rl_sub: list[dict], ga_sub: list[dict]) -> None:
        if label != "ALL":
            md_lines.append(f"\n### {label}灯\n")
        else:
            md_lines.append("\n### 总体\n")
        md_lines.append("| 指标 | RL | GA | 差值(RL-GA) |")
        md_lines.append("|------|---:|---:|---:|")

        for key, metric_label, higher_better in METRICS:
            rl_val = _mean([r[key] for r in rl_sub])
            ga_val = _mean([r[key] for r in ga_sub])
            diff = rl_val - ga_val
            sign = "+" if diff >= 0 else ""

            # terminal
            lines.append(
                f"{label:>5}  {metric_label:<16}  {rl_val:{col_w}.2f}  "
                f"{ga_val:{col_w}.2f}  {sign}{diff:{col_w}.2f}"
            )
            # markdown
            better = "↑ RL更好" if (higher_better and diff > 0) or (not higher_better and diff < 0) else ("↓ GA更好" if diff != 0 else "持平")
            md_lines.append(f"| {metric_label} | {rl_val:.2f} | {ga_val:.2f} | {sign}{diff:.2f} ({better}) |")

        if label != "ALL":
            lines.append(thin)

    for lamps in lamp_counts:
        rl_sub = [r for r in rl_records if r["lamp_count"] == lamps]
        ga_sub = [r for r in ga_records if r["lamp_count"] == lamps]
        _add_group(str(lamps), rl_sub, ga_sub)

    _add_group("ALL", rl_records, ga_records)

    lines.append(sep)
    lines.append("（均匀度、对齐度越高越好；时间越低越好；差值 = RL - GA）")

    return "\n".join(lines), "\n".join(md_lines)


def write_markdown(
    md_path: Path,
    test_rooms: list[dict],
    rl_records: list[dict],
    ga_records: list[dict],
    md_table: str,
    model_path: str,
) -> None:
    reg = sum(1 for r in test_rooms if r.get("_shape_type") == "regular")
    irr = len(test_rooms) - reg

    lines = [
        "# RL vs GA 布局对比评估",
        "",
        "## 测试集概况",
        f"- 测试房间数：{len(test_rooms)}",
        f"- 规则房间：{reg}，不规则房间：{irr}",
        f"- 模型：`{model_path}`",
        "",
        "## 指标说明",
        "- **均匀度**：按 `1 - 当前势能 / 初始势能` 归一化到 `[0,1]`，越高越好",
        "- **对齐度**：灯具行列软对齐分数，越高越好（[0,1]）",
        "- **时间（ms）**：单房间推理/优化平均耗时",
        "",
        "## 按灯具数分组对比",
        md_table,
        "",
        "---",
        "*由 `RL/eval_compare.py` 自动生成*",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[compare] markdown saved to {md_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RL vs GA on test rooms")
    parser.add_argument("--room_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="RL/config.yaml")
    parser.add_argument("--output_dir", type=str, default="RL/eval_results")
    parser.add_argument("--split_dir", type=str, default=None,
                        help="Pre-split index dir (train.json/val.json/test.json)")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    config_payload = load_yaml_config(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_ga_dir = output_dir / "_ga_tmp"
    tmp_ga_dir.mkdir(exist_ok=True)

    images_dir: Path | None = None
    if args.save_images:
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

    # ── load test set ─────────────────────────────────────────────────────────
    if args.split_dir:
        _, _, test_rooms = load_split(Path(args.split_dir), Path(args.room_dir))
    else:
        all_rooms = load_room_dataset(Path(args.room_dir))
        _, _, test_rooms = split_by_shape_stratified(
            all_rooms, train_ratio=0.64, val_ratio=0.16, test_ratio=0.20, seed=args.seed,
        )
    print(f"[compare] test set: {len(test_rooms)} rooms")

    # ── load RL model ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightingActorCritic(target_lamp_count=None)
    ckpt = torch.load(args.model_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[compare] loaded model from {args.model_path}")

    # ── build base configs ────────────────────────────────────────────────────
    reward_raw = config_payload.get("reward", {})
    base_reward_cfg = RewardConfig(**reward_raw)
    env_raw = config_payload.get("environment", {})
    base_env_cfg = EnvironmentConfig(
        padded_size=env_raw.get("padded_size", 48),
        max_steps=env_raw.get("max_steps", 16),
        target_lamp_count=2,
        turn_penalty=env_raw.get("turn_penalty", 0.2),
        wall_margin=env_raw.get("wall_margin", 1),
        reward_config=base_reward_cfg,
    )
    ga_cfg = GAConfig(**config_payload.get("ga", {}))
    ga_cfg.generations = 200

    # ── evaluate ──────────────────────────────────────────────────────────────
    rl_records: list[dict] = []
    ga_records: list[dict] = []

    for i, room_data in enumerate(test_rooms):
        lamp_count = room_data["lamp_count"]
        room_name = room_data.get("room_name", f"room_{i:04d}")
        print(f"[{i+1:3d}/{len(test_rooms)}] {room_name}  lamps={lamp_count}", end="  ", flush=True)

        rl_res = run_rl(room_data, model, base_env_cfg, device, images_dir, i)
        ga_res = run_ga(room_data, ga_cfg, base_env_cfg, images_dir, i, tmp_ga_dir)

        print(
            f"RL u={rl_res['uniformity']:.3f} a={rl_res['alignment']:.2f} ({rl_res['time_ms']:.0f}ms)  |  "
            f"GA u={ga_res['uniformity']:.3f} a={ga_res['alignment']:.2f} ({ga_res['time_ms']:.0f}ms)"
        )

        rl_records.append({"lamp_count": lamp_count, "room_name": room_name, **rl_res})
        ga_records.append({"lamp_count": lamp_count, "room_name": room_name, **ga_res})

    # ── print & write results ─────────────────────────────────────────────────
    terminal_table, md_table = _build_table(rl_records, ga_records)
    print(terminal_table)

    # repo root is two levels up from RL/
    repo_root = Path(__file__).resolve().parents[1]
    md_path = repo_root / "summary_docs" / "RL-GA-layout.md"
    write_markdown(md_path, test_rooms, rl_records, ga_records, md_table, args.model_path)

    # save raw JSON
    out_path = output_dir / "compare_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"rl": rl_records, "ga": ga_records}, f, ensure_ascii=False, indent=2)
    print(f"[compare] json saved to {out_path}")

    shutil.rmtree(tmp_ga_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
