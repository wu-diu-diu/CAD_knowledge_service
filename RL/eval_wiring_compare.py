"""
RL 布线模型 vs MST+A* 基线对比评估

指标：
  - total_cost    总布线代价（A* cost 之和，含转弯惩罚），越低越好
  - length_score  线长得分 [0,1]，越高越好
  - sharing_score 共享线段比例 [0,1]，越高越好
  - max_depth     最长单条路径长度，越低越好
  - time_ms       单房间耗时，越低越好
  - cost_ratio    rl_cost / mst_cost，<1.0 表示 RL 更好

用法：
  cd /home/chen/punchy/CAD_knowledge_service/RL
  ../.venv/bin/python eval_wiring_compare.py \\
    --room_dir test_room/layout_room/json \\
    --split_dir test_room/layout_room/split \\
    --model_path output_wiring/<timestamp>/best_model.pt \\
    --output_dir eval_wiring_results
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_wiring import load_room_dataset, load_wiring_split, split_train_test
from wiring_env import (
    WiringEnv,
    WiringEnvConfig,
    _compute_max_depth,
    _compute_sharing_score,
    compute_mst_baseline,
)
from wiring_model import WiringPolicyNet, build_lamp_coords_tensor


# ── helpers ───────────────────────────────────────────────────────────────────

def _mean(lst: list[float]) -> float:
    valid = [x for x in lst if x == x]
    return float(np.mean(valid)) if valid else float("nan")


def _length_score(total_cost: float, routes: list, switch_pos: tuple, lamp_positions: list) -> float:
    """与 WiringEnv._terminal_reward 相同的 length_score 计算。"""
    max_possible = sum(
        abs(pos[0] - switch_pos[0]) + abs(pos[1] - switch_pos[1])
        for pos in lamp_positions
    ) * 2.0
    max_possible = max(max_possible, 1.0)
    return 1.0 - min(total_cost / max_possible, 1.0)


# ── RL inference ──────────────────────────────────────────────────────────────

def run_rl_wiring(
    room_data: dict[str, Any],
    model: WiringPolicyNet,
    env_cfg: WiringEnvConfig,
    device: torch.device,
) -> dict[str, float]:
    env = WiringEnv(room_data, config=env_cfg)
    obs = env.reset()
    done = False
    last_info: dict[str, Any] = {}

    t0 = time.perf_counter()
    with torch.no_grad():
        while not done:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
            lamp_coords = build_lamp_coords_tensor(
                env.lamp_positions, env.row_offset, env.col_offset, device
            )
            action_mask = torch.tensor(
                [not env.connected[i] for i in range(env.n_lamps)],
                dtype=torch.bool, device=device,
            ).unsqueeze(0)
            out = model.act(obs_t, lamp_coords, action_mask, deterministic=True)
            action = int(out["action"].item())
            obs, _, done, last_info = env.step(action)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "total_cost":    float(last_info.get("total_cost", 0.0)),
        "length_score":  float(last_info.get("length_score", 0.0)),
        "sharing_score": float(last_info.get("sharing_score", 0.0)),
        "max_depth":     float(last_info.get("max_depth", 0)),
        "time_ms":       elapsed_ms,
    }


# ── MST+A* baseline ───────────────────────────────────────────────────────────

def run_mst_baseline(
    room_data: dict[str, Any],
    env_cfg: WiringEnvConfig,
) -> dict[str, float]:
    import numpy as np

    t0 = time.perf_counter()
    result = compute_mst_baseline(room_data, turn_penalty=env_cfg.turn_penalty)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    total_cost = float(result.get("total_cost", 0.0))
    routes = result.get("routes", [])

    # 解析 switch_pos 和 lamp_positions（与 compute_mst_baseline 相同逻辑）
    matrix = np.asarray(room_data["matrix"], dtype=np.int32)
    switch_coords = np.argwhere(matrix == 3)
    lamp_coords_arr = np.argwhere(matrix == 4)
    switch_pos = tuple(map(int, switch_coords[0])) if len(switch_coords) > 0 else (0, 0)
    lamp_positions = [tuple(map(int, c)) for c in lamp_coords_arr]

    ls = _length_score(total_cost, routes, switch_pos, lamp_positions)
    ss = _compute_sharing_score(routes)
    md = float(_compute_max_depth(routes, switch_pos, lamp_positions))

    return {
        "total_cost":    total_cost,
        "length_score":  ls,
        "sharing_score": ss,
        "max_depth":     md,
        "time_ms":       elapsed_ms,
    }


# ── reporting ─────────────────────────────────────────────────────────────────

METRICS = [
    ("total_cost",    "总布线代价（↓）",   False),
    ("length_score",  "线长得分（↑）",     True),
    ("sharing_score", "共享线段比例（↑）", True),
    ("max_depth",     "最大路径深度（↓）", False),
    ("time_ms",       "时间 ms（↓）",      False),
]


def _build_table(
    rl_records: list[dict],
    mst_records: list[dict],
) -> tuple[str, str]:
    lamp_counts = sorted({r["n_lamps"] for r in rl_records})
    col_w = 12

    header = f"{'lamps':>6}  {'指标':<18}  {'RL':>{col_w}}  {'MST':>{col_w}}  {'差值(RL-MST)':>{col_w}}"
    sep = "=" * len(header)
    thin = "-" * len(header)
    lines = ["\n" + sep, header, sep]
    md_lines: list[str] = []

    def _add_group(label: str, rl_sub: list[dict], mst_sub: list[dict]) -> None:
        if label == "ALL":
            md_lines.append("\n### 总体\n")
        else:
            md_lines.append(f"\n### {label}灯\n")
        md_lines.append("| 指标 | RL | MST | 差值(RL-MST) |")
        md_lines.append("|------|---:|---:|---:|")

        # cost_ratio
        rl_costs = [r["total_cost"] for r in rl_sub]
        mst_costs = [r["total_cost"] for r in mst_sub]
        ratios = [rc / max(mc, 1e-6) for rc, mc in zip(rl_costs, mst_costs)]
        avg_ratio = _mean(ratios)
        sign = "+" if avg_ratio - 1.0 >= 0 else ""
        lines.append(f"{label:>6}  {'cost_ratio(RL/MST)':<18}  {avg_ratio:{col_w}.4f}  {'1.0000':>{col_w}}  {sign}{avg_ratio-1.0:{col_w}.4f}")
        md_lines.append(f"| cost_ratio (RL/MST) | {avg_ratio:.4f} | 1.0000 | {avg_ratio-1.0:+.4f} |")

        for key, metric_label, higher_better in METRICS:
            rl_val = _mean([r[key] for r in rl_sub])
            mst_val = _mean([r[key] for r in mst_sub])
            diff = rl_val - mst_val
            sign = "+" if diff >= 0 else ""
            lines.append(f"{label:>6}  {metric_label:<18}  {rl_val:{col_w}.2f}  {mst_val:{col_w}.2f}  {sign}{diff:{col_w}.2f}")
            better = ("↑ RL更好" if (higher_better and diff > 0) or (not higher_better and diff < 0)
                      else ("↓ MST更好" if diff != 0 else "持平"))
            md_lines.append(f"| {metric_label} | {rl_val:.2f} | {mst_val:.2f} | {diff:+.2f} ({better}) |")

        if label != "ALL":
            lines.append(thin)

    for lamps in lamp_counts:
        rl_sub = [r for r in rl_records if r["n_lamps"] == lamps]
        mst_sub = [r for r in mst_records if r["n_lamps"] == lamps]
        _add_group(str(lamps), rl_sub, mst_sub)

    _add_group("ALL", rl_records, mst_records)
    lines.append(sep)
    lines.append("（cost_ratio<1.0 表示 RL 布线代价更低；差值 = RL - MST）")

    return "\n".join(lines), "\n".join(md_lines)


def write_markdown(
    md_path: Path,
    test_rooms: list[dict],
    rl_records: list[dict],
    mst_records: list[dict],
    md_table: str,
    model_path: str,
) -> None:
    from collections import Counter
    lamp_dist = dict(sorted(Counter(r.get("_n_lamps", r.get("n_lamps", 0)) for r in test_rooms).items()))
    lines = [
        "# RL 布线模型 vs MST+A* 基线对比",
        "",
        "## 测试集概况",
        f"- 测试房间数：{len(test_rooms)}",
        f"- 灯具数分布：{lamp_dist}",
        f"- 模型：`{model_path}`",
        "",
        "## 指标说明",
        "- **总布线代价**：A* 路径代价之和（含转弯惩罚），越低越好",
        "- **线长得分**：`1 - cost/max_possible`，归一化 [0,1]，越高越好",
        "- **共享线段比例**：多条路径共用的段数/总段数，越高越好",
        "- **最大路径深度**：最长单条路径的格子数，越低越好",
        "- **时间 ms**：单房间推理/优化平均耗时",
        "- **cost_ratio**：RL总代价 / MST总代价，<1.0 表示 RL 更优",
        "",
        "## 按灯具数分组对比",
        md_table,
        "",
        "---",
        "*由 `RL/eval_wiring_compare.py` 自动生成*",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[compare] markdown saved to {md_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RL wiring vs MST+A* baseline")
    parser.add_argument("--room_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_wiring_results")
    parser.add_argument("--split_dir", type=str, default=None)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load test set ─────────────────────────────────────────────────────────
    all_rooms = load_room_dataset(Path(args.room_dir))
    if args.split_dir:
        _, test_rooms = load_wiring_split(Path(args.split_dir), all_rooms)
    else:
        _, test_rooms = split_train_test(all_rooms, test_ratio=args.test_ratio, seed=args.seed)
    print(f"[compare] test set: {len(test_rooms)} rooms")

    # ── load RL model ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WiringPolicyNet(in_channels=6)
    ckpt = torch.load(args.model_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[compare] loaded model from {args.model_path}")

    env_cfg = WiringEnvConfig()

    # ── evaluate ──────────────────────────────────────────────────────────────
    rl_records: list[dict] = []
    mst_records: list[dict] = []

    for i, room_data in enumerate(test_rooms):
        n_lamps = room_data.get("_n_lamps", 0)
        room_name = room_data.get("room_name", f"room_{i:04d}")
        print(f"[{i+1:3d}/{len(test_rooms)}] {room_name}  lamps={n_lamps}", end="  ", flush=True)

        rl_res = run_rl_wiring(room_data, model, env_cfg, device)
        mst_res = run_mst_baseline(room_data, env_cfg)

        ratio = rl_res["total_cost"] / max(mst_res["total_cost"], 1e-6)
        print(
            f"RL cost={rl_res['total_cost']:.1f} ls={rl_res['length_score']:.2f} ({rl_res['time_ms']:.0f}ms)  |  "
            f"MST cost={mst_res['total_cost']:.1f} ls={mst_res['length_score']:.2f} ({mst_res['time_ms']:.0f}ms)  "
            f"ratio={ratio:.3f}"
        )

        rl_records.append({"n_lamps": n_lamps, "room_name": room_name, **rl_res})
        mst_records.append({"n_lamps": n_lamps, "room_name": room_name, **mst_res})

    # ── print & write results ─────────────────────────────────────────────────
    terminal_table, md_table = _build_table(rl_records, mst_records)
    print(terminal_table)

    repo_root = Path(__file__).resolve().parents[1]
    md_path = repo_root / "summary_docs" / "RL-wiring-compare.md"
    write_markdown(md_path, test_rooms, rl_records, mst_records, md_table, args.model_path)

    out_path = output_dir / "wiring_compare_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"rl": rl_records, "mst": mst_records}, f, ensure_ascii=False, indent=2)
    print(f"[compare] json saved to {out_path}")


if __name__ == "__main__":
    main()
