"""
用布局RL模型对空房间推理，筛选质量好的布置结果，生成布线训练数据集。

算法：
  - 维护 pending_queue，每次取一个房间推理
  - 若对齐度 >= align_thresh 且势能比 <= potential_thresh，则保存
  - 否则放回队尾，下次继续推理
  - 直到 saved 达到 target 或迭代次数达到 max_iter

用法：
  cd /home/chen/punchy/CAD_knowledge_service
  .venv/bin/python RL/wiring_dataset_gen.py \\
    --room_dir RL/room_gen/all/json \\
    --model RL/output_multiroom/20260414_002000/ppo_multi_room_best.pt \\
    --output_dir RL/room_gen/RL_layouted_better \\
    --target 600 --max_iter 3000 \\
    --align_thresh 0.6 --potential_thresh 0.5 \\
    --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.env import EnvironmentConfig, SingleRoomLightingEnv
from RL.model import LightingActorCritic
from RL.reward import RewardCalculator, RewardConfig, RoomState
from RL.visualize import render_room_grid


def load_rooms(room_dir: Path) -> list[dict[str, Any]]:
    """加载目录下所有 JSON，返回 room_data 列表。"""
    rooms = []
    for json_path in sorted(room_dir.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if "matrix" in payload:
                candidates = [payload]
            else:
                candidates = list(payload.values())
            for room in candidates:
                if isinstance(room, dict) and "matrix" in room and "lamp_count" in room:
                    # 用文件名（不含扩展名）作为唯一标识，避免 room_name 重复
                    room["_src_stem"] = json_path.stem
                    room["_src_file"] = json_path.name
                    rooms.append(room)
        except Exception as e:
            print(f"[warn] skip {json_path.name}: {e}")
    return rooms


def run_inference(
    room_data: dict[str, Any],
    model: LightingActorCritic,
    device: torch.device,
    padded_size: int,
) -> tuple[np.ndarray, float, float] | None:
    """
    用布局RL模型对单个房间推理。

    Returns:
        (encoded_matrix, alignment_score, potential_ratio) 或 None（房间不合法）
    """
    target_lamps = int(room_data.get("lamp_count", 4))
    env_cfg = EnvironmentConfig(
        padded_size=padded_size,
        target_lamp_count=target_lamps,
    )
    try:
        env = SingleRoomLightingEnv(room_data, config=env_cfg)
    except Exception as e:
        print(f"[warn] env init failed for {room_data.get('room_name','?')}: {e}")
        return None

    initial_potential = env.initial_potential
    obs = env.reset()
    initial_potential = env.initial_potential  # reset 后重新读取

    done = False
    with torch.no_grad():
        while not done:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device, dtype=torch.float32)
            out = model.act(obs_t, deterministic=True)
            action = int(out["action"].item())
            obs, _, done, _ = env.step(action)

    encoded = env.current_encoded_matrix()

    # 计算质量指标
    state = RoomState.from_encoded_matrix(encoded)
    calc = RewardCalculator(RewardConfig(target_lamp_count=target_lamps))
    final_potential = calc.potential(state)
    potential_ratio = final_potential / max(initial_potential, 1.0)
    alignment_score = calc.soft_alignment_score(
        [tuple(map(int, p)) for p in np.argwhere(encoded == 4)]
    )

    return encoded, float(alignment_score), float(potential_ratio)


def save_result(
    room_data: dict[str, Any],
    encoded_matrix: np.ndarray,
    alignment_score: float,
    potential_ratio: float,
    iteration_count: int,
    json_dir: Path,
    image_dir: Path,
    src_stem: str,
) -> None:
    """保存布置结果为 JSON 和可视化图片。"""
    # 构建输出 JSON
    result = {k: v for k, v in room_data.items() if not k.startswith("_")}
    result["matrix"] = encoded_matrix.tolist()
    result["_gen_meta"] = {
        "alignment_score": round(alignment_score, 4),
        "potential_ratio": round(potential_ratio, 4),
        "iterations": iteration_count,
    }

    json_path = json_dir / f"{src_stem}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 可视化
    img = render_room_grid(encoded_matrix, cell_size=36, room_name=None)
    img_path = image_dir / f"{src_stem}.png"
    cv2.imwrite(str(img_path), img)


def main() -> None:
    parser = argparse.ArgumentParser(description="布线数据集生成")
    parser.add_argument("--room_dir", default="RL/room_gen/all/json")
    parser.add_argument("--model", default="RL/output_multiroom/20260414_002000/ppo_multi_room_best.pt")
    parser.add_argument("--output_dir", default="RL/room_gen/RL_layouted_better")
    parser.add_argument("--target", type=int, default=600)
    parser.add_argument("--max_iter", type=int, default=3000)
    parser.add_argument("--align_thresh", type=float, default=0.4)
    parser.add_argument("--potential_thresh", type=float, default=0.9)
    parser.add_argument("--max_retries", type=int, default=5,
                        help="单个房间最多重试次数，超过后直接跳过")
    parser.add_argument("--padded_size", type=int, default=48)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 输出目录
    output_dir = Path(args.output_dir)
    json_dir = output_dir / "json"
    image_dir = output_dir / "images"
    json_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"[init] loading model from {args.model}")
    state_dict = torch.load(args.model, map_location=device)
    model = LightingActorCritic(in_channels=6, target_lamp_count=None)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # 加载房间
    rooms = load_rooms(Path(args.room_dir))
    print(f"[init] loaded {len(rooms)} rooms from {args.room_dir}")
    random.shuffle(rooms)

    # 主循环
    pending: deque[dict[str, Any]] = deque(rooms)
    saved: dict[str, dict] = {}   # src_stem -> meta，用文件名去重
    iterations = 0
    total_align = 0.0
    total_potential = 0.0

    print(f"[gen] target={args.target}  max_iter={args.max_iter}  "
          f"align_thresh={args.align_thresh}  potential_thresh={args.potential_thresh}  "
          f"max_retries={args.max_retries}")

    while len(saved) < args.target and iterations < args.max_iter and pending:
        room = pending.popleft()
        src_stem = room.get("_src_stem", f"room_{iterations:04d}")
        iterations += 1

        result = run_inference(room, model, device, args.padded_size)
        if result is None:
            continue

        encoded, align, pot_ratio = result

        if align >= args.align_thresh and pot_ratio <= args.potential_thresh:
            save_result(room, encoded, align, pot_ratio, iterations, json_dir, image_dir, src_stem)
            saved[src_stem] = {"alignment": align, "potential_ratio": pot_ratio}
            total_align += align
            total_potential += pot_ratio

        if iterations % 100 == 0:
            pass_rate = len(saved) / iterations * 100
            print(f"[gen] iter={iterations}/{len(rooms)}  saved={len(saved)}/{args.target}  "
                  f"pass_rate={pass_rate:.1f}%")

    # 汇总
    n = len(saved)
    avg_align = total_align / max(n, 1)
    avg_pot = total_potential / max(n, 1)
    print(f"\n[done] saved={n}/{args.target}  total_iter={iterations}  "
          f"avg_alignment={avg_align:.4f}  avg_potential_ratio={avg_pot:.4f}")
    print(f"[done] json -> {json_dir}")
    print(f"[done] images -> {image_dir}")


if __name__ == "__main__":
    main()
