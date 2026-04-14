"""
布线RL训练脚本

复用灯具布局RL的PPO框架（GAE、rollout buffer、PPO更新），
针对布线任务的特点做适配：
  - 动作空间是灯具索引（N维），而非格子索引（H×W维）
  - 每步需要传入 lamp_coords 和 action_mask
  - 支持课程学习：先训练灯具少的房间，再训练灯具多的房间
  - 训练结束后与 MST baseline 对比

用法：
  # 单房间训练
  python RL/train_wiring.py --room RL/test_room/origin_room/unregular.json

  # 多房间训练（从目录加载）
  python RL/train_wiring.py --room_dir RL/generated_rooms --episodes 5000

  # 启用课程学习
  python RL/train_wiring.py --room_dir RL/generated_rooms --curriculum

  # 使用预定义的训练/测试集划分训练
  cd /home/chen/punchy/CAD_knowledge_service/RL
../.venv/bin/python train_wiring.py \
  --room_dir test_room/layout_room/json \
  --split_dir test_room/layout_room/split \
  --output_dir output_wiring
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wiring_env import WiringEnv, WiringEnvConfig, compute_mst_baseline
from wiring_model import WiringPolicyNet, build_lamp_coords_tensor

import yaml

def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2


# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------

@dataclass
class WiringPPOConfig:
    episodes: int = 3000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    ppo_epochs: int = 4
    rollout_episodes: int = 16
    minibatch_size: int = 256
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip_norm: float = 0.5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every_episodes: int = 32
    visualize_every_episodes: int = 100
    reward_curve_moving_window: int = 50
    reward_curve_bias: float = 0.0


@dataclass
class CurriculumStage:
    min_lamps: int
    max_lamps: int
    episodes: int


@dataclass
class WiringCurriculumConfig:
    stages: list[CurriculumStage] = None

    def __post_init__(self):
        if self.stages is None:
            self.stages = [
                CurriculumStage(min_lamps=2, max_lamps=3, episodes=500),
                CurriculumStage(min_lamps=2, max_lamps=5, episodes=1000),
                CurriculumStage(min_lamps=2, max_lamps=7, episodes=1500),
                CurriculumStage(min_lamps=2, max_lamps=9, episodes=2000),
            ]


# ------------------------------------------------------------------
# 数据加载
# ------------------------------------------------------------------

def load_room_dataset(room_dir: Path) -> list[dict[str, Any]]:
    """从目录加载所有房间 JSON，过滤掉没有灯具的房间。"""
    rooms = []
    for json_file in sorted(room_dir.glob("*.json")):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                continue
            # 支持单房间格式（直接含 matrix）和多房间格式
            if "matrix" in payload:
                candidates = [payload]
            else:
                candidates = list(payload.values())
            for room_data in candidates:
                if not isinstance(room_data, dict) or "matrix" not in room_data:
                    continue
                matrix = np.asarray(room_data["matrix"], dtype=np.int32)
                n_lamps = int((matrix == 4).sum())
                n_switches = int((matrix == 3).sum())
                if n_lamps >= 2 and n_switches >= 1:
                    room_data["_n_lamps"] = n_lamps
                    room_data["_filename"] = json_file.name
                    rooms.append(room_data)
        except Exception as e:
            print(f"[warn] skip {json_file}: {e}")
    print(f"[dataset] loaded {len(rooms)} rooms from {room_dir}")
    return rooms


def filter_by_lamp_count(rooms: list[dict], min_lamps: int, max_lamps: int) -> list[dict]:
    return [r for r in rooms if min_lamps <= r.get("_n_lamps", 0) <= max_lamps]


def load_wiring_split(
    split_dir: Path,
    all_rooms: list[dict[str, Any]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """从预分割索引文件加载训练集、验证集和测试集。"""
    by_name: dict[str, dict] = {}
    for room in all_rooms:
        fname = room.get("_filename", "")
        by_name[fname] = room

    result = []
    for split_name in ("train", "val", "test"):
        idx_path = split_dir / f"{split_name}.json"
        if not idx_path.exists():
            print(f"[load_wiring_split] {split_name}.json not found, returning empty list")
            result.append([])
            continue
        with idx_path.open("r", encoding="utf-8") as f:
            entries = json.load(f)
        rooms = []
        for entry in entries:
            fname = entry["filename"]
            if fname in by_name:
                rooms.append(by_name[fname])
            else:
                print(f"[load_wiring_split] WARNING: {fname} not found, skipping")
        print(f"[load_wiring_split] {split_name}: {len(rooms)} rooms")
        result.append(rooms)
    return result[0], result[1], result[2]


def split_train_test(
    rooms: list[dict[str, Any]],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """按灯具数量分层划分训练集和测试集。"""
    from collections import defaultdict
    rng = random.Random(seed)

    by_lamps: defaultdict[int, list[dict]] = defaultdict(list)
    for r in rooms:
        by_lamps[r.get("_n_lamps", 0)].append(r)

    train, test = [], []
    for n_lamps in sorted(by_lamps):
        group = by_lamps[n_lamps]
        rng.shuffle(group)
        n_test = max(1, int(len(group) * test_ratio))
        test.extend(group[:n_test])
        train.extend(group[n_test:])

    print(f"[split] train={len(train)}, test={len(test)}")
    return train, test


# ------------------------------------------------------------------
# GAE 计算（与 train.py 相同）
# ------------------------------------------------------------------

def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else last_value
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    adv = torch.tensor(advantages, dtype=torch.float32)
    val = torch.tensor(values, dtype=torch.float32)
    return adv, adv + val


# ------------------------------------------------------------------
# Episode 收集
# ------------------------------------------------------------------

def _collect_one_episode(
    env: WiringEnv,
    model: WiringPolicyNet,
    device: torch.device,
) -> dict[str, Any]:
    """
    执行一个完整的布线 episode，返回轨迹数据和统计信息。
    """
    obs = env.reset()

    # 预计算灯具坐标 tensor（固定不变）
    lamp_coords_t = build_lamp_coords_tensor(
        env.lamp_positions, env.row_offset, env.col_offset, device
    )  # [1, N, 2]

    obs_list: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    log_probs: list[float] = []
    values: list[float] = []
    # 每步的 action_mask（用于 evaluate_actions）
    masks_list: list[np.ndarray] = []

    done = False
    ##TODO: 这里存在一个bug，如果从开关的位置出发，无论如何都无法连接到灯具，即房间的设计不合理，在此情况下进行强化学习布线会陷入死循环而且很难察觉
    while not done:
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
        mask = env.action_mask()  # [N] bool
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device=device)  # [1, N]

        rollout = model.act(obs_t, lamp_coords_t, mask_t, deterministic=False)
        action = int(rollout["action"].item())

        next_obs, reward, done, info = env.step(action)

        obs_list.append(obs)
        actions.append(action)
        rewards.append(float(reward))
        log_probs.append(float(rollout["log_prob"].item()))
        values.append(float(rollout["value"].squeeze().item()))
        masks_list.append(mask)
        obs = next_obs

    # 终局统计
    last_info = info
    return {
        "obs_list": obs_list,
        "actions": actions,
        "rewards": rewards,
        "log_probs": log_probs,
        "values": values,
        "masks_list": masks_list,
        "lamp_positions": env.lamp_positions,
        "row_offset": env.row_offset,
        "col_offset": env.col_offset,
        "episode_reward": float(sum(rewards)),
        "n_lamps": env.n_lamps,
        "total_cost": float(last_info.get("total_cost", 0.0)),
        "length_score": float(last_info.get("length_score", 0.0)),
        "sharing_score": float(last_info.get("sharing_score", 0.0)),
        "max_depth": int(last_info.get("max_depth", 0)),
    }


# ------------------------------------------------------------------
# PPO 更新
# ------------------------------------------------------------------

def ppo_update(
    model: WiringPolicyNet,
    optimizer: torch.optim.Optimizer,
    rollout_buffer: list[dict[str, Any]],
    cfg: WiringPPOConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    对 rollout_buffer 中的所有 episode 做 PPO 更新。
    返回 (policy_loss, value_loss, entropy)。
    """
    # ── 拼接所有 episode 的轨迹 ──
    all_obs: list[np.ndarray] = []
    all_actions: list[int] = []
    all_advantages: list[float] = []
    all_returns: list[float] = []
    all_old_log_probs: list[float] = []
    # 每步的 lamp_coords 和 action_mask（因为不同 episode 的灯具数可能不同，需要 padding）
    all_lamp_coords: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []

    max_n_lamps = max(ep["n_lamps"] for ep in rollout_buffer)

    for ep in rollout_buffer:
        adv, ret = compute_gae(ep["rewards"], ep["values"], cfg.gamma, cfg.gae_lambda)
        T = len(ep["rewards"])
        N = ep["n_lamps"]

        # 计算该 episode 的灯具坐标 tensor（填充到 max_n_lamps）
        lamp_coords_t = build_lamp_coords_tensor(
            ep["lamp_positions"], ep["row_offset"], ep["col_offset"], device
        )  # [1, N, 2]
        # padding 到 max_n_lamps
        if N < max_n_lamps:
            pad = torch.zeros(1, max_n_lamps - N, 2, device=device)
            lamp_coords_t = torch.cat([lamp_coords_t, pad], dim=1)

        for t in range(T):
            all_obs.append(ep["obs_list"][t])
            all_actions.append(ep["actions"][t])
            all_advantages.append(float(adv[t]))
            all_returns.append(float(ret[t]))
            all_old_log_probs.append(ep["log_probs"][t])

            # action_mask padding
            mask = ep["masks_list"][t]  # [N] bool
            if N < max_n_lamps:
                mask = np.concatenate([mask, np.zeros(max_n_lamps - N, dtype=bool)])
            all_masks.append(torch.from_numpy(mask).to(device))
            all_lamp_coords.append(lamp_coords_t.squeeze(0))  # [max_N, 2]

    # ── 转为 tensor ──
    obs_t = torch.from_numpy(np.stack(all_obs)).to(device=device, dtype=torch.float32)
    actions_t = torch.tensor(all_actions, dtype=torch.long, device=device)
    old_log_probs_t = torch.tensor(all_old_log_probs, dtype=torch.float32, device=device)
    returns_t = torch.tensor(all_returns, dtype=torch.float32, device=device)
    adv_t = torch.tensor(all_advantages, dtype=torch.float32, device=device)
    masks_t = torch.stack(all_masks)  # [T, max_N]
    lamp_coords_t = torch.stack(all_lamp_coords)  # [T, max_N, 2]

    # ── 归一化 advantages ──
    if adv_t.numel() > 1:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

    total_steps = obs_t.shape[0]
    mb_size = max(1, cfg.minibatch_size)

    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0

    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(total_steps, device=device)
        for start in range(0, total_steps, mb_size):
            idx = perm[start:start + mb_size]
            evaluated = model.evaluate_actions(obs_t[idx], lamp_coords_t[idx], masks_t[idx], actions_t[idx])
            log_probs_mb = evaluated["log_prob"]
            entropy_mb = evaluated["entropy"]
            values_mb = evaluated["value"].squeeze(-1)

            ratio = torch.exp(log_probs_mb - old_log_probs_t[idx])
            adv_mb = adv_t[idx]
            s1 = ratio * adv_mb
            s2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_mb
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = nn.functional.mse_loss(values_mb, returns_t[idx])
            entropy_bonus = entropy_mb.mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_entropy = float(entropy_bonus.item())

    return last_policy_loss, last_value_loss, last_entropy


# ------------------------------------------------------------------
# 主训练函数
# ------------------------------------------------------------------

def train_wiring(
    rooms: list[dict[str, Any]],
    model: WiringPolicyNet,
    cfg: WiringPPOConfig,
    env_cfg: WiringEnvConfig,
    output_dir: Path,
    curriculum: WiringCurriculumConfig | None = None,
    val_rooms: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    多房间布线PPO训练主循环。

    支持课程学习：按灯具数量从少到多逐步扩展训练集。
    """
    device = torch.device(cfg.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history: list[dict[str, Any]] = []
    best_reward = float("-inf")
    best_model_path = output_dir / "wiring_best.pt"

    # 构建 episode 迭代器（支持课程学习）
    if curriculum is not None:
        episode_plan: list[tuple[list[dict], int]] = []
        for stage in curriculum.stages:
            pool = filter_by_lamp_count(rooms, stage.min_lamps, stage.max_lamps)
            if not pool:
                print(f"[warn] curriculum stage {stage.min_lamps}-{stage.max_lamps} has no rooms, skip")
                continue
            episode_plan.append((pool, stage.episodes))
        total_episodes = sum(n for _, n in episode_plan)
    else:
        episode_plan = [(rooms, cfg.episodes)]
        total_episodes = cfg.episodes

    print(f"[train_wiring] total_episodes={total_episodes}, rooms={len(rooms)}, device={device}")

    episode_idx = 0
    last_policy_loss = last_value_loss = last_entropy = 0.0

    for pool, stage_episodes in episode_plan:
        stage_end = episode_idx + stage_episodes
        stage_name = f"lamps={pool[0].get('_n_lamps','?')}-{pool[-1].get('_n_lamps','?')}" if pool else "all"

        while episode_idx < stage_end:
            # ── 收集 rollout_episodes 个 episode ──
            rollout_buffer: list[dict[str, Any]] = []
            batch_size = min(cfg.rollout_episodes, stage_end - episode_idx)

            for _ in range(batch_size):
                episode_idx += 1
                room_data = random.choice(pool)
                try:
                    env = WiringEnv(room_data, config=env_cfg)
                except Exception as e:
                    print(f"[warn] skip room {room_data.get('room_name','?')}: {e}")
                    episode_idx -= 1
                    continue
                ep = _collect_one_episode(env, model, device)
                rollout_buffer.append(ep)

            if not rollout_buffer:
                continue

            # ── PPO 更新 ──
            last_policy_loss, last_value_loss, last_entropy = ppo_update(
                model, optimizer, rollout_buffer, cfg, device
            )

            # ── 记录统计 ──
            for ep in rollout_buffer:
                history.append({
                    "episode": len(history) + 1,
                    "episode_reward": ep["episode_reward"],
                    "n_lamps": ep["n_lamps"],
                    "total_cost": ep["total_cost"],
                    "length_score": ep["length_score"],
                    "sharing_score": ep["sharing_score"],
                    "max_depth": ep["max_depth"],
                    "policy_loss": last_policy_loss,
                    "value_loss": last_value_loss,
                    "entropy": last_entropy,
                    "stage": stage_name,
                })

            # 按 batch 平均奖励保存最佳模型
            batch_avg_reward = sum(ep["episode_reward"] for ep in rollout_buffer) / len(rollout_buffer)
            if batch_avg_reward > best_reward:
                best_reward = batch_avg_reward
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "episode": episode_idx,
                    "episode_reward": batch_avg_reward,
                }, best_model_path)

            # ── 日志 ──
            if episode_idx % cfg.log_every_episodes == 0 or episode_idx >= total_episodes:
                recent = [h for h in history[-cfg.log_every_episodes * 2:] if h.get("type") != "validation"][-cfg.log_every_episodes:]
                moving_reward = sum(h["episode_reward"] for h in recent) / len(recent)
                last_ep = rollout_buffer[-1]
                print(
                    f"[wiring] ep={episode_idx:05d}/{total_episodes} "
                    f"reward={last_ep['episode_reward']:7.3f} "
                    f"moving={moving_reward:7.3f} "
                    f"cost={last_ep['total_cost']:.1f} "
                    f"len_score={last_ep['length_score']:.3f} "
                    f"share={last_ep['sharing_score']:.3f} "
                    f"depth={last_ep['max_depth']} "
                    f"lamps={last_ep['n_lamps']} "
                    f"ploss={last_policy_loss:.4f} vloss={last_value_loss:.4f} "
                    f"stage={stage_name}"
                )

            # ── 验证集评估（每个课程阶段末 + 每4个log周期）──
            run_val = (
                val_rooms
                and (
                    episode_idx >= stage_end
                    or episode_idx % (cfg.log_every_episodes * 4) == 0
                )
            )
            if run_val:
                model.eval()
                val_rewards, val_costs = [], []
                with torch.no_grad():
                    for vroom in val_rooms:
                        venv = WiringEnv(vroom, config=env_cfg)
                        vobs = venv.reset()
                        vdone = False
                        vtotal = 0.0
                        while not vdone:
                            vobs_t = torch.from_numpy(vobs).unsqueeze(0).to(device, dtype=torch.float32)
                            vlamp_coords = build_lamp_coords_tensor(
                                venv.lamp_positions, venv.row_offset, venv.col_offset, device
                            )
                            vmask = torch.tensor(
                                [not venv.connected[i] for i in range(venv.n_lamps)],
                                dtype=torch.bool, device=device,
                            ).unsqueeze(0)
                            vout = model.act(vobs_t, vlamp_coords, vmask, deterministic=True)
                            vobs, vr, vdone, vinfo = venv.step(int(vout["action"].item()))
                            vtotal += float(vr)
                        val_rewards.append(vtotal)
                        val_costs.append(float(vinfo.get("total_cost", 0.0)))
                model.train()
                avg_val_reward = float(np.mean(val_rewards))
                avg_val_cost = float(np.mean(val_costs))
                history.append({
                    "type": "validation",
                    "episode": episode_idx,
                    "avg_reward": avg_val_reward,
                    "avg_cost": avg_val_cost,
                    "stage": stage_name,
                })
                print(
                    f"[val]    ep={episode_idx:05d} avg_reward={avg_val_reward:7.3f} "
                    f"avg_cost={avg_val_cost:.1f} stage={stage_name}"
                )

            # ── 可视化（训练过程中不再保存，训练结束后统一用测试集生成）──

    summary = {
        "total_episodes": total_episodes,
        "best_reward": best_reward,
        "best_model_path": str(best_model_path),
        "history": history,
    }
    return summary


# ------------------------------------------------------------------
# 布线结果可视化
# ------------------------------------------------------------------

def visualize_wiring_result(
    room_data: dict[str, Any],
    model: WiringPolicyNet,
    env_cfg: WiringEnvConfig,
    device: torch.device,
    output_path: Path,
    episode_idx: int,
    *,
    cell_size: int = 32,
) -> None:
    """
    用贪婪策略跑一遍布线，将 A* 路径画在房间网格图上并保存。

    颜色约定（BGR）：
      - 房间底图：复用 render_room_grid（0=红, 1=绿, 2=蓝, 3=黄, 4=黑）
      - 线路：青色 (0, 220, 220)，线宽 2px
      - 接入点：橙色圆点
    """
    from visualize import render_room_grid

    env = WiringEnv(room_data, config=env_cfg)
    obs = env.reset()
    lamp_coords_t = build_lamp_coords_tensor(
        env.lamp_positions, env.row_offset, env.col_offset, device
    )

    # 贪婪 rollout，收集每步路径
    paths: list[list[tuple[int, int]]] = []
    entry_points: list[tuple[int, int]] = []
    done = False
    total_cost = 0.0
    while not done:
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
        mask_t = torch.from_numpy(env.action_mask()).unsqueeze(0).to(device)
        rollout = model.act(obs_t, lamp_coords_t, mask_t, deterministic=True)
        action = int(rollout["action"].item())
        obs, _, done, info = env.step(action)
        if "route_cost" in info:
            total_cost += info["route_cost"]
        # 从 env 的 step_paths 取最新路径
        if env.step_paths:
            paths.append(env.step_paths[-1])
            entry_points.append(info.get("entry_point", env.switch_pos))

    # 渲染房间底图（使用原始 matrix，灯具已编码为 4）
    matrix = np.asarray(room_data["matrix"], dtype=np.int32)
    img = render_room_grid(matrix, cell_size=cell_size, room_name=None)

    PADDING = 12
    x0, y0 = PADDING, PADDING

    # 在底图上绘制线路
    color_wire = (0, 220, 220)   # 青色
    color_entry = (0, 140, 255)  # 橙色
    thickness = 2

    for path in paths:
        pts = []
        for r, c in path:
            px = x0 + c * cell_size + cell_size // 2
            py = y0 + r * cell_size + cell_size // 2
            pts.append((px, py))
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], color_wire, thickness, cv2.LINE_AA)

    # 标记接入点（橙色圆）
    for ep in entry_points:
        r, c = ep
        px = x0 + c * cell_size + cell_size // 2
        py = y0 + r * cell_size + cell_size // 2
        cv2.circle(img, (px, py), cell_size // 3, color_entry, -1)

    # 在图片底部写统计信息
    rows, cols = matrix.shape
    text_y = y0 + rows * cell_size + 6
    stats = (
        f"轮次={episode_idx}  代价={total_cost:.1f}  "
        f"灯具数={env.n_lamps}  路径数={len(paths)}"
    )
    # 用 Pillow 写中英文
    from PIL import Image, ImageDraw
    from visualize import _load_font
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _load_font(20)
    draw.text((x0, text_y), stats, fill=(30, 30, 30), font=font)
    img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)




def compare_with_mst(
    rooms: list[dict[str, Any]],
    model: WiringPolicyNet,
    env_cfg: WiringEnvConfig,
    device: torch.device,
    n_samples: int | None = None,
) -> dict[str, Any]:
    """
    对比 RL 策略与 MST baseline。n_samples=None 时评估全部房间。
    """
    if n_samples is not None:
        sample_rooms = random.sample(rooms, min(n_samples, len(rooms)))
    else:
        sample_rooms = rooms
    results = []

    for idx, room_data in enumerate(sample_rooms):
        try:
            mst = compute_mst_baseline(room_data, turn_penalty=env_cfg.turn_penalty)

            env = WiringEnv(room_data, config=env_cfg)
            obs = env.reset()
            lamp_coords_t = build_lamp_coords_tensor(
                env.lamp_positions, env.row_offset, env.col_offset, device
            )
            done = False
            while not done:
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
                mask_t = torch.from_numpy(env.action_mask()).unsqueeze(0).to(device)
                rollout = model.act(obs_t, lamp_coords_t, mask_t, deterministic=True)
                action = int(rollout["action"].item())
                obs, _, done, info = env.step(action)
            rl_total_cost = info.get("total_cost", 0.0)

            results.append({
                "room_name": room_data.get("room_name", f"room_{idx}"),
                "n_lamps": env.n_lamps,
                "mst_cost": mst["total_cost"],
                "rl_cost": rl_total_cost,
                "cost_ratio": rl_total_cost / max(mst["total_cost"], 1.0),
            })
        except Exception as e:
            print(f"[warn] compare failed for {room_data.get('room_name','?')}: {e}")

    if not results:
        return {}

    avg_cost_ratio = np.mean([r["cost_ratio"] for r in results])
    avg_rl = np.mean([r["rl_cost"] for r in results])
    avg_mst = np.mean([r["mst_cost"] for r in results])
    print(f"\n[MST comparison] n_rooms={len(results)}")
    print(f"  avg RL cost:  {avg_rl:.1f}")
    print(f"  avg MST cost: {avg_mst:.1f}")
    print(f"  RL/MST cost ratio: {avg_cost_ratio:.3f} (1.0=same, <1.0=RL better)")

    return {"results": results, "avg_cost_ratio": float(avg_cost_ratio)}


# ------------------------------------------------------------------
# 可视化
# ------------------------------------------------------------------

def plot_wiring_reward_curve(history: list[dict], output_path: Path, window: int = 50, bias: float = 0.0) -> None:
    from matplotlib.font_manager import FontProperties
    from matplotlib import rcParams

    _CN_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
    cn_font = FontProperties(fname=_CN_FONT_PATH, size=16)
    cn_font_legend = FontProperties(fname=_CN_FONT_PATH, size=16)  ## 修改字号
    rcParams['axes.unicode_minus'] = False

    if not history:
        return
    train_h = [h for h in history if h.get("type") != "validation"]
    val_h   = [h for h in history if h.get("type") == "validation"]

    episodes = [h["episode"] for h in train_h]
    rewards  = np.array([h["episode_reward"] for h in train_h], dtype=np.float32) + bias
    costs    = np.array([h["total_cost"]     for h in train_h], dtype=np.float32)

    w = max(1, min(window, len(rewards)))
    kernel = np.ones(w) / w
    moving = np.convolve(rewards, kernel, mode="valid")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

    # ── 上图：奖励曲线 ──
    ax1.plot(episodes, rewards, color="#7aa6ff", alpha=0.5, linewidth=0.8, label="训练奖励")
    ax1.plot(episodes[w - 1:], moving, color="#2563eb", linewidth=2.0, label=f"训练滑动平均（窗口={w}）")
    if val_h:
        val_eps = [h["episode"] for h in val_h]
        val_rws = np.array([h["avg_reward"] for h in val_h], dtype=np.float32) + bias
        ax1.plot(val_eps, val_rws, color="#e74c3c", linewidth=2.0,
                 marker="o", markersize=4, label="验证集平均奖励")
    ax1.set_xlabel("训练轮次", fontproperties=cn_font)
    ax1.set_ylabel("奖励", fontproperties=cn_font)
    ax1.legend(prop=cn_font_legend)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.tick_params(axis='both', labelsize=11)

    # ── 下图：布线代价曲线 ──
    ax2.plot(episodes, costs, color="#4caf50", alpha=0.6, linewidth=1.0, label="训练布线代价")
    if val_h:
        val_eps = [h["episode"] for h in val_h]
        val_costs = np.array([h["avg_cost"] for h in val_h], dtype=np.float32)
        ax2.plot(val_eps, val_costs, color="#e67e22", linewidth=2.0,
                 marker="s", markersize=4, label="验证集平均代价")
    ax2.set_xlabel("训练轮次", fontproperties=cn_font)
    ax2.set_ylabel("布线代价（格子步数）", fontproperties=cn_font)
    ax2.legend(prop=cn_font_legend)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.tick_params(axis='both', labelsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rl_vs_mst_comparison(results: list[dict[str, Any]], output_path: Path) -> None:
    """测试集上 RL vs MST 成本对比柱状图。"""
    if not results:
        return
    from collections import defaultdict
    from matplotlib.font_manager import FontProperties
    from matplotlib import rcParams

    _CN_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
    cn_font = FontProperties(fname=_CN_FONT_PATH, size=14)
    cn_font_legend = FontProperties(fname=_CN_FONT_PATH, size=12)
    cn_font_small = FontProperties(fname=_CN_FONT_PATH, size=9)
    rcParams['axes.unicode_minus'] = False

    # 按灯具数排序
    results = sorted(results, key=lambda r: (r["n_lamps"], r["room_name"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(results) * 0.5), 10))

    # ── 上图：逐房间对比 ──
    x = np.arange(len(results))
    width = 0.35
    rl_costs = [r["rl_cost"] for r in results]
    mst_costs = [r["mst_cost"] for r in results]
    labels = [f"{r['room_name']}\n({r['n_lamps']}灯)" for r in results]

    ax1.bar(x - width / 2, rl_costs, width, label="强化学习", color="#4a90d9", alpha=0.85)
    ax1.bar(x + width / 2, mst_costs, width, label="最小生成树", color="#e8833a", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, fontsize=6)
    ax1.set_ylabel("布线代价", fontproperties=cn_font)
    avg_ratio = np.mean([r["cost_ratio"] for r in results])
    ax1.text(0.99, 0.97, f"平均比值={avg_ratio:.3f}", transform=ax1.transAxes,
             ha='right', va='top', fontproperties=cn_font_small)
    ax1.legend(prop=cn_font_legend)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax1.tick_params(axis='both', labelsize=11)

    # ── 下图：按灯具数分组平均 ──
    by_lamps: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_lamps[r["n_lamps"]].append(r)

    lamp_groups = sorted(by_lamps.keys())
    avg_rl = [np.mean([r["rl_cost"] for r in by_lamps[n]]) for n in lamp_groups]
    avg_mst = [np.mean([r["mst_cost"] for r in by_lamps[n]]) for n in lamp_groups]
    group_labels = [f"{n}灯\n(n={len(by_lamps[n])})" for n in lamp_groups]

    x2 = np.arange(len(lamp_groups))
    ax2.bar(x2 - width / 2, avg_rl, width, label="强化学习（均值）", color="#4a90d9", alpha=0.85)
    ax2.bar(x2 + width / 2, avg_mst, width, label="最小生成树（均值）", color="#e8833a", alpha=0.85)
    # 在柱子上标注 ratio
    for i, n in enumerate(lamp_groups):
        ratio = avg_rl[i] / max(avg_mst[i], 1.0)
        ax2.text(i, max(avg_rl[i], avg_mst[i]) + 1, f"{ratio:.2f}", ha="center", fontsize=9)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(group_labels)
    ax2.set_ylabel("平均布线代价", fontproperties=cn_font)
    ax2.legend(prop=cn_font_legend)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.tick_params(axis='both', labelsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[plot] RL vs MST comparison saved to {output_path}")


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="布线RL训练")
    parser.add_argument("--room", type=str, default=None, help="单个房间 JSON 路径")
    parser.add_argument("--room_dir", type=str, default="RL/room_gen/RL_layouted_better/json", help="房间数据集目录")
    parser.add_argument("--curriculum", action="store_true", help="启用课程学习")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="RL/output_wiring")
    parser.add_argument("--compare_mst", action="store_true", help="训练后与MST对比")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--split_dir", type=str, default="RL/room_gen/RL_layouted_better/split", help="预分割索引目录（train.json/test.json）")
    args = parser.parse_args()

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # 配置
    config_path = Path(__file__).resolve().parent / "config_wiring.yaml"
    cfg_payload = _load_yaml_config(config_path)

    ppo_raw = cfg_payload.get("wiring_training", {})
    ppo_cfg = WiringPPOConfig(
        episodes=ppo_raw.get("episodes", 5000),
        gamma=ppo_raw.get("gamma", 0.99),
        gae_lambda=ppo_raw.get("gae_lambda", 0.95),
        learning_rate=ppo_raw.get("learning_rate", 3e-4),
        ppo_epochs=ppo_raw.get("ppo_epochs", 4),
        rollout_episodes=ppo_raw.get("rollout_episodes", 16),
        minibatch_size=ppo_raw.get("minibatch_size", 256),
        clip_eps=ppo_raw.get("clip_eps", 0.2),
        value_coef=ppo_raw.get("value_coef", 0.5),
        entropy_coef=ppo_raw.get("entropy_coef", 0.01),
        grad_clip_norm=ppo_raw.get("grad_clip_norm", 0.5),
        seed=ppo_raw.get("seed", 42),
        device=args.device if args.device else ppo_raw.get("device", "cuda"),
        log_every_episodes=ppo_raw.get("log_every_episodes", 32),
        visualize_every_episodes=ppo_raw.get("visualize_every_episodes", 100),
        reward_curve_moving_window=ppo_raw.get("reward_curve_moving_window", 50),
        reward_curve_bias=ppo_raw.get("reward_curve_bias", 0.0),
    )

    env_raw = cfg_payload.get("wiring_environment", {})
    env_cfg = WiringEnvConfig(
        padded_size=env_raw.get("padded_size", 48),
        turn_penalty=env_raw.get("turn_penalty", 0.2),
        step_cost_coef=env_raw.get("step_cost_coef", 0.3),
        invalid_action_penalty=env_raw.get("invalid_action_penalty", 1.0),
        terminal_length_coef=env_raw.get("terminal_length_coef", 1.0),
    )

    if args.curriculum:
        cur_raw = cfg_payload.get("wiring_curriculum", {})
        raw_stages = cur_raw.get("stages", [])
        if raw_stages:
            curriculum = WiringCurriculumConfig(
                stages=[CurriculumStage(**s) for s in raw_stages]
            )
        else:
            curriculum = WiringCurriculumConfig()
    else:
        curriculum = None

    random.seed(ppo_cfg.seed)
    np.random.seed(ppo_cfg.seed)
    torch.manual_seed(ppo_cfg.seed)

    # 加载房间数据
    test_rooms: list[dict[str, Any]] = []
    val_rooms: list[dict[str, Any]] = []
    if args.room_dir:
        all_rooms = load_room_dataset(Path(args.room_dir))
        if args.split_dir:
            train_rooms, val_rooms, test_rooms = load_wiring_split(Path(args.split_dir), all_rooms)
        else:
            train_rooms, test_rooms = split_train_test(all_rooms, test_ratio=args.test_ratio, seed=ppo_cfg.seed)
        rooms = train_rooms
    elif args.room:
        room_path = Path(args.room)
        with room_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "matrix" in payload:
            rooms = [payload]
        else:
            rooms = list(payload.values())
        for r in rooms:
            matrix = np.asarray(r["matrix"], dtype=np.int32)
            r["_n_lamps"] = int((matrix == 4).sum())
        rooms = [r for r in rooms if r["_n_lamps"] >= 2]
    else:
        parser.error("需要指定 --room 或 --room_dir")

    if not rooms:
        print("[error] 没有可用的房间数据")
        return

    print(f"[main] train={len(rooms)}, val={len(val_rooms)}, test={len(test_rooms)}, episodes={ppo_cfg.episodes}, device={ppo_cfg.device}")

    # 预热所有房间的 MST 缓存，避免训练开始后长时间无响应
    all_rooms_to_warm = rooms + (val_rooms or []) + (test_rooms or [])
    print(f"[main] 预计算 MST cache ({len(all_rooms_to_warm)} 个房间)...")
    for i, room in enumerate(all_rooms_to_warm):
        WiringEnv(room, config=env_cfg)  # __init__ 里会计算并缓存 MST
        if (i + 1) % 50 == 0 or (i + 1) == len(all_rooms_to_warm):
            print(f"[main] MST cache: {i + 1}/{len(all_rooms_to_warm)}")
    print("[main] MST cache 预计算完成")

    # 创建模型
    model = WiringPolicyNet(in_channels=6)

    # 训练（只用训练集，验证集用于监控过拟合）
    summary = train_wiring(rooms, model, ppo_cfg, env_cfg, output_dir, curriculum=curriculum, val_rooms=val_rooms or None)

    # 保存训练曲线
    plot_wiring_reward_curve(
        summary["history"],
        output_dir / "wiring_reward_curve.png",
        window=ppo_cfg.reward_curve_moving_window,
        bias=ppo_cfg.reward_curve_bias,
    )

    # 保存训练摘要
    summary_path = output_dir / "wiring_training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in summary.items() if k != "history"}, f, ensure_ascii=False, indent=2)
    print(f"[main] summary saved to {summary_path}")

    # 复制配置文件到输出目录，方便复现
    import shutil
    shutil.copy(config_path, output_dir / "config_wiring.yaml")
    print(f"[main] config saved to {output_dir / 'config_wiring.yaml'}")

    # 加载最佳模型
    device = torch.device(ppo_cfg.device)
    best_ckpt = torch.load(summary["best_model_path"], map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # 在测试集上评估 RL vs MST
    eval_rooms = test_rooms if test_rooms else rooms
    eval_label = "test" if test_rooms else "train"
    if args.compare_mst or test_rooms:
        compare_result = compare_with_mst(eval_rooms, model, env_cfg, device, n_samples=None)
        if compare_result:
            compare_path = output_dir / "mst_comparison.json"
            with compare_path.open("w", encoding="utf-8") as f:
                json.dump(compare_result, f, ensure_ascii=False, indent=2)
            print(f"[main] MST comparison ({eval_label} set, {len(eval_rooms)} rooms) saved to {compare_path}")

            # 生成对比柱状图
            plot_rl_vs_mst_comparison(
                compare_result["results"],
                output_dir / "rl_vs_mst_comparison.png",
            )

    # 在测试集上保存布线可视化（全部测试房间）
    vis_dir = output_dir / "wiring_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"[main] saving wiring vis for {len(eval_rooms)} test rooms...")
    for idx, room_data in enumerate(eval_rooms):
        room_name = room_data.get("room_name", f"room_{idx:04d}")
        safe_name = room_name.replace("/", "_").replace("\\", "_")
        vis_path = vis_dir / f"{idx:04d}_{safe_name}.png"
        try:
            visualize_wiring_result(
                room_data, model, env_cfg, device, vis_path,
                episode_idx=summary["total_episodes"],
            )
        except Exception as e:
            print(f"[warn] vis failed for {room_name}: {e}")
    print(f"[main] wiring vis saved to {vis_dir}")

    print(f"\n[main] done. best_reward={summary['best_reward']:.3f}")
    print(f"[main] output: {output_dir}")


if __name__ == "__main__":
    main()
