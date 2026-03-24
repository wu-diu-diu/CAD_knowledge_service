from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import yaml
from torch import nn

from env import EnvironmentConfig, SingleRoomLightingEnv
from model import LightingActorCritic
from reward import RewardConfig
from visualize import plot_episode_step_breakdown, save_padded_room_image, save_room_grid_image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class PPOConfig:
    """Training hyper-parameters for the single-room PPO smoke test."""

    episodes: int = 1500
    gamma: float = 0.99
    gae_lambda: float = 0.95    # GAE lambda，用于平衡偏差和方差
    learning_rate: float = 3e-4
    ppo_epochs: int = 4         # 每次rollout批量的更新轮数
    rollout_episodes: int = 16  # 每次PPO更新前累积的episode数量，增大batch size以稳定优势估计
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip_norm: float = 0.5  # 适当收紧梯度裁剪
    seed: int = 42
    device: str = "GPU"

    visualize_every_episodes: int = 50
    log_every_episodes: int = 10
    reward_curve_moving_window: int = 20


@dataclass
class RoomConfig:
    """Room-selection settings loaded from config.yaml."""

    json_path: str = "RL/test_room/origin_room/test_room.json"
    room_name: str = "办公室1"


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducible smoke tests."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load the RL training config from YAML."""
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level mapping in {config_path}")
    return payload


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Compute episodic discounted returns G_t."""
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and discounted returns for one episode.

    GAE(lambda) 以 lambda 在低方差（纯TD，lambda=0）和低偏差（蒙特卡洛，lambda=1）
    之间取得平衡，比纯蒙特卡洛return在短episode（4步）上方差更小、梯度更稳定。

    Args:
        rewards: 该episode的奖励序列
        values: 每步的V(s)估计，长度与rewards相同
        gamma: 折扣因子
        lam: GAE lambda
        last_value: 终止状态的V(s')，episode结束为0
    Returns:
        advantages: [T] GAE优势
        returns: [T] 用于value loss的目标returns（advantages + values）
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else last_value
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    adv_tensor = torch.tensor(advantages, dtype=torch.float32)
    val_tensor = torch.tensor(values, dtype=torch.float32)
    returns_tensor = adv_tensor + val_tensor
    return adv_tensor, returns_tensor


def should_visualize_episode(episode_idx: int, cfg: PPOConfig) -> bool:
    """Whether this episode should dump intermediate room-state images."""
    return episode_idx % cfg.visualize_every_episodes == 0


def export_episode_snapshot(
    env: SingleRoomLightingEnv,
    output_dir: Path,
    episode_idx: int,
    step_idx: int,
    *,
    suffix: str = "",
) -> None:
    """Save one room-state snapshot into the current run output directory."""
    name = f"episode_{episode_idx:04d}_step_{step_idx:03d}{suffix}.png"
    env.export_snapshot(output_dir / name)


def create_timestamped_output_dir(base_output_dir: Path) -> Path:
    """Create one timestamped subdirectory for the current training run."""
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = base_output_dir / timestamp
    suffix = 1
    while run_output_dir.exists():
        run_output_dir = base_output_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1

    run_output_dir.mkdir(parents=True, exist_ok=False)
    return run_output_dir


def plot_reward_curve(
    history: list[dict[str, Any]],
    output_path: Path,
    *,
    moving_window: int = 20,
) -> Path:
    """Plot episode reward and its moving average, then save to disk."""
    if not history:
        raise ValueError("History is empty. Cannot plot reward curve.")

    episodes = [int(item["episode"]) for item in history]
    rewards = np.asarray([float(item["episode_reward"]) for item in history], dtype=np.float32)

    window = max(1, min(int(moving_window), len(rewards)))
    kernel = np.ones(window, dtype=np.float32) / window
    moving = np.convolve(rewards, kernel, mode="valid")
    moving_episodes = episodes[window - 1 :]

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, color="#7aa6ff", linewidth=1.2, alpha=0.75, label="Episode Reward")
    plt.plot(moving_episodes, moving, color="#d94f30", linewidth=2.0, label=f"Moving Avg ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Single-Room Training Reward Curve")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_score_trends(history: list[dict[str, Any]], output_path: Path) -> Path:
    """Plot episode-level mean raw-score trends over the whole training run."""
    if not history:
        raise ValueError("History is empty. Cannot plot score trends.")

    episodes = [int(item["episode"]) for item in history]
    potential = np.asarray([float(item.get("mean_potential_normalized", 0.0)) for item in history], dtype=np.float32)
    alignment = np.asarray([float(item.get("alignment_normalized", 0.0)) for item in history], dtype=np.float32)
    wiring = np.asarray([float(item.get("wiring_normalized", 0.0)) for item in history], dtype=np.float32)
    # mst_cost = np.asarray([float(item.get("mst_cost", 0.0)) for item in history], dtype=np.float32)

    plt.figure(figsize=(11, 5.5))
    plt.plot(episodes, potential, label="mean_potential_normalized", linewidth=2.0)
    plt.plot(episodes, alignment, label="alignment_normalized", linewidth=2.0)
    plt.plot(episodes, wiring, label="wiring_normalized", linewidth=2.0)
    # plt.plot(episodes, mst_cost, label="mst_cost", linewidth=2.0)
    plt.xlabel("Episode")
    plt.ylabel("Score / Cost")
    plt.title("Episode-Level Mean Score Trends")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _collect_one_episode(
    env: SingleRoomLightingEnv,
    model: LightingActorCritic,
    device: torch.device,
    episode_idx: int,
    cfg: PPOConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """
    执行一个完整episode的rollout，返回轨迹数据和episode级统计信息。

    Returns 包含:
        obs_list, actions, rewards, log_probs, values（用于PPO更新）
        episode_reward, mean_potential_normalized, mean_potential_item,
        alignment_normalized, alignment_term, wiring_normalized, wiring_term,
        mst_cost, terminal_bonus（用于日志和JSON记录）
    """
    obs = env.reset()
    visualize_episode = should_visualize_episode(episode_idx, cfg)

    obs_list: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    log_probs: list[float] = []
    values: list[float] = []

    potential_normalized_list: list[float] = []
    potential_item_list: list[float] = []

    done = False
    while not done:
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
        rollout = model.act(obs_tensor, deterministic=False)
        action = int(rollout["action"].item())
        next_obs, reward, done, info = env.step(action)

        obs_list.append(obs)
        actions.append(action)
        rewards.append(float(reward))
        log_probs.append(float(rollout["log_prob"].item()))
        values.append(float(rollout["value"].squeeze().item()))
        obs = next_obs

        bd = info["step_breakdown"]
        potential_normalized_list.append(float(bd.potential_reduction_normalized))
        potential_item_list.append(float(bd.potential_reduction_item))

        if visualize_episode and (env.current_step % env.config.target_lamp_count) == 0:
            export_episode_snapshot(env, output_dir, episode_idx, env.current_step)

    final_bd = env.last_breakdown
    n = max(len(potential_normalized_list), 1)
    return {
        # PPO rollout data
        "obs_list": obs_list,
        "actions": actions,
        "rewards": rewards,
        "log_probs": log_probs,
        "values": values,
        # episode-level stats (averages / terminal values)
        "episode_reward": float(sum(rewards)),
        "mean_potential_normalized": sum(potential_normalized_list) / n,
        "mean_potential_item": sum(potential_item_list) / n,
        "alignment_normalized": float(final_bd.alignment_normalized) if final_bd else 0.0,
        "alignment_term": float(final_bd.alignment_term) if final_bd else 0.0,
        "wiring_normalized": float(final_bd.wiring_normalized) if final_bd else 0.0,
        "wiring_term": float(final_bd.wiring_term) if final_bd else 0.0,
        "mst_cost": float(final_bd.mst_cost) if final_bd else 0.0,
        "terminal_bonus": float(final_bd.terminal_bonus) if final_bd else 0.0,
    }


def train_single_room(
    env: SingleRoomLightingEnv,
    model: LightingActorCritic,
    cfg: PPOConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Run PPO on a single room and periodically dump room-state visualizations.

    核心改动：
    - 每 rollout_episodes 个 episode 收集一个批次后再做 PPO 更新，
      避免只用 4 步数据进行梯度更新导致的高方差问题。
    - 使用 GAE(lambda) 替代纯蒙特卡洛 return，降低优势估计方差。
    """
    device = torch.device(cfg.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history: list[dict[str, Any]] = []
    best_reward = float("-inf")
    best_model_path = output_dir / "ppo_single_room_best.pt"

    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0

    episode_idx = 0
    while episode_idx < cfg.episodes:
        # ---- Phase 1: 收集 rollout_episodes 个 episode 的轨迹 ----
        rollout_obs: list[np.ndarray] = []
        rollout_actions: list[int] = []
        rollout_advantages: list[float] = []
        rollout_returns: list[float] = []
        rollout_old_log_probs: list[float] = []

        # 本批次内所有episode的诊断统计，用于日志记录
        batch_ep_data: list[dict[str, Any]] = []

        episodes_this_batch = min(cfg.rollout_episodes, cfg.episodes - episode_idx)
        for _ in range(episodes_this_batch):
            episode_idx += 1
            ep = _collect_one_episode(env, model, device, episode_idx, cfg, output_dir)

            # GAE 计算：每个 episode 独立计算，终止状态 last_value=0
            adv, ret = compute_gae(
                ep["rewards"],
                ep["values"],
                cfg.gamma,
                cfg.gae_lambda,
                last_value=0.0,
            )
            rollout_obs.extend(ep["obs_list"])
            rollout_actions.extend(ep["actions"])
            rollout_advantages.extend(adv.tolist())
            rollout_returns.extend(ret.tolist())
            rollout_old_log_probs.extend(ep["log_probs"])
            batch_ep_data.append(ep)

        # ---- Phase 2: 对本批次整体做 PPO 更新 ----
        batch_obs_t = torch.from_numpy(np.stack(rollout_obs)).to(device=device, dtype=torch.float32)
        batch_actions_t = torch.tensor(rollout_actions, dtype=torch.long, device=device)
        batch_old_log_probs_t = torch.tensor(rollout_old_log_probs, dtype=torch.float32, device=device)
        batch_returns_t = torch.tensor(rollout_returns, dtype=torch.float32, device=device)

        # 对整个批次的优势做归一化，稳定训练
        adv_t = torch.tensor(rollout_advantages, dtype=torch.float32, device=device)
        if adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        for _ in range(cfg.ppo_epochs):
            evaluated = model.evaluate_actions(batch_obs_t, batch_actions_t)
            log_probs_t = evaluated["log_prob"]
            entropy_t = evaluated["entropy"]
            values_t = evaluated["value"].squeeze(-1)

            ratio = torch.exp(log_probs_t - batch_old_log_probs_t)
            surrogate1 = ratio * adv_t
            surrogate2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_t
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = nn.functional.mse_loss(values_t, batch_returns_t)
            entropy_bonus = entropy_t.mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_entropy = float(entropy_bonus.item())

        # ---- Phase 3: 记录本批次每个 episode 的统计 ----
        for ep in batch_ep_data:
            ep_reward = ep["episode_reward"]
            history.append(
                {
                    "episode": len(history) + 1,
                    "episode_reward": ep_reward,
                    "mean_potential_normalized": ep["mean_potential_normalized"],
                    "mean_potential_item": ep["mean_potential_item"],
                    "alignment_normalized": ep["alignment_normalized"],
                    "alignment_term": ep["alignment_term"],
                    "wiring_normalized": ep["wiring_normalized"],
                    "wiring_term": ep["wiring_term"],
                    "mst_cost": ep["mst_cost"],
                    "terminal_bonus": ep["terminal_bonus"],
                    "policy_loss": last_policy_loss,
                    "value_loss": last_value_loss,
                    "entropy": last_entropy,
                }
            )

            if ep_reward > best_reward:
                best_reward = ep_reward
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "episode": episode_idx,
                        "episode_reward": ep_reward,
                        "ppo_config": asdict(cfg),
                    },
                    best_model_path,
                )

        # ---- Phase 4: 日志输出（以批次末尾episode为代表） ----
        last_ep = batch_ep_data[-1]
        if episode_idx % cfg.log_every_episodes == 0 or episode_idx == cfg.episodes:
            recent = history[-cfg.log_every_episodes:]
            moving_reward = sum(item["episode_reward"] for item in recent) / len(recent)
            print(
                f"[train] ep={episode_idx:04d} "
                f"reward={last_ep['episode_reward']:7.3f} "
                f"moving={moving_reward:7.3f} "
                f"pot_norm={last_ep['mean_potential_normalized']:.3f} "
                f"pot_item={last_ep['mean_potential_item']:.3f} "
                f"align={last_ep['alignment_normalized']:.3f}*{last_ep['alignment_term']:.3f} "
                f"wire={last_ep['wiring_normalized']:.3f}*{last_ep['wiring_term']:.3f} "
                f"mst={last_ep['mst_cost']:6.2f} "
                f"bonus={last_ep['terminal_bonus']:.3f} "
                f"ploss={last_policy_loss:.4f} vloss={last_value_loss:.4f}"
            )

    summary = {
        "ppo_config": asdict(cfg),
        "reward_config": asdict(env.config.reward_config),
        "room_name": env.room_name,
        "best_reward": best_reward,
        "best_model_path": str(best_model_path),
        "history": history,
    }
    return summary


def evaluate_greedy_policy(
    env: SingleRoomLightingEnv,
    model: LightingActorCritic,
    output_dir: Path,
    *,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run one greedy rollout with the trained model and export every room state."""
    obs = env.reset()
    export_episode_snapshot(env, output_dir, env.episode_index, 0, suffix="_greedy")

    device = next(model.parameters()).device
    done = False
    rewards: list[float] = []
    while not done:
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
        rollout = model.act(obs_tensor, deterministic=True)
        action = int(rollout["action"].item())
        obs, reward, done, _ = env.step(action)
        rewards.append(float(reward))
        export_episode_snapshot(env, output_dir, env.episode_index, env.current_step, suffix="_greedy")
        if max_steps is not None and env.current_step >= max_steps:
            break

    final_breakdown = env.last_breakdown
    return {
        "greedy_reward": float(sum(rewards)),
        "mst_cost": float(final_breakdown.mst_cost) if final_breakdown else 0.0,
        "alignment_term": float(final_breakdown.alignment_term) if final_breakdown else 0.0,
        "wiring_term": float(final_breakdown.wiring_term) if final_breakdown else 0.0,
    }


def load_room_dataset(room_dir: Path) -> list[dict]:
    """Load all room JSON files from a directory."""
    rooms = []
    for json_file in sorted(room_dir.glob("*.json")):
        with json_file.open('r', encoding='utf-8') as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            continue
        for room_data in payload.values():
            if isinstance(room_data, dict) and 'matrix' in room_data and 'lamp_count' in room_data:
                rooms.append(room_data)
    print(f"[train] Loaded {len(rooms)} rooms from {room_dir}")
    return rooms


def train_multi_room(
    room_dataset: list[dict],
    model: LightingActorCritic,
    ppo_cfg: PPOConfig,
    env_cfg: EnvironmentConfig,
    output_dir: Path,
) -> dict:
    """Train PPO model on multiple rooms with random sampling."""
    device = torch.device(ppo_cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate)

    history = []
    best_reward = float("-inf")
    best_episode = 0

    rollout_buffer = []

    for episode in range(1, ppo_cfg.episodes + 1):
        # Randomly select a room for this episode
        room_data = random.choice(room_dataset)

        # Create environment for this room
        # Update target_lamp_count from room data
        current_env_cfg = EnvironmentConfig(
            padded_size=env_cfg.padded_size,
            max_steps=env_cfg.max_steps,
            target_lamp_count=room_data['lamp_count'],
            turn_penalty=env_cfg.turn_penalty,
            reward_config=env_cfg.reward_config,
        )
        current_env_cfg.reward_config.target_lamp_count = room_data['lamp_count']

        # Create temporary environment
        env = SingleRoomLightingEnv(room_data, config=current_env_cfg)

        # Run one episode
        obs = env.reset()
        done = False
        episode_rewards = []
        episode_states = []
        episode_actions = []
        episode_log_probs = []
        episode_values = []

        while not done:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
            rollout = model.act(obs_tensor, deterministic=False)

            action = int(rollout["action"].item())
            log_prob = float(rollout["log_prob"].item())
            value = float(rollout["value"].item())

            episode_states.append(obs)
            episode_actions.append(action)
            episode_log_probs.append(log_prob)
            episode_values.append(value)

            obs, reward, done, _ = env.step(action)
            episode_rewards.append(float(reward))

        # Compute advantages and returns
        advantages, returns = compute_gae(
            episode_rewards,
            episode_values,
            ppo_cfg.gamma,
            ppo_cfg.gae_lambda,
            last_value=0.0,
        )

        # Add to rollout buffer
        rollout_buffer.append({
            "states": episode_states,
            "actions": episode_actions,
            "log_probs": episode_log_probs,
            "advantages": advantages,
            "returns": returns,
            "room_name": room_data['room_name'],
        })

        episode_reward = sum(episode_rewards)

        # Record history
        final_breakdown = env.last_breakdown
        history.append({
            "episode": episode,
            "episode_reward": episode_reward,
            "room_name": room_data['room_name'],
            "room_type": room_data.get('room_type', 'unknown'),
            "alignment_normalized": float(final_breakdown.alignment_normalized) if final_breakdown else 0.0,
            "wiring_normalized": float(final_breakdown.wiring_normalized) if final_breakdown else 0.0,
            "mst_cost": float(final_breakdown.mst_cost) if final_breakdown else 0.0,
        })

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_episode = episode
            torch.save(model.state_dict(), output_dir / "ppo_multi_room_best.pt")

        # PPO update when buffer is full
        if len(rollout_buffer) >= ppo_cfg.rollout_episodes:
            ppo_update_multi_room(model, optimizer, rollout_buffer, ppo_cfg, device)
            rollout_buffer.clear()

        # Logging
        if episode % ppo_cfg.log_every_episodes == 0:
            recent = history[-ppo_cfg.log_every_episodes:]
            avg_reward = sum(h["episode_reward"] for h in recent) / len(recent)
            avg_align = sum(h["alignment_normalized"] for h in recent) / len(recent)
            print(f"[train] episode={episode:04d} avg_reward={avg_reward:7.2f} "
                  f"avg_align={avg_align:.3f} best={best_reward:7.2f}")

    return {
        "history": history,
        "best_reward": best_reward,
        "best_episode": best_episode,
        "total_episodes": ppo_cfg.episodes,
    }


def ppo_update_multi_room(
    model: LightingActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout_buffer: list[dict],
    ppo_cfg: PPOConfig,
    device: torch.device,
) -> None:
    """PPO update using collected rollouts from multiple rooms."""
    # Flatten all rollouts
    all_states = []
    all_actions = []
    all_old_log_probs = []
    all_advantages = []
    all_returns = []

    for rollout in rollout_buffer:
        all_states.extend(rollout["states"])
        all_actions.extend(rollout["actions"])
        all_old_log_probs.extend(rollout["log_probs"])
        all_advantages.extend(rollout["advantages"].tolist())
        all_returns.extend(rollout["returns"].tolist())

    # Convert to tensors
    states_tensor = torch.from_numpy(np.array(all_states)).to(device=device, dtype=torch.float32)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long, device=device)
    old_log_probs_tensor = torch.tensor(all_old_log_probs, dtype=torch.float32, device=device)
    advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32, device=device)
    returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=device)

    # Normalize advantages
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

    # PPO epochs
    for _ in range(ppo_cfg.ppo_epochs):
        eval_result = model.evaluate_actions(states_tensor, actions_tensor)
        new_log_probs = eval_result["log_prob"]
        entropy = eval_result["entropy"]
        values = eval_result["value"].squeeze(-1)

        # Policy loss
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1.0 - ppo_cfg.clip_eps, 1.0 + ppo_cfg.clip_eps) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = ((values - returns_tensor) ** 2).mean()

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Total loss
        loss = policy_loss + ppo_cfg.value_coef * value_loss + ppo_cfg.entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.grad_clip_norm)
        optimizer.step()


def evaluate_all_rooms(
    room_dataset: list[dict],
    model: LightingActorCritic,
    env_cfg: EnvironmentConfig,
    results_dir: Path,
) -> list[dict]:
    """Greedy rollout on every room, save layout image to results_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    results = []
    for i, room_data in enumerate(room_dataset):
        lamp_count = room_data['lamp_count']
        current_cfg = EnvironmentConfig(
            padded_size=env_cfg.padded_size,
            max_steps=env_cfg.max_steps,
            target_lamp_count=lamp_count,
            turn_penalty=env_cfg.turn_penalty,
            reward_config=env_cfg.reward_config,
        )
        current_cfg.reward_config.target_lamp_count = lamp_count
        env = SingleRoomLightingEnv(room_data, config=current_cfg)

        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
            action = int(model.act(obs_t, deterministic=True)["action"].item())
            obs, reward, done, _ = env.step(action)
            total_reward += float(reward)

        final_bd = env.last_breakdown
        room_name = room_data.get('room_name', f'room_{i:04d}')
        title = (
            f"{room_name} | r={total_reward:.2f} "
            f"p={final_bd.potential_reduction_normalized:.2f} "
            f"a={final_bd.alignment_normalized:.2f} "
            f"w={final_bd.wiring_normalized:.2f}"
        ) if final_bd else f"{room_name} | r={total_reward:.2f}"
        img_path = results_dir / f"room_{i:04d}_{lamp_count}lamp.png"
        save_room_grid_image(env.current_encoded_matrix(), img_path, cell_size=16, room_name=title)

        results.append({
            "index": i,
            "room_name": room_name,
            "lamp_count": lamp_count,
            "total_reward": total_reward,
            "potential_normalized": float(final_bd.potential_reduction_normalized) if final_bd else 0.0,
            "alignment_normalized": float(final_bd.alignment_normalized) if final_bd else 0.0,
            "wiring_normalized": float(final_bd.wiring_normalized) if final_bd else 0.0,
        })

    avg_reward = sum(r["total_reward"] for r in results) / max(len(results), 1)
    avg_align = sum(r["alignment_normalized"] for r in results) / max(len(results), 1)
    avg_wire = sum(r["wiring_normalized"] for r in results) / max(len(results), 1)
    print(f"[eval] {len(results)} rooms | avg_reward={avg_reward:.3f} avg_align={avg_align:.3f} avg_wire={avg_wire:.3f}")
    print(f"[eval] results saved to {results_dir}")
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Train PPO model on room layout tasks')
    parser.add_argument('--room_dir', type=str, default=None, help='Directory containing room JSON files for multi-room training')
    parser.add_argument('--episodes', type=int, default=None, help='Override number of training episodes')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "RL" / "config.yaml"
    config_payload = load_yaml_config(config_path)

    output_root_dir = repo_root / "RL" / "output"
    output_dir = create_timestamped_output_dir(output_root_dir)
    shutil.copy2(config_path, output_dir / "config.yaml")

    ppo_cfg = PPOConfig(**config_payload.get("training", {}))
    if args.episodes:
        ppo_cfg.episodes = args.episodes
    set_seed(ppo_cfg.seed)

    reward_cfg = RewardConfig(**config_payload.get("reward", {}))
    room_cfg = RoomConfig(**config_payload.get("room", {}))
    env_cfg = EnvironmentConfig(
        **config_payload.get("environment", {}),
        reward_config=reward_cfg,
    )

    # Check if multi-room training is requested
    if args.room_dir:
        # Multi-room training mode
        room_dataset = load_room_dataset(Path(args.room_dir))

        if not room_dataset:
            raise ValueError(f"No rooms found in {args.room_dir}")

        # Use dynamic target_lamp_count (will be set per room)
        model = LightingActorCritic(target_lamp_count=None)

        print(f"[train] Starting multi-room training with {len(room_dataset)} rooms")
        summary = train_multi_room(room_dataset, model, ppo_cfg, env_cfg, output_dir)
        summary["training_mode"] = "multi_room"
        summary["num_rooms"] = len(room_dataset)

        # Evaluate on all rooms
        results_dir = Path(args.room_dir).parent / "results"
        print(f"[train] Evaluating on all {len(room_dataset)} rooms...")
        eval_results = evaluate_all_rooms(room_dataset, model, env_cfg, results_dir)
        summary["all_room_evaluations"] = eval_results
        summary_filename = "ppo_multi_room_training_summary.json"
    else:
        # Single-room training mode (original behavior)
        reward_cfg.target_lamp_count = env_cfg.target_lamp_count

        env = SingleRoomLightingEnv.from_json(
            repo_root / room_cfg.json_path,
            room_name=room_cfg.room_name,
            config=env_cfg,
        )

        # Save padded room visualization
        padded_room_path = output_dir / "padded_room.png"
        save_padded_room_image(
            env.original_matrix,
            env.padded_size,
            padded_room_path,
            cell_size=32,
            room_name=env.room_name,
        )
        print(f"[train] padded room visualization saved to {padded_room_path}")

        model = LightingActorCritic(target_lamp_count=env_cfg.target_lamp_count)

        summary = train_single_room(env, model, ppo_cfg, output_dir)
        summary["greedy_evaluation"] = evaluate_greedy_policy(env, model, output_dir)
        summary["training_mode"] = "single_room"
        summary_filename = "ppo_single_room_training_summary.json"

    summary["output_dir"] = str(output_dir)
    summary["config_path"] = str(config_path)

    reward_curve_path = output_dir / "reward_curve.png"
    plot_reward_curve(summary["history"], reward_curve_path, moving_window=ppo_cfg.reward_curve_moving_window)
    summary["reward_curve_path"] = str(reward_curve_path)

    score_trend_path = output_dir / "score_trends.png"
    plot_score_trends(summary["history"], score_trend_path)
    summary["score_trend_path"] = str(score_trend_path)

    summary_path = output_dir / summary_filename
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[train] run output dir: {output_dir}")
    print(f"[train] reward curve saved to {reward_curve_path}")
    print(f"[train] score trends saved to {score_trend_path}")
    print(f"[train] summary saved to {summary_path}")
    print(f"[train] best_reward={summary['best_reward']:.3f}")


if __name__ == "__main__":
    main()
