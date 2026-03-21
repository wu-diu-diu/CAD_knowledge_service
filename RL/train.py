from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch import nn

from env import EnvironmentConfig, SingleRoomLightingEnv
from model import LightingActorCritic
from reward import RewardConfig


matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class PPOConfig:
    """Training hyper-parameters for the single-room PPO smoke test."""

    episodes: int = 1500  ## 120表示训练120轮，每轮代表一次完整的布局过程直到done=True
    gamma: float = 0.99  ## 折扣因子，接近1表示更重视长期奖励
    learning_rate: float = 3e-4  ## PPO的学习率
    ppo_epochs: int = 6  ## 运行所有episodes后，使用收集到的轨迹数据进行多少轮的PPO更新，每轮都会从轨迹数据中拿出一个不同的batch，去计算loss
    clip_eps: float = 0.2 ## PPO的剪切范围，限制新旧策略的变化幅度 新旧策略的比值被限制在[1-clip_eps, 1+clip_eps]，即新策略不能比旧策略更好或更差超过20%
    value_coef: float = 0.5  ## critic损失的权重 总损失等于 policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
    entropy_coef: float = 0.01  ## 鼓励策略多样性的熵奖励权重
    grad_clip_norm: float = 1.0  ## 梯度裁剪的最大范数
    seed: int = 42
    device: str = "cpu"

    visualize_every_episodes: int = 50  ## 每隔多少轮导出一次房间状态图片
    visualize_step_interval: int = 4  ## 在可视化的轮次中，每隔多少步骤导出一次房间状态图片
    log_every_episodes: int = 10  ## 每隔多少轮打印一次训练日志


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducible smoke tests."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Compute episodic discounted returns G_t."""
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


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
    """Save one room-state snapshot to RL/output."""
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


def train_single_room(
    env: SingleRoomLightingEnv,
    model: LightingActorCritic,
    cfg: PPOConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Run PPO on a single room and periodically dump room-state visualizations.

    The goal here is not benchmark-grade convergence, but a first end-to-end
    smoke test showing:
        - the environment can be rolled out
        - PPO updates run without shape issues
        - room states evolve and are exported to RL/output
    """
    device = torch.device(cfg.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history: list[dict[str, Any]] = []
    best_reward = float("-inf")
    best_model_path = output_dir / "ppo_single_room_best.pt"

    for episode_idx in range(1, cfg.episodes + 1):
        obs = env.reset()
        visualize_episode = should_visualize_episode(episode_idx, cfg)
        if visualize_episode:
            export_episode_snapshot(env, output_dir, episode_idx, 0)

        obs_buffer: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        old_log_probs: list[float] = []
        old_values: list[float] = []

        done = False
        while not done:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
            rollout = model.act(obs_tensor, deterministic=False)
            action = int(rollout["action"].item())
            next_obs, reward, done, info = env.step(action)

            obs_buffer.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            old_log_probs.append(float(rollout["log_prob"].item()))  ## 记录当前动作的对数概率，这个值在PPO更新时会用来计算新旧策略的比值，从而决定是否剪切
            old_values.append(float(rollout["value"].squeeze().item()))

            obs = next_obs

            if visualize_episode and (env.current_step % cfg.visualize_step_interval == 0):
                export_episode_snapshot(env, output_dir, episode_idx, env.current_step)

        returns = discounted_returns(rewards, cfg.gamma).to(device)
        old_values_tensor = torch.tensor(old_values, dtype=torch.float32, device=device)
        advantages = returns - old_values_tensor
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)  ## 对优势函数进行归一化处理，使其具有零均值和单位方差，这有助于稳定训练过程，避免过大或过小的优势值导致梯度更新不稳定。

        batch_obs = torch.from_numpy(np.stack(obs_buffer)).to(device=device, dtype=torch.float32)
        batch_actions = torch.tensor(actions, dtype=torch.long, device=device)
        batch_old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=device)

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        for _ in range(cfg.ppo_epochs):
            evaluated = model.evaluate_actions(batch_obs, batch_actions)  ## evaluate_actions方法会计算当前策略在给定状态和动作上的对数概率、熵和状态值，这些信息将用于计算PPO的损失函数，包括策略损失、值函数损失和熵奖励。
            log_probs = evaluated["log_prob"]
            entropy = evaluated["entropy"]
            values = evaluated["value"].squeeze(-1)

            ratio = torch.exp(log_probs - batch_old_log_probs)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = nn.functional.mse_loss(values, returns)
            entropy_bonus = entropy.mean()
            ## loss由三部分组成：策略损失、值函数损失和熵奖励。策略损失是PPO的核心，使用了剪切的优势函数来限制新旧策略的变化幅度；值函数损失是均方误差，衡量当前状态值估计与实际回报之间的差距；熵奖励鼓励策略保持多样性，避免过早收敛到次优解。通过调整value_coef和entropy_coef，可以平衡这三部分在总损失中的贡献，从而影响训练的稳定性和最终性能。
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_entropy = float(entropy_bonus.item())

        episode_reward = float(sum(rewards))  ## 一个episode的总奖励是所有步骤奖励的累积和，这个值可以用来评估当前策略在这个房间上的表现，通常希望随着训练的进行，episode_reward能够逐渐增加，表明智能体学会了更有效的布置策略。
        final_breakdown = env.last_breakdown  ## 
        lamp_count = int(final_breakdown.diagnostics["lamp_count"]) if final_breakdown else 0
        mst_cost = float(final_breakdown.diagnostics["mst_cost"]) if final_breakdown else 0.0
        alignment_score = float(final_breakdown.diagnostics.get("alignment_score", 0.0)) if final_breakdown else 0.0
        centering_score = float(final_breakdown.diagnostics.get("centering_score", 0.0)) if final_breakdown else 0.0

        history.append(
            {
                "episode": episode_idx,
                "episode_reward": episode_reward,
                "lamp_count": lamp_count,
                "mst_cost": mst_cost,
                "alignment_score": alignment_score,
                "centering_score": centering_score,
                "policy_loss": last_policy_loss,
                "value_loss": last_value_loss,
                "entropy": last_entropy,
            }
        )

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "episode": episode_idx,
                    "episode_reward": episode_reward,
                    "ppo_config": asdict(cfg),
                },
                best_model_path,
            )
        ## 每隔cfg.log_every_episodes轮，或者在第一轮和最后一轮时，打印一次训练日志，包括当前轮数、当前轮奖励、最近几轮的平均奖励、灯具数量、对齐分数和MST成本等信息，这些信息可以帮助我们监控训练过程，了解智能体的学习进展和策略改进情况。
        if episode_idx % cfg.log_every_episodes == 0 or episode_idx == 1 or episode_idx == cfg.episodes:
            recent = history[-cfg.log_every_episodes :]
            moving_reward = sum(item["episode_reward"] for item in recent) / len(recent)  ## 计算最近几轮的平均奖励，这个值可以更平滑地反映训练趋势，避免单轮奖励的波动过大导致误导。
            print(
                f"[train] episode={episode_idx:04d} "
                f"reward={episode_reward:8.3f} "
                f"moving_reward={moving_reward:8.3f} "
                f"lamps={lamp_count:02d} align={alignment_score:5.2f} center={centering_score:5.2f} "
                f"mst_cost={mst_cost:7.2f}"
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
    export_episode_snapshot(env, output_dir, env.episode_index, 0, suffix="_greedy")  ## 

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
        "lamp_count": int(final_breakdown.diagnostics["lamp_count"]) if final_breakdown else 0,
        "mst_cost": float(final_breakdown.diagnostics["mst_cost"]) if final_breakdown else 0.0,
        "alignment_score": float(final_breakdown.diagnostics.get("alignment_score", 0.0)) if final_breakdown else 0.0,
        "centering_score": float(final_breakdown.diagnostics.get("centering_score", 0.0)) if final_breakdown else 0.0,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root_dir = repo_root / "RL" / "output"
    output_dir = create_timestamped_output_dir(output_root_dir)

    ppo_cfg = PPOConfig()
    set_seed(ppo_cfg.seed)

    reward_cfg = RewardConfig()
    env_cfg = EnvironmentConfig(
        padded_size=32,
        max_steps=16,
        target_lamp_count=4,
        turn_penalty=0.2,
        reward_config=reward_cfg,
    )

    env = SingleRoomLightingEnv.from_json(
        repo_root / "RL" / "test_room" / "test_room.json",
        room_name="办公室1",
        config=env_cfg,
    )
    model = LightingActorCritic(target_lamp_count=env_cfg.target_lamp_count)

    summary = train_single_room(env, model, ppo_cfg, output_dir)
    summary["greedy_evaluation"] = evaluate_greedy_policy(env, model, output_dir)
    summary["output_dir"] = str(output_dir)

    reward_curve_path = output_dir / "reward_curve.png"
    plot_reward_curve(summary["history"], reward_curve_path, moving_window=ppo_cfg.log_every_episodes)
    summary["reward_curve_path"] = str(reward_curve_path)

    summary_path = output_dir / "ppo_single_room_training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[train] run output dir: {output_dir}")
    print(f"[train] reward curve saved to {reward_curve_path}")
    print(f"[train] summary saved to {summary_path}")
    print(f"[train] best_reward={summary['best_reward']:.3f}")
    print(f"[train] greedy_eval={summary['greedy_evaluation']}")


if __name__ == "__main__":
    main()
