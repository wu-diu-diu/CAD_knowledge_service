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
from visualize import plot_episode_step_breakdown

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class PPOConfig:
    """Training hyper-parameters for the single-room PPO smoke test."""

    episodes: int = 1500
    gamma: float = 0.99
    learning_rate: float = 3e-4
    ppo_epochs: int = 6  ## 这个数表示，得到一批轨迹之后，要重复拿来更新策略多少轮
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip_norm: float = 1.0
    seed: int = 42
    device: str = "cpu"

    visualize_every_episodes: int = 50
    log_every_episodes: int = 10
    reward_curve_moving_window: int = 20


@dataclass
class RoomConfig:
    """Room-selection settings loaded from config.yaml."""

    json_path: str = "RL/test_room/test_room.json"
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
    uniformity = np.asarray([float(item["mean_uniformity_score"]) for item in history], dtype=np.float32)
    illum_centroid = np.asarray([float(item["mean_illum_centroid_score"]) for item in history], dtype=np.float32)
    alignment = np.asarray([float(item["mean_alignment_score"]) for item in history], dtype=np.float32)
    wiring = np.asarray([float(item["mean_wiring_score"]) for item in history], dtype=np.float32)

    plt.figure(figsize=(11, 5.5))
    plt.plot(episodes, uniformity, label="mean_uniformity_score", linewidth=2.0)
    plt.plot(episodes, illum_centroid, label="mean_illum_centroid_score", linewidth=2.0)
    plt.plot(episodes, alignment, label="mean_alignment_score", linewidth=2.0)
    plt.plot(episodes, wiring, label="mean_wiring_score", linewidth=2.0)
    plt.xlabel("Episode")
    plt.ylabel("Mean Raw Score")
    plt.title("Episode-Level Mean Score Trends")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(loc="best")
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

    Visualized episodes save:
        - every step snapshot
        - one score breakdown figure for the whole episode
    """
    device = torch.device(cfg.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history: list[dict[str, Any]] = []
    best_reward = float("-inf")
    best_model_path = output_dir / "ppo_single_room_best.pt"

    for episode_idx in range(1, cfg.episodes + 1):
        obs = env.reset()
        ## 布尔变量，当episode的idx和cfg中设置的visualize_every_episodes的间隔条件满足时为True，表示这个episode需要进行可视化处理。在这个训练循环中，每当visualize_episode为True时，代码会在每一步结束后调用export_episode_snapshot函数
        visualize_episode = should_visualize_episode(episode_idx, cfg)
        episode_step_records: list[dict[str, float | int | bool]] = []
        # if visualize_episode:
        #     export_episode_snapshot(env, output_dir, episode_idx, 0)

        obs_buffer: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        old_log_probs: list[float] = []
        old_values: list[float] = []
        uniformity_scores_episode: list[float] = []
        illum_centroid_scores_episode: list[float] = []
        alignment_scores_episode: list[float] = []
        wiring_scores_episode: list[float] = []

        done = False
        while not done:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
            rollout = model.act(obs_tensor, deterministic=False)
            action = int(rollout["action"].item())
            next_obs, reward, done, info = env.step(action)

            obs_buffer.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            old_log_probs.append(float(rollout["log_prob"].item()))
            old_values.append(float(rollout["value"].squeeze().item()))

            obs = next_obs

            breakdown = info["reward_breakdown"]
            diagnostics = breakdown.diagnostics
            uniformity_scores_episode.append(float(breakdown.uniformity))
            illum_centroid_scores_episode.append(float(breakdown.illum_centroid))
            alignment_scores_episode.append(float(breakdown.rules - breakdown.invalid_action))
            wiring_scores_episode.append(float(breakdown.wiring))

            if visualize_episode and env.current_step % env.config.target_lamp_count == 0:  ## 每隔visualize_episode的间隔隔episode才会可视化，而且只可视化该episode的最后一步绘制完的结果
                episode_step_records.append(
                    {
                        "step": env.current_step,
                        "uniformity_score": float(diagnostics.get("uniformity_score", 0.0)),  ## 计算出来的照度均匀的奖励分数
                        "illum_centroid_score": float(diagnostics.get("illum_centroid_score", 0.0)),
                        "alignment_score": float(diagnostics.get("alignment_score", 0.0)),
                        "wiring_score": float(diagnostics.get("wiring_score", 0.0)),
                        "invalid_penalty": float(diagnostics.get("invalid_penalty", 0.0)),
                        "uniformity_term": float(breakdown.uniformity),  ## 乘以权重之后的照度均匀奖励的分数
                        "illum_centroid_term": float(breakdown.illum_centroid),
                        "alignment_term": float(breakdown.rules - breakdown.invalid_action),
                        "wiring_term": float(breakdown.wiring),
                        "step_total": float(reward),
                    }
                )
                export_episode_snapshot(env, output_dir, episode_idx, env.current_step)

        returns = discounted_returns(rewards, cfg.gamma).to(device)
        old_values_tensor = torch.tensor(old_values, dtype=torch.float32, device=device)
        advantages = returns - old_values_tensor
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_obs = torch.from_numpy(np.stack(obs_buffer)).to(device=device, dtype=torch.float32)
        batch_actions = torch.tensor(actions, dtype=torch.long, device=device)
        batch_old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=device)

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        ## 实际模型更新的地方，在每个episode结束后进行多轮PPO优化，每轮都使用当前episode的完整轨迹数据来计算损失并更新模型参数。每轮优化中，首先通过模型评估当前批次的观察和动作，得到新的log_prob、entropy和value，然后计算PPO的剪切目标函数(policy_loss)以及值函数的均方误差(value_loss)，再加上熵奖励(entropy_bonus)来构成总损失(loss)。最后进行反向传播和梯度裁剪，并更新模型参数。这个过程会重复cfg.ppo_epochs次，以充分利用当前episode的数据来优化模型。
        for _ in range(cfg.ppo_epochs): 
            # 使用更新后的模型输入旧轨迹中的状态，得到当前策略的 logits，并构造策略分布。
            # 然后用旧轨迹中实际执行过的动作 a_t 计算 log_prob，得到这些旧动作在当前策略下的概率。
            # 其目的是和旧策略下这些动作的概率做比较，计算 ratio = π_new(a_t|s_t) / π_old(a_t|s_t)。

            # 如果 ratio > 1，说明当前策略相比旧策略更倾向于选择这个旧动作；
            # 如果该动作的优势 A_t > 0，说明这个动作是有利的，因此训练会倾向于进一步提高它的概率；
            # 如果 A_t < 0，则训练会倾向于降低它的概率。

            # PPO 实际优化的是剪切后的 surrogate objective，
            # 并通过最小化 policy_loss = -mean(min(surrogate1, surrogate2))
            # 来实现“增加有优势动作的概率、减少劣势动作的概率，同时限制策略更新幅度”。
            evaluated = model.evaluate_actions(batch_obs, batch_actions)
            log_probs = evaluated["log_prob"]
            entropy = evaluated["entropy"]
            values = evaluated["value"].squeeze(-1)

            ratio = torch.exp(log_probs - batch_old_log_probs)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = nn.functional.mse_loss(values, returns)
            entropy_bonus = entropy.mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_entropy = float(entropy_bonus.item())

        episode_reward = float(sum(rewards))
        final_breakdown = env.last_breakdown
        mst_cost = float(final_breakdown.diagnostics["mst_cost"]) if final_breakdown else 0.0
        uniformity_score = float(final_breakdown.diagnostics.get("uniformity_score", 0.0)) if final_breakdown else 0.0
        illum_centroid_score = (
            float(final_breakdown.diagnostics.get("illum_centroid_score", 0.0)) if final_breakdown else 0.0
        )
        alignment_score = float(final_breakdown.diagnostics.get("alignment_score", 0.0)) if final_breakdown else 0.0
        wiring_score = float(final_breakdown.diagnostics.get("wiring_score", 0.0)) if final_breakdown else 0.0
        mean_uniformity_score = float(np.mean(uniformity_scores_episode)) if uniformity_scores_episode else 0.0
        mean_illum_centroid_score = (
            float(np.mean(illum_centroid_scores_episode)) if illum_centroid_scores_episode else 0.0
        )
        mean_alignment_score = float(np.mean(alignment_scores_episode)) if alignment_scores_episode else 0.0
        mean_wiring_score = float(np.mean(wiring_scores_episode)) if wiring_scores_episode else 0.0

        history.append(
            {
                "episode": episode_idx,
                "episode_reward": episode_reward,
                "mst_cost": mst_cost,
                "uniformity_score": uniformity_score,
                "illum_centroid_score": illum_centroid_score,
                "alignment_score": alignment_score,
                "wiring_score": wiring_score,
                "mean_uniformity_score": mean_uniformity_score,
                "mean_illum_centroid_score": mean_illum_centroid_score,
                "mean_alignment_score": mean_alignment_score,
                "mean_wiring_score": mean_wiring_score,
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

        if episode_idx % cfg.log_every_episodes == 0 or episode_idx == 1 or episode_idx == cfg.episodes:
            recent = history[-cfg.log_every_episodes :]
            moving_reward = sum(item["episode_reward"] for item in recent) / len(recent)
            print(
                f"[train] episode={episode_idx:04d} "
                f"reward={episode_reward:8.3f} "
                f"moving_reward={moving_reward:8.3f} "
                f"uniform={uniformity_score:5.2f} "
                f"center={illum_centroid_score:5.2f} align={alignment_score:5.2f} "
                f"wire={wiring_score:5.2f}"
            )

        # if visualize_episode and episode_step_records:
        #     breakdown_path = output_dir / f"episode_{episode_idx:04d}_score_breakdown.png"
        #     plot_episode_step_breakdown(episode_step_records, breakdown_path, episode_idx=episode_idx)

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
        "lamp_count": int(final_breakdown.diagnostics["lamp_count"]) if final_breakdown else 0,
        "mst_cost": float(final_breakdown.diagnostics["mst_cost"]) if final_breakdown else 0.0,
        "uniformity_score": float(final_breakdown.diagnostics.get("uniformity_score", 0.0)) if final_breakdown else 0.0,
        "illum_centroid_score": (
            float(final_breakdown.diagnostics.get("illum_centroid_score", 0.0)) if final_breakdown else 0.0
        ),
        "alignment_score": float(final_breakdown.diagnostics.get("alignment_score", 0.0)) if final_breakdown else 0.0,
        "wiring_score": float(final_breakdown.diagnostics.get("wiring_score", 0.0)) if final_breakdown else 0.0,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "RL" / "config.yaml"
    config_payload = load_yaml_config(config_path)

    output_root_dir = repo_root / "RL" / "output"
    output_dir = create_timestamped_output_dir(output_root_dir)
    shutil.copy2(config_path, output_dir / "config.yaml")

    ppo_cfg = PPOConfig(**config_payload.get("training", {}))
    set_seed(ppo_cfg.seed)

    reward_cfg = RewardConfig(**config_payload.get("reward", {}))
    room_cfg = RoomConfig(**config_payload.get("room", {}))
    env_cfg = EnvironmentConfig(
        **config_payload.get("environment", {}),
        reward_config=reward_cfg,
    )
    reward_cfg.target_lamp_count = env_cfg.target_lamp_count

    env = SingleRoomLightingEnv.from_json(
        repo_root / room_cfg.json_path,
        room_name=room_cfg.room_name,
        config=env_cfg,
    )
    model = LightingActorCritic(target_lamp_count=env_cfg.target_lamp_count)

    summary = train_single_room(env, model, ppo_cfg, output_dir)
    summary["greedy_evaluation"] = evaluate_greedy_policy(env, model, output_dir)
    summary["output_dir"] = str(output_dir)
    summary["config_path"] = str(config_path)

    reward_curve_path = output_dir / "reward_curve.png"
    plot_reward_curve(summary["history"], reward_curve_path, moving_window=ppo_cfg.reward_curve_moving_window)
    summary["reward_curve_path"] = str(reward_curve_path)

    score_trend_path = output_dir / "score_trends.png"
    plot_score_trends(summary["history"], score_trend_path)
    summary["score_trend_path"] = str(score_trend_path)

    summary_path = output_dir / "ppo_single_room_training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[train] run output dir: {output_dir}")
    print(f"[train] reward curve saved to {reward_curve_path}")
    print(f"[train] score trends saved to {score_trend_path}")
    print(f"[train] summary saved to {summary_path}")
    print(f"[train] best_reward={summary['best_reward']:.3f}")
    print(f"[train] greedy_eval={summary['greedy_evaluation']}")


if __name__ == "__main__":
    main()
