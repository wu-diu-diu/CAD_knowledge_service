from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from reward import RewardConfig
from tqdm import tqdm

from env import EnvironmentConfig, SingleRoomLightingEnv
from model import LightingActorCritic
from train import PPOConfig, create_timestamped_output_dir, load_yaml_config, plot_reward_curve, plot_score_trends, set_seed
from visualize import save_room_grid_image


@dataclass
class CurriculumStage:
    """One stage in the curriculum: train on rooms with lamp_count in [min_lamps, max_lamps]."""

    min_lamps: int
    max_lamps: int
    episodes: int


@dataclass
class CurriculumConfig:
    """Curriculum learning schedule for multi-room training loaded from config.yaml."""

    stages: list[CurriculumStage] = field(default_factory=list)


def load_curriculum_config(config_payload: dict[str, Any]) -> CurriculumConfig:
    """Load curriculum stages from config.yaml payload."""
    raw_stages = config_payload.get("curriculum", {}).get("stages", [])
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("config.yaml 中缺少 curriculum.stages 配置，或 stages 为空。")

    stages = [CurriculumStage(**stage_config) for stage_config in raw_stages]
    return CurriculumConfig(stages=stages)


def _compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns for one episode in multi-room PPO."""
    steps = len(rewards)
    advantages = [0.0] * steps
    gae = 0.0
    for t in reversed(range(steps)):
        next_val = values[t + 1] if t + 1 < steps else last_value
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
    returns_tensor = advantages_tensor + torch.tensor(values, dtype=torch.float32)
    return advantages_tensor, returns_tensor


def load_room_dataset(room_dir: Path) -> list[dict]:
    """Load all room JSON files from a directory."""
    rooms = []
    for json_file in sorted(room_dir.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            continue
        for room_data in payload.values():
            if isinstance(room_data, dict) and "matrix" in room_data and "lamp_count" in room_data:
                rooms.append(room_data)
    print(f"[train] Loaded {len(rooms)} rooms from {room_dir}")
    return rooms


def _build_curriculum_pool(
    room_dataset: list[dict],
    stage: CurriculumStage,
) -> list[dict]:
    """Filter dataset to rooms whose lamp_count is within the stage range."""
    pool = [r for r in room_dataset if stage.min_lamps <= r["lamp_count"] <= stage.max_lamps]
    if not pool:
        print(f"[curriculum] WARNING: no rooms in lamp range [{stage.min_lamps}, {stage.max_lamps}], using full dataset")
        return room_dataset
    return pool


def ppo_update_multi_room(
    model: LightingActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout_buffer: list[dict],
    ppo_cfg: Any,
    device: torch.device,
) -> None:
    """PPO update using collected rollouts from multiple rooms."""
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

    states_tensor = torch.from_numpy(np.array(all_states)).to(device=device, dtype=torch.float32)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long, device=device)
    old_log_probs_tensor = torch.tensor(all_old_log_probs, dtype=torch.float32, device=device)
    advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32, device=device)
    returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=device)

    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

    for _ in range(ppo_cfg.ppo_epochs):
        eval_result = model.evaluate_actions(states_tensor, actions_tensor)
        new_log_probs = eval_result["log_prob"]
        entropy = eval_result["entropy"]
        values = eval_result["value"].squeeze(-1)

        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1.0 - ppo_cfg.clip_eps, 1.0 + ppo_cfg.clip_eps) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = ((values - returns_tensor) ** 2).mean()
        entropy_loss = -entropy.mean()
        loss = policy_loss + ppo_cfg.value_coef * value_loss + ppo_cfg.entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.grad_clip_norm)
        optimizer.step()


def train_multi_room(
    room_dataset: list[dict],
    model: LightingActorCritic,
    ppo_cfg: Any,
    env_cfg: EnvironmentConfig,
    output_dir: Path,
    curriculum: CurriculumConfig | None = None,
) -> dict:
    """
    Train PPO model on multiple rooms with random sampling.

    When `curriculum` is provided, training is split into stages defined by
    CurriculumConfig. Each stage restricts sampling to rooms whose lamp_count
    falls within [min_lamps, max_lamps], progressing from easy to hard.
    The total episode count is the sum of all stage episodes; ppo_cfg.episodes
    is ignored when curriculum is active.
    """
    device = torch.device(ppo_cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate)

    history = []
    best_reward = float("-inf")
    best_episode = 0
    rollout_buffer = []

    if curriculum is not None:
        total_episodes = sum(s.episodes for s in curriculum.stages)

        def episode_iter():
            ep = 0
            for stage_idx, stage in enumerate(curriculum.stages):
                pool = _build_curriculum_pool(room_dataset, stage)
                print(
                    f"[curriculum] Stage {stage_idx + 1}/{len(curriculum.stages)}: "
                    f"lamps=[{stage.min_lamps},{stage.max_lamps}] "
                    f"pool={len(pool)} rooms, episodes={stage.episodes}"
                )
                for _ in range(stage.episodes):
                    ep += 1
                    yield ep, pool, stage_idx
    else:
        total_episodes = ppo_cfg.episodes

        def episode_iter():
            for ep in range(1, total_episodes + 1):
                yield ep, room_dataset, 0

    for episode, active_pool, stage_idx in episode_iter():
        room_data = random.choice(active_pool)

        current_env_cfg = EnvironmentConfig(
            padded_size=env_cfg.padded_size,
            max_steps=env_cfg.max_steps,
            target_lamp_count=room_data["lamp_count"],
            turn_penalty=env_cfg.turn_penalty,
            reward_config=env_cfg.reward_config,
        )
        current_env_cfg.reward_config.target_lamp_count = room_data["lamp_count"]

        env = SingleRoomLightingEnv(room_data, config=current_env_cfg)

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

        advantages, returns = _compute_gae(
            episode_rewards,
            episode_values,
            ppo_cfg.gamma,
            ppo_cfg.gae_lambda,
            last_value=0.0,
        )

        rollout_buffer.append({
            "states": episode_states,
            "actions": episode_actions,
            "log_probs": episode_log_probs,
            "advantages": advantages,
            "returns": returns,
            "room_name": room_data["room_name"],
        })

        episode_reward = sum(episode_rewards)
        final_breakdown = env.last_breakdown
        history.append({
            "episode": episode,
            "episode_reward": episode_reward,
            "room_name": room_data["room_name"],
            "room_type": room_data.get("room_type", "unknown"),
            "lamp_count": room_data["lamp_count"],
            "curriculum_stage": stage_idx,
            "alignment_normalized": float(final_breakdown.alignment_normalized) if final_breakdown else 0.0,
            "wiring_normalized": float(final_breakdown.wiring_normalized) if final_breakdown else 0.0,
            "mst_cost": float(final_breakdown.mst_cost) if final_breakdown else 0.0,
        })

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_episode = episode
            torch.save(model.state_dict(), output_dir / "ppo_multi_room_best.pt")

        if len(rollout_buffer) >= ppo_cfg.rollout_episodes:
            ppo_update_multi_room(model, optimizer, rollout_buffer, ppo_cfg, device)
            rollout_buffer.clear()

        if episode % ppo_cfg.log_every_episodes == 0:
            recent = history[-ppo_cfg.log_every_episodes:]
            avg_reward = sum(h["episode_reward"] for h in recent) / len(recent)
            avg_align = sum(h["alignment_normalized"] for h in recent) / len(recent)
            stage_info = f" stage={stage_idx + 1}" if curriculum is not None else ""
            print(
                f"[train] episode={episode:04d}{stage_info} avg_reward={avg_reward:7.2f} "
                f"avg_align={avg_align:.3f} best={best_reward:7.2f}"
            )

    return {
        "history": history,
        "best_reward": best_reward,
        "best_episode": best_episode,
        "total_episodes": total_episodes,
    }


def evaluate_all_rooms(
    room_dataset: list[dict],
    model: LightingActorCritic,
    env_cfg: EnvironmentConfig,
    results_dir: Path,
) -> list[dict]:
    """Greedy rollout on every room, save layout image to results_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)
    for old_file in results_dir.glob("*.png"):
        old_file.unlink()
    device = next(model.parameters()).device
    results = []
    for i, room_data in tqdm(enumerate(room_dataset), total=len(room_dataset), desc="Evaluating rooms"):
        lamp_count = room_data["lamp_count"]
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
        room_name = room_data.get("room_name", f"room_{i:04d}")
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
    avg_potential = sum(r["potential_normalized"] for r in results) / max(len(results), 1)
    avg_align = sum(r["alignment_normalized"] for r in results) / max(len(results), 1)
    avg_wire = sum(r["wiring_normalized"] for r in results) / max(len(results), 1)
    print(
        f"[eval] {len(results)} rooms | avg_reward={avg_reward:.3f} "
        f"avg_potential={avg_potential:.3f} avg_align={avg_align:.3f} avg_wire={avg_wire:.3f}"
    )
    print(f"[eval] results saved to {results_dir}")
    return results


def run_multi_room_training(
    room_dir: Path,
    config_payload: dict[str, Any],
    ppo_cfg: Any,
    env_cfg: EnvironmentConfig,
    output_dir: Path,
) -> tuple[dict, str]:
    """Top-level multi-room training flow used by RL/train.py main()."""
    room_dataset = load_room_dataset(room_dir)
    if not room_dataset:
        raise ValueError(f"No rooms found in {room_dir}")

    model = LightingActorCritic(target_lamp_count=None)

    print(f"[train] Starting multi-room training with {len(room_dataset)} rooms")
    curriculum = None
    curriculum_payload = config_payload.get("curriculum", {})
    stage_payload = curriculum_payload.get("stages", []) if isinstance(curriculum_payload, dict) else []
    if stage_payload:
        curriculum = load_curriculum_config(config_payload)
        total = sum(s.episodes for s in curriculum.stages)
        print(f"[train] Curriculum learning enabled from config.yaml: {len(curriculum.stages)} stages, {total} total episodes")
    else:
        print("[train] Curriculum learning disabled: no curriculum.stages found in config.yaml")

    summary = train_multi_room(room_dataset, model, ppo_cfg, env_cfg, output_dir, curriculum=curriculum)
    summary["training_mode"] = "multi_room"
    summary["num_rooms"] = len(room_dataset)

    results_dir = room_dir.parent / "results"
    print(f"[train] Evaluating on all {len(room_dataset)} rooms...")
    eval_results = evaluate_all_rooms(room_dataset, model, env_cfg, results_dir)
    summary["all_room_evaluations"] = eval_results
    return summary, "ppo_multi_room_training_summary.json"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO model on multiple room tasks")
    parser.add_argument("--room_dir", type=str, required=True, help="Directory containing room JSON files for multi-room training")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of training episodes when curriculum is disabled")
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
    env_cfg = EnvironmentConfig(
        **config_payload.get("environment", {}),
        reward_config=reward_cfg,
    )

    summary, summary_filename = run_multi_room_training(
        Path(args.room_dir),
        config_payload,
        ppo_cfg,
        env_cfg,
        output_dir,
    )

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
