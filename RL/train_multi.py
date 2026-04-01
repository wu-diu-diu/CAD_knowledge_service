from __future__ import annotations

import copy
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from reward import RewardConfig
from tqdm import tqdm

from env import EnvironmentConfig, SingleRoomLightingEnv
from model import LightingActorCritic
from train import PPOConfig, load_yaml_config, plot_reward_curve, plot_score_trends, set_seed
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


def _extract_shape_type(filename: str) -> str:
    """Extract shape type from filename like shape_L_6lamp_0001.json -> L"""
    # Handle multi_cut and other compound names: shape_multi_cut_8lamp_0003.json
    match = re.search(r'^shape_(.+)_\d+lamp_\d+', filename)
    return match.group(1) if match else "unknown"


def load_room_dataset(room_dir: Path) -> list[dict]:
    """Load all room JSON files from a directory, preserving filename and shape type."""
    rooms = []
    for json_file in sorted(room_dir.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        shape_type = _extract_shape_type(json_file.name)

        if not isinstance(payload, dict):
            continue

        # Handle both single-room and multi-room JSON formats
        if "matrix" in payload and "lamp_count" in payload:
            # Single room JSON
            payload["_filename"] = json_file.name
            payload["_shape_type"] = shape_type
            rooms.append(payload)
        else:
            # Multi-room JSON
            for room_data in payload.values():
                if isinstance(room_data, dict) and "matrix" in room_data and "lamp_count" in room_data:
                    room_data["_filename"] = json_file.name
                    room_data["_shape_type"] = shape_type
                    rooms.append(room_data)

    print(f"[load] Loaded {len(rooms)} rooms from {room_dir}")
    return rooms


def split_by_shape_stratified(
    rooms: list[dict],
    train_ratio: float = 0.64,
    val_ratio: float = 0.16,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split rooms by shape type, maintaining distribution across train/val/test sets."""
    rng = random.Random(seed)
    by_shape: dict[str, list[dict]] = defaultdict(list)

    for room in rooms:
        shape_type = room.get("_shape_type", "unknown")
        by_shape[shape_type].append(room)

    train, val, test = [], [], []

    print("[split] Stratified split by shape type:")
    for shape_type in sorted(by_shape.keys()):
        group = by_shape[shape_type]
        rng.shuffle(group)
        n = len(group)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Remaining goes to test

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

        print(f"  {shape_type:12s}: {n:3d} total -> "
              f"train={n_train:3d}, val={n_val:3d}, test={n - n_train - n_val:3d}")

    print(f"[split] Total: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def _count_shapes(rooms: list[dict]) -> dict[str, int]:
    """Count rooms by shape type."""
    return dict(Counter(r.get("_shape_type", "unknown") for r in rooms))


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


def evaluate_on_val_set(
    val_rooms: list[dict],
    model: LightingActorCritic,
    env_cfg: EnvironmentConfig,
    device: torch.device,
) -> dict[str, float]:
    """Greedy rollout on validation set, return average metrics."""
    model.eval()
    rewards, alignments, wirings = [], [], []
    with torch.no_grad():
        for room_data in val_rooms:
            lamp_count = room_data["lamp_count"]
            current_cfg = EnvironmentConfig(
                padded_size=env_cfg.padded_size,
                max_steps=env_cfg.max_steps,
                target_lamp_count=lamp_count,
                turn_penalty=env_cfg.turn_penalty,
                reward_config=copy.copy(env_cfg.reward_config),
            )
            current_cfg.reward_config.target_lamp_count = lamp_count
            env = SingleRoomLightingEnv(room_data, config=current_cfg)
            obs = env.reset()
            done = False
            total_r = 0.0
            while not done:
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
                action = int(model.act(obs_t, deterministic=True)["action"].item())
                obs, r, done, _ = env.step(action)
                total_r += float(r)
            rewards.append(total_r)
            bd = env.last_breakdown
            if bd:
                alignments.append(float(bd.alignment_normalized))
                wirings.append(float(bd.wiring_normalized))
    model.train()
    return {
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_alignment": float(np.mean(alignments)) if alignments else 0.0,
        "avg_wiring": float(np.mean(wirings)) if wirings else 0.0,
    }


def train_multi_room(
    room_dataset: list[dict],
    model: LightingActorCritic,
    ppo_cfg: Any,
    env_cfg: EnvironmentConfig,
    output_dir: Path,
    curriculum: CurriculumConfig | None = None,
    val_rooms: list[dict] | None = None,
) -> dict:
    """
    Train PPO model on multiple rooms with random sampling.

    When `curriculum` is provided, training is split into stages defined by
    CurriculumConfig. Each stage restricts sampling to rooms whose lamp_count
    falls within [min_lamps, max_lamps], progressing from easy to hard.
    The total episode count is the sum of all stage episodes; ppo_cfg.episodes
    is ignored when curriculum is active.

    When `val_rooms` is provided, validation is run at the end of each
    curriculum stage (or every log_every_episodes when no curriculum).
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
        # Compute episode boundaries where each stage ends
        stage_end_episodes: set[int] = set()
        ep_acc = 0
        for stage in curriculum.stages:
            ep_acc += stage.episodes
            stage_end_episodes.add(ep_acc)

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
        stage_end_episodes = set()

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
            reward_config=copy.copy(env_cfg.reward_config),
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
            "type": "train",
            "episode": episode,
            "episode_reward": episode_reward,
            "room_name": room_data.get("room_name", ""),
            "shape_type": room_data.get("_shape_type", "unknown"),
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
            train_recent = [h for h in history[-ppo_cfg.log_every_episodes:] if h.get("type") == "train"]
            avg_reward = sum(h["episode_reward"] for h in train_recent) / max(len(train_recent), 1)
            avg_align = sum(h["alignment_normalized"] for h in train_recent) / max(len(train_recent), 1)
            stage_info = f" stage={stage_idx + 1}" if curriculum is not None else ""
            print(
                f"[train] episode={episode:04d}{stage_info} avg_reward={avg_reward:7.2f} "
                f"avg_align={avg_align:.3f} best={best_reward:7.2f}"
            )

        # Run validation at end of each curriculum stage (or periodically without curriculum)
        run_val = (
            val_rooms is not None and (
                episode in stage_end_episodes
                or (not stage_end_episodes and episode % (ppo_cfg.log_every_episodes * 4) == 0)
            )
        )
        if run_val:
            val_metrics = evaluate_on_val_set(val_rooms, model, env_cfg, device)
            history.append({
                "type": "validation",
                "episode": episode,
                "curriculum_stage": stage_idx,
                **val_metrics,
            })
            print(
                f"[val]   episode={episode:04d} stage={stage_idx + 1 if curriculum else 0} "
                f"avg_reward={val_metrics['avg_reward']:7.2f} "
                f"avg_align={val_metrics['avg_alignment']:.3f} "
                f"avg_wiring={val_metrics['avg_wiring']:.3f}"
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
            "shape_type": room_data.get("_shape_type", "unknown"),
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
    output_base: Path,
) -> tuple[dict, str]:
    """Top-level multi-room training flow with train/val/test split and timestamped output."""
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all rooms
    all_rooms = load_room_dataset(room_dir)
    if not all_rooms:
        raise ValueError(f"No rooms found in {room_dir}")

    # Split by shape type
    seed = config_payload.get("training", {}).get("seed", 42)
    train_rooms, val_rooms, test_rooms = split_by_shape_stratified(
        all_rooms,
        train_ratio=0.64,
        val_ratio=0.16,
        test_ratio=0.20,
        seed=seed,
    )

    # Create model
    model = LightingActorCritic(target_lamp_count=None)

    # Load curriculum if present
    print(f"[train] Starting multi-room training on {len(train_rooms)} training rooms")
    curriculum = None
    curriculum_payload = config_payload.get("curriculum", {})
    stage_payload = curriculum_payload.get("stages", []) if isinstance(curriculum_payload, dict) else []
    if stage_payload:
        curriculum = load_curriculum_config(config_payload)
        total = sum(s.episodes for s in curriculum.stages)
        print(f"[train] Curriculum learning enabled: {len(curriculum.stages)} stages, {total} total episodes")
    else:
        print("[train] Curriculum learning disabled: no curriculum.stages found in config.yaml")

    # Train on training set with validation monitoring
    summary = train_multi_room(
        train_rooms,
        model,
        ppo_cfg,
        env_cfg,
        output_dir,
        curriculum=curriculum,
        val_rooms=val_rooms,
    )

    # Evaluate on test set
    test_results_dir = output_dir / "test_results"
    print(f"[train] Evaluating on {len(test_rooms)} test rooms...")
    test_results = evaluate_all_rooms(test_rooms, model, env_cfg, test_results_dir)

    # Compute test metrics
    test_metrics = {
        "avg_reward": float(np.mean([r["total_reward"] for r in test_results])),
        "avg_potential": float(np.mean([r["potential_normalized"] for r in test_results])),
        "avg_alignment": float(np.mean([r["alignment_normalized"] for r in test_results])),
        "avg_wiring": float(np.mean([r["wiring_normalized"] for r in test_results])),
    }

    # Enrich summary
    summary.update({
        "timestamp": timestamp,
        "output_dir": str(output_dir),
        "training_mode": "multi_room",
        "dataset": {
            "total_rooms": len(all_rooms),
            "train_rooms": len(train_rooms),
            "val_rooms": len(val_rooms),
            "test_rooms": len(test_rooms),
            "shape_distribution": {
                "train": _count_shapes(train_rooms),
                "val": _count_shapes(val_rooms),
                "test": _count_shapes(test_rooms),
            },
        },
        "test_evaluations": test_results,
        "test_metrics": test_metrics,
        "best_model": {
            "episode": summary["best_episode"],
            "reward": summary["best_reward"],
            "checkpoint_path": "ppo_multi_room_best.pt",
        },
    })

    return summary, "ppo_multi_room_training_summary.json"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO model on multiple room tasks")
    parser.add_argument("--room_dir", type=str, required=True, help="Directory containing room JSON files")
    parser.add_argument("--output_base", type=str, default=None, help="Base output directory (default: RL/output_multiroom)")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of training episodes when curriculum is disabled")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "RL" / "config.yaml"
    config_payload = load_yaml_config(config_path)

    output_base = Path(args.output_base) if args.output_base else repo_root / "RL" / "output_multiroom"

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
        output_base,
    )

    output_dir = Path(summary["output_dir"])

    # Copy config for reproducibility
    shutil.copy2(config_path, output_dir / "config.yaml")
    summary["config_path"] = str(config_path)

    # Plot training curves (filter to train entries only)
    train_history = [h for h in summary["history"] if h.get("type") == "train"]
    reward_curve_path = output_dir / "reward_curve.png"
    plot_reward_curve(train_history, reward_curve_path, moving_window=ppo_cfg.reward_curve_moving_window)
    summary["reward_curve_path"] = str(reward_curve_path)

    score_trend_path = output_dir / "score_trends.png"
    plot_score_trends(train_history, score_trend_path)
    summary["score_trend_path"] = str(score_trend_path)

    # Save summary JSON
    summary_path = output_dir / summary_filename
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[done] Output dir:     {output_dir}")
    print(f"[done] Reward curve:   {reward_curve_path}")
    print(f"[done] Score trends:   {score_trend_path}")
    print(f"[done] Summary:        {summary_path}")
    print(f"[done] Best model:     {output_dir / 'ppo_multi_room_best.pt'}")
    print(f"[done] Test results:   {output_dir / 'test_results'}")
    print(f"[done] best_reward={summary['best_reward']:.3f}")
    print(f"[done] test_metrics={summary['test_metrics']}")


if __name__ == "__main__":
    main()
