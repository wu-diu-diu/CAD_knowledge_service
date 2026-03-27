"""单房间 GCN-PPO 验证脚本。

使用 graph/unregular.json（热量室准备间，35×33，5盏灯）验证 GCN actor-critic 可行性。
复用 RL/ 下的环境、奖励和 PPO 逻辑，不修改任何现有文件。

运行方式：
    cd /home/chen/punchy/CAD_knowledge_service
    .venv/bin/python graph/train_gcn.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 路径设置
REPO_ROOT = Path(__file__).resolve().parents[1]
RL_DIR = REPO_ROOT / "RL"
GRAPH_DIR = REPO_ROOT / "graph"
sys.path.insert(0, str(RL_DIR))
sys.path.insert(0, str(GRAPH_DIR))

from visualize import save_room_grid_image
from env import GraphRoomEnv, GNNEnvConfig
from gcn_actor_critic import GCNActorCritic

# ── 超参数 ──────────────────────────────────────────────────────────────────
EPISODES = 6000
ROLLOUT_EPISODES = 32
PPO_EPOCHS = 4
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.05
GRAD_CLIP = 0.5
LOG_EVERY = 100
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = GRAPH_DIR / "results"


def load_room() -> dict:
    json_path = GRAPH_DIR / "unregular.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    room_data = next(iter(payload.values()))
    # 统一 lamp_count 字段（此 JSON 嵌套在 lamp 子字典里）
    room_data["lamp_count"] = room_data["lamp"]["lamp_count"]
    return room_data


def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    adv = torch.tensor(advantages, dtype=torch.float32)
    ret = adv + torch.tensor(values, dtype=torch.float32)
    return adv, ret


def collect_episode(
    env: GraphRoomEnv,
    model: GCNActorCritic,
    device: torch.device,
) -> dict:
    obs = env.reset()
    obs_list, actions, rewards, log_probs, values = [], [], [], [], []
    done = False
    while not done:
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            rollout = model.act(obs_t, deterministic=False)
        action = int(rollout["action"].item())
        next_obs, reward, done, info = env.step(action)

        obs_list.append(obs)
        actions.append(action)
        rewards.append(float(reward))
        log_probs.append(float(rollout["log_prob"].item()))
        values.append(float(rollout["value"].squeeze().item()))
        obs = next_obs

    final_bd = env.last_breakdown
    return {
        "obs_list": obs_list,
        "actions": actions,
        "rewards": rewards,
        "log_probs": log_probs,
        "values": values,
        "episode_reward": float(sum(rewards)),
        "alignment_normalized": float(final_bd.alignment_normalized) if final_bd else 0.0,
        "wiring_normalized": float(final_bd.wiring_normalized) if final_bd else 0.0,
    }


def ppo_update(
    model: GCNActorCritic,
    optimizer: torch.optim.Optimizer,
    batch_obs: torch.Tensor,
    batch_actions: torch.Tensor,
    batch_old_log_probs: torch.Tensor,
    batch_returns: torch.Tensor,
    batch_advantages: torch.Tensor,
    device: torch.device,
) -> None:
    for _ in range(PPO_EPOCHS):
        evaluated = model.evaluate_actions(batch_obs, batch_actions)
        log_probs_t = evaluated["log_prob"]
        entropy_t = evaluated["entropy"]
        values_t = evaluated["value"].squeeze(-1)

        ratio = torch.exp(log_probs_t - batch_old_log_probs)
        s1 = ratio * batch_advantages
        s2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * batch_advantages
        policy_loss = -torch.min(s1, s2).mean()
        value_loss = nn.functional.mse_loss(values_t, batch_returns)
        entropy_bonus = entropy_t.mean()
        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()


def greedy_eval(
    env: GraphRoomEnv,
    model: GCNActorCritic,
    device: torch.device,
    save_path: Path,
) -> float:
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
        action = int(model.act(obs_t, deterministic=True)["action"].item())
        obs, reward, done, _ = env.step(action)
        total_reward += float(reward)

    final_bd = env.last_breakdown
    title = (
        f"热量室准备间 GCN | r={total_reward:.2f} "
        f"a={final_bd.alignment_normalized:.2f} w={final_bd.wiring_normalized:.2f}"
    ) if final_bd else f"热量室准备间 GCN | r={total_reward:.2f}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_room_grid_image(env.current_encoded_matrix(), save_path, cell_size=20, room_name=title)
    print(f"[eval] greedy reward={total_reward:.3f} | saved → {save_path}")
    return total_reward


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(DEVICE)
    print(f"[train_gcn] device={device}")

    room_data = load_room()
    lamp_count = room_data["lamp_count"]
    print(f"[train_gcn] room={room_data['room_name']} size={room_data['grid_rows']}×{room_data['grid_cols']} lamps={lamp_count}")

    env_cfg = GNNEnvConfig(
        max_steps=lamp_count * 4,
        target_lamp_count=lamp_count,
    )
    env = GraphRoomEnv(room_data, config=env_cfg)
    print(f"[train_gcn] interior nodes={env.n_interior} action_space={env.n_interior+1}")

    model = GCNActorCritic(in_features=env.N_FEATURES, hidden=64, target_lamp_count=lamp_count).to(device)
    model.set_adj(env.n_interior, env.edge_index, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    episode_idx = 0
    reward_history: list[float] = []

    while episode_idx < EPISODES:
        rollout_obs, rollout_actions = [], []
        rollout_advantages, rollout_returns, rollout_old_log_probs = [], [], []
        batch_rewards: list[float] = []

        episodes_this_batch = min(ROLLOUT_EPISODES, EPISODES - episode_idx)
        for _ in range(episodes_this_batch):
            episode_idx += 1
            ep = collect_episode(env, model, device)
            adv, ret = compute_gae(ep["rewards"], ep["values"], GAMMA, GAE_LAMBDA)
            rollout_obs.extend(ep["obs_list"])
            rollout_actions.extend(ep["actions"])
            rollout_advantages.extend(adv.tolist())
            rollout_returns.extend(ret.tolist())
            rollout_old_log_probs.extend(ep["log_probs"])
            batch_rewards.append(ep["episode_reward"])

        reward_history.extend(batch_rewards)

        batch_obs_t = torch.from_numpy(np.stack(rollout_obs)).to(device=device, dtype=torch.float32)
        batch_actions_t = torch.tensor(rollout_actions, dtype=torch.long, device=device)
        batch_old_lp_t = torch.tensor(rollout_old_log_probs, dtype=torch.float32, device=device)
        batch_ret_t = torch.tensor(rollout_returns, dtype=torch.float32, device=device)
        adv_t = torch.tensor(rollout_advantages, dtype=torch.float32, device=device)
        if adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        ppo_update(model, optimizer, batch_obs_t, batch_actions_t, batch_old_lp_t, batch_ret_t, adv_t, device)

        if episode_idx % LOG_EVERY == 0:
            window = reward_history[-LOG_EVERY:]
            avg_r = sum(window) / len(window)
            print(f"[train_gcn] episode={episode_idx:04d} avg_reward={avg_r:.3f}")

    # 训练结束，greedy 推理并保存布局图
    model.eval()
    greedy_eval(env, model, device, RESULTS_DIR / "unregular_gcn.png")


if __name__ == "__main__":
    main()
