"""GNN 版单房间灯具布局环境。

与 RL/env.py 的核心区别：
  - 无 padding，只对房间内部节点建图
  - obs 是节点级特征 (N_interior, F)，不含墙壁节点
  - 动作空间：0..N_interior-1 放灯（内部节点索引），N_interior 是 stop
  - 奖励逻辑与 RL/env.py 一致：单步势能降低 + 终局对齐/布线奖励
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# 确保能 import RL 下的 reward 和 preprocess
REPO_ROOT = Path(__file__).resolve().parents[1]
RL_DIR = REPO_ROOT / "RL"
for p in (str(REPO_ROOT), str(RL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from preprocess.wiring_layout import _astar_route, _build_edge_candidates, _build_mst, _orient_edges_from_switch
from RL.reward import RewardBreakdown, RewardCalculator, RewardConfig, RoomState

GridPoint = tuple[int, int]


@dataclass
class GNNEnvConfig:
    max_steps: int = 24
    target_lamp_count: int = 4
    turn_penalty: float = 0.2
    reward_config: RewardConfig | None = None


class GraphRoomEnv:
    """单房间灯具布局环境（GNN 版，只对房间内部节点建图）。

    Observation: (N_interior, 6) float32，每个内部节点 6 维特征：
        f0: 归一化行坐标
        f1: 归一化列坐标
        f2: 是否可放灯（placeable & ~placed）
        f3: 是否已放灯
        f4: 是否是开关
        f5: 放灯进度 = placed / target

    Actions:
        0 .. N_interior-1  放灯（内部节点索引）
        N_interior          stop
    """

    # 节点特征维度
    N_FEATURES = 6

    def __init__(self, room_data: dict[str, Any], config: GNNEnvConfig | None = None) -> None:
        self.config = config or GNNEnvConfig()
        if self.config.reward_config is None:
            self.config.reward_config = RewardConfig(target_lamp_count=self.config.target_lamp_count)
        elif self.config.reward_config.target_lamp_count is None:
            self.config.reward_config.target_lamp_count = self.config.target_lamp_count

        self.reward_calculator = RewardCalculator(self.config.reward_config)

        self.room_name = str(room_data["room_name"])
        self.original_matrix = np.asarray(room_data["matrix"], dtype=np.int32)
        self.grid_rows, self.grid_cols = self.original_matrix.shape

        self.room_mask = self.original_matrix != 0
        self.door_mask = self.original_matrix == 2
        self.switch_mask = self.original_matrix == 3
        self.placeable_mask = self.original_matrix == 1

        # ── 内部节点映射 ──
        # interior_coords: (N_interior, 2) 每个内部节点的 (row, col)
        self.interior_coords = np.argwhere(self.room_mask)  # (N_int, 2)
        self.n_interior = len(self.interior_coords)
        self.stop_action = self.n_interior

        # grid (r,c) -> 内部节点索引，墙壁为 -1
        self._grid_to_node = np.full((self.grid_rows, self.grid_cols), -1, dtype=np.int32)
        for idx, (r, c) in enumerate(self.interior_coords):
            self._grid_to_node[r, c] = idx

        # 预计算每个内部节点的静态属性
        self._is_placeable_node = np.array(
            [bool(self.placeable_mask[r, c]) for r, c in self.interior_coords], dtype=bool
        )
        self._is_switch_node = np.array(
            [bool(self.switch_mask[r, c]) for r, c in self.interior_coords], dtype=bool
        )
        # 归一化坐标
        self._norm_rows = self.interior_coords[:, 0].astype(np.float32) / max(self.grid_rows - 1, 1)
        self._norm_cols = self.interior_coords[:, 1].astype(np.float32) / max(self.grid_cols - 1, 1)

        # ── 邻接关系（8邻域，只连接内部节点）──
        self.edge_index = self._build_interior_edges()

        # A* 布线用的网格（门不可走线）
        self.route_grid = np.where(self.original_matrix == 0, 0, 1).astype(np.int32)
        self.route_grid[self.door_mask] = 0

        self._pair_cost_cache: dict[tuple[GridPoint, GridPoint], float] = {}
        self.episode_index = 0
        self.reset()

    def _build_interior_edges(self) -> np.ndarray:
        """构建内部节点的 8 邻域边列表，返回 (2, E) int32。"""
        src_list, dst_list = [], []
        for idx, (r, c) in enumerate(self.interior_coords):
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                        neighbor_idx = self._grid_to_node[nr, nc]
                        if neighbor_idx >= 0:
                            src_list.append(idx)
                            dst_list.append(int(neighbor_idx))
        return np.array([src_list, dst_list], dtype=np.int32)

    def reset(self) -> np.ndarray:
        self.lamp_mask = np.zeros_like(self.placeable_mask, dtype=bool)
        self.current_step = 0
        self.done = False
        self.episode_index += 1
        self.last_breakdown: RewardBreakdown | None = None

        initial_state = self._room_state()
        self.initial_potential = self.reward_calculator.initial_potential(initial_state)
        self.current_potential = self.initial_potential
        return self._obs()

    def _obs(self) -> np.ndarray:
        """返回 (N_interior, 6) 节点特征。"""
        obs = np.zeros((self.n_interior, self.N_FEATURES), dtype=np.float32)
        # f0: 归一化行坐标
        obs[:, 0] = self._norm_rows
        # f1: 归一化列坐标
        obs[:, 1] = self._norm_cols
        # f2: 是否可放灯（placeable & ~placed）
        lamp_placed = self.lamp_mask[self.interior_coords[:, 0], self.interior_coords[:, 1]]
        obs[:, 2] = (self._is_placeable_node & ~lamp_placed).astype(np.float32)
        # f3: 是否已放灯
        obs[:, 3] = lamp_placed.astype(np.float32)
        # f4: 是否是开关
        obs[:, 4] = self._is_switch_node.astype(np.float32)
        # f5: 放灯进度
        obs[:, 5] = self.lamp_count / max(self.config.target_lamp_count, 1)
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Environment is done. Call reset() first.")

        invalid_action = False
        stop = action == self.stop_action

        if stop:
            if self.lamp_count < self.config.target_lamp_count:
                invalid_action = True
        else:
            if not (0 <= action < self.n_interior):
                invalid_action = True
            else:
                r, c = int(self.interior_coords[action, 0]), int(self.interior_coords[action, 1])
                if not self._is_placeable_node[action] or self.lamp_mask[r, c]:
                    invalid_action = True
                else:
                    self.lamp_mask[r, c] = True

        state = self._room_state()
        step_bd = self.reward_calculator.calculate_step_reward(
            state, prev_potential=self.current_potential, invalid_action=invalid_action
        )
        self.current_potential = self.reward_calculator.potential(state)
        reward = step_bd.total

        self.current_step += 1
        reached_limit = self.current_step >= self.config.max_steps
        reached_target = self.lamp_count >= self.config.target_lamp_count
        done = stop or reached_limit or reached_target

        if done:
            term_bd = self.reward_calculator.calculate_terminal_reward(
                state,
                pair_cost_provider=self.pair_cost,
                initial_potential=self.initial_potential,
                potential_quality_threshold=0.5,
            )
            reward += term_bd.total
            step_bd.terminal_bonus = term_bd.terminal_bonus
            step_bd.alignment_normalized = term_bd.alignment_normalized
            step_bd.alignment_term = term_bd.alignment_term
            step_bd.wiring_normalized = term_bd.wiring_normalized
            step_bd.wiring_term = term_bd.wiring_term
            step_bd.mst_cost = term_bd.mst_cost
            step_bd.total = reward

        self.done = done
        self.last_breakdown = step_bd
        info: dict[str, Any] = {
            "step": self.current_step,
            "stop_action": stop,
            "invalid_action": invalid_action,
            "step_breakdown": step_bd,
        }
        return self._obs(), reward, done, info

    def _room_state(self) -> RoomState:
        return RoomState.from_channels(
            room_mask=self.room_mask,
            lamp_mask=self.lamp_mask,
            switch_mask=self.switch_mask,
            door_mask=self.door_mask,
        )

    def current_encoded_matrix(self) -> np.ndarray:
        encoded = np.zeros_like(self.original_matrix, dtype=np.int32)
        encoded[self.placeable_mask] = 1
        encoded[self.door_mask] = 2
        encoded[self.switch_mask] = 3
        encoded[self.lamp_mask] = 4
        return encoded

    def pair_cost(self, a: GridPoint, b: GridPoint) -> float:
        key = (a, b) if a <= b else (b, a)
        if key not in self._pair_cost_cache:
            _, cost = _astar_route(self.route_grid, a, b, turn_penalty=self.config.turn_penalty)
            self._pair_cost_cache[key] = float(cost) if cost is not None else float("inf")
        return self._pair_cost_cache[key]

    @property
    def lamp_count(self) -> int:
        return int(self.lamp_mask.sum())
