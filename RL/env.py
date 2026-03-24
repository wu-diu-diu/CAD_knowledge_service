from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.wiring_layout import _astar_route, _build_edge_candidates, _build_mst, _orient_edges_from_switch

from reward import RewardBreakdown, RewardCalculator, RewardConfig, RoomState
from visualize import save_room_grid_image


GridPoint = tuple[int, int]
# STOP_ACTION_INDEX is now computed dynamically based on padded_size
# It will be set as an instance attribute in __init__


@dataclass
class EnvironmentConfig:
    """Configuration for the single-room lamp placement environment."""

    padded_size: int = 32  ## 输入给PPO模型的观察空间是一个32x32的网格，因此需要将原始房间矩阵填充或裁剪到这个大小
    max_steps: int = 24  ## 每轮布局的最大步骤数，即智能体在一个房间上连续经过了24个状态，还没有布置结束就会被强制终止
    target_lamp_count: int = 4  ## 期望的灯具数量，智能体达到或超过这个数量时布局完成，进入终局奖励计算
    turn_penalty: float = 0.2  ## A*路径规划中的转弯惩罚，鼓励更直的布线路径，减少不必要的转弯
    reward_config: RewardConfig | None = None


class SingleRoomLightingEnv:
    """
    Single-room PPO environment for scheme 1 in `布局布线优化.md`.

    State:
        observation tensor with 6 channels
            0: current placeable mask
            1: placed lamps
            2: switch
            3: lamp progress (dynamic)
            4: room mask
            5: target lamp count (conditional input, fixed)

    Actions:
        0..H*W-1 -> place a lamp at the flattened H×W cell
        H*W      -> stop
    """

    def __init__(self, room_data: dict[str, Any], config: EnvironmentConfig | None = None) -> None:
        self.config = config or EnvironmentConfig()
        if self.config.reward_config is None:
            self.config.reward_config = RewardConfig(target_lamp_count=self.config.target_lamp_count)
        elif self.config.reward_config.target_lamp_count is None:
            self.config.reward_config.target_lamp_count = self.config.target_lamp_count

        self.reward_calculator = RewardCalculator(self.config.reward_config)

        self.room_name = str(room_data["room_name"])
        self.original_matrix = np.asarray(room_data["matrix"], dtype=np.int32)
        self.grid_rows, self.grid_cols = self.original_matrix.shape
        self.padded_size = self.config.padded_size
        self.stop_action_index = self.padded_size * self.padded_size  # Dynamic stop action index
        if self.grid_rows > self.padded_size or self.grid_cols > self.padded_size:
            raise ValueError(
                f"Room {self.room_name} shape {self.original_matrix.shape} exceeds padded size {self.padded_size}."
            )

        self.row_offset = (self.padded_size - self.grid_rows) // 2
        self.col_offset = (self.padded_size - self.grid_cols) // 2

        self.original_room_mask = self.original_matrix != 0
        self.original_door_mask = self.original_matrix == 2
        self.original_switch_mask = self.original_matrix == 3
        self.original_placeable_mask = self.original_matrix == 1

        self.route_grid = np.where(self.original_matrix == 0, 0, 1).astype(np.int32)
        self.route_grid[self.original_door_mask] = 0

        self._pair_cost_cache: dict[tuple[GridPoint, GridPoint], float] = {}
        self.episode_index = 0
        self.reset()

    @classmethod
    def from_json(
        cls,
        json_path: str | Path,
        *,
        room_name: str | None = None,
        config: EnvironmentConfig | None = None,
    ) -> "SingleRoomLightingEnv":
        """Create an environment from the stored room JSON."""
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"Expected non-empty room mapping in {path}")

        if room_name is None:
            _, room_payload = next(iter(payload.items()))
        else:
            room_payload = payload[room_name]
        return cls(room_payload, config=config)

    def reset(self) -> np.ndarray:
        """Reset the environment to an empty-lamp initial state."""
        self.original_lamp_mask = np.zeros_like(self.original_placeable_mask, dtype=bool)  ## 重置环境，创建一个与原始可放置区域形状相同的布尔数组，初始值为false，表示没有放置任何灯具
        self.current_step = 0
        self.done = False
        self.episode_index += 1
        self.last_breakdown: RewardBreakdown | None = None
        initial_state = self.current_room_state()
        self.initial_potential = self.reward_calculator.initial_potential(initial_state)
        self.current_potential = self.initial_potential
        return self.observation()

    def observation(self) -> np.ndarray:
        """
        Build the observation tensor expected by the actor-critic.

        Channels:
            0: current placeable mask（当前仍可放灯的格子，已放过灯的格子为0）
            1: placed lamp mask（已放灯位置）
            2: switch mask（开关位置）
            3: lamp progress（标量广播到全图，= 已放灯数 / 目标灯数）
               让网络感知"当前是第几步"，从而推断"剩余灯应放在哪个区域互补"
            4: room mask（房间内部区域，包括门、开关、已放灯；墙外为0）
               让网络区分房间边界与可放区域，辅助空间感知
            5: target lamp count（目标灯具数量，条件输入，固定不变）
               让模型知道"总共要放多少灯"，从而在第一步就能选择正确的布局策略
               例如：1个灯→中心，4个灯→2x2网格，9个灯→3x3网格
        """
        obs = np.zeros((6, self.padded_size, self.padded_size), dtype=np.float32)
        sl_r = slice(self.row_offset, self.row_offset + self.grid_rows)
        sl_c = slice(self.col_offset, self.col_offset + self.grid_cols)
        current_placeable = self.original_placeable_mask & ~self.original_lamp_mask
        obs[0, sl_r, sl_c] = current_placeable.astype(np.float32)
        obs[1, sl_r, sl_c] = self.original_lamp_mask.astype(np.float32)
        obs[2, sl_r, sl_c] = self.original_switch_mask.astype(np.float32)
        # ch3: 放灯进度，归一化到 [0, 1]，全图广播为常数平面（动态变化）
        progress = self.lamp_count / max(self.config.target_lamp_count, 1)
        obs[3, sl_r, sl_c] = progress
        # ch4: 房间内部掩码（墙外为0，房间内全为1）
        obs[4, sl_r, sl_c] = self.original_room_mask.astype(np.float32)
        # ch5: 目标灯具数量，归一化到 [0, 1]，全图广播为常数平面（固定不变，条件输入）
        max_possible = int(np.sum(self.original_placeable_mask))  # 可放置区域的格子数
        target_normalized = self.config.target_lamp_count / max(max_possible, 1)
        obs[5, sl_r, sl_c] = target_normalized
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Apply one action and return `(obs, reward, done, info)`.

        Invalid placement actions keep the state unchanged but incur a penalty.
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset() before stepping again.")

        invalid_action = False
        stop = action == self.stop_action_index  ## 如果action等于stop_action_index，表示策略模型输出的分布中，停止这个动作的概率最高，所以当action等于stop_action_index时，智能体选择了停止布局的动作，此时环境会进入终局状态，计算终局奖励并结束当前轮次的训练。

        if stop:
            if self.lamp_count < self.config.target_lamp_count:
                invalid_action = True
        else:
            pr, pc = divmod(int(action), self.padded_size)
            original_point = self.padded_to_original((pr, pc))  ## 得到了动作对应的32x32网格坐标后，需要将其转换回原始房间矩阵的坐标系，以便检查这个位置是否可以放置灯具，以及更新灯具的状态。padded_to_original方法会根据之前计算的行列偏移，将32x32的坐标映射回原始矩阵的坐标，如果映射后的坐标在原始矩阵范围内且是可放置区域，就返回这个坐标，否则返回None。
            if original_point is None or not self._is_placeable(original_point):
                invalid_action = True
            else:
                self.original_lamp_mask[original_point] = True  ## 如果动作合法且对应的位置可以放置灯具，就在original_lamp_mask中将这个位置标记为True，表示已经放置了一个灯具。这个布尔掩码会在构建观察状态和计算奖励时使用，智能体的目标是通过放置灯具来满足房间的照明需求，同时遵守约束条件。

        state = self.current_room_state()  ## 更新了灯具状态后，构建房间状态
        step_breakdown = self.reward_calculator.calculate_step_reward(
            state,
            prev_potential=self.current_potential,
            invalid_action=invalid_action,
        )
        self.current_potential = self.reward_calculator.potential(state)
        reward = step_breakdown.total

        self.current_step += 1
        reached_limit = self.current_step >= self.config.max_steps  ## 如果本轮布置的步骤数量超过了最大数量，那么强制结束当前轮次的训练，进入终局状态，计算终局奖励。这是为了防止智能体在某些房间上陷入无效的尝试，导致训练停滞不前
        ## self.lamp_count是一个只读属性，调用的时候不需要括号，直接访问就会通过计算lamp_mask中TRUE的数量，来计算当前的灯具数量
        reached_target = self.lamp_count >= self.config.target_lamp_count  ## 如果智能体放置的灯具数量达到了预设的目标数量，那么认为布置完成，进入终局状态，计算终局奖励。这是为了鼓励智能体尽快满足照明需求，完成布置任务。
        done = stop or reached_limit or reached_target

        terminal_bonus = 0.0
        routing_summary = None
        ## 如果布置完成，那么除了本步得到的奖励增量外，还会根据终局状态计算一个额外的奖励。
        if done:
            terminal_breakdown = self.reward_calculator.calculate_terminal_reward(
                state,
                pair_cost_provider=self.pair_cost,
                initial_potential=self.initial_potential,
            )
            terminal_bonus = terminal_breakdown.total
            reward += terminal_bonus
            step_breakdown.terminal_bonus = terminal_breakdown.terminal_bonus
            step_breakdown.alignment_normalized = terminal_breakdown.alignment_normalized
            step_breakdown.alignment_term = terminal_breakdown.alignment_term
            step_breakdown.wiring_normalized = terminal_breakdown.wiring_normalized
            step_breakdown.wiring_term = terminal_breakdown.wiring_term
            step_breakdown.mst_cost = terminal_breakdown.mst_cost
            step_breakdown.total = reward
            routing_summary = self.compute_terminal_routing()

        self.done = done
        self.last_breakdown = step_breakdown  ## 每调用一次step方法，都会计算一次step_breakdown，并将其保存在last_breakdown属性中，用于在训练过程中的日志记录和调试分析。
        info: dict[str, Any] = {
            "step": self.current_step,
            "stop_action": stop,
            "invalid_action": invalid_action,
            "target_lamp_count": self.config.target_lamp_count,
            "step_breakdown": step_breakdown,
        }
        if routing_summary is not None:
            info["routing_summary"] = routing_summary
        return self.observation(), reward, done, info

    def current_room_state(self) -> RoomState:
        """Build the reward state from the current environment masks."""
        return RoomState.from_channels(
            room_mask=self.original_room_mask,
            lamp_mask=self.original_lamp_mask,
            switch_mask=self.original_switch_mask,
            door_mask=self.original_door_mask,
        )

    def current_encoded_matrix(self) -> np.ndarray:
        """Return the original room matrix with current lamp placements encoded as 4."""
        encoded = np.zeros_like(self.original_matrix, dtype=np.int32)
        encoded[self.original_placeable_mask] = 1
        encoded[self.original_door_mask] = 2
        encoded[self.original_switch_mask] = 3
        encoded[self.original_lamp_mask] = 4
        return encoded

    def export_snapshot(self, output_path: str | Path) -> Path:
        """Save the current room state to a PNG image."""
        return save_room_grid_image(
            self.current_encoded_matrix(),
            output_path,
            cell_size=32,
            room_name=(
                f"{self.room_name} | episode={self.episode_index} "
                f"step={self.current_step} lamps={self.lamp_count}/{self.config.target_lamp_count}"
            ),
        )

    def padded_to_original(self, padded_point: GridPoint) -> GridPoint | None:
        """Map a 32x32 padded coordinate back to the room's original grid."""
        pr, pc = padded_point
        r = pr - self.row_offset
        c = pc - self.col_offset
        if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
            return int(r), int(c)
        return None

    def original_to_padded(self, original_point: GridPoint) -> GridPoint:
        """Map an original grid point into padded 32x32 coordinates."""
        return original_point[0] + self.row_offset, original_point[1] + self.col_offset

    def pair_cost(self, a: GridPoint, b: GridPoint) -> float:
        """
        Return cached A* routing cost between two legal grid points.

        This is the `M` lookup mentioned in scheme 1. The cache is populated
        lazily so the single-room test remains fast enough to iterate on.
        """
        key = (a, b) if a <= b else (b, a)
        if key in self._pair_cost_cache:
            return self._pair_cost_cache[key]
        ## 计算两个点之间的路径成本，首先检查缓存中是否已经存在这个点对的成本，如果存在则直接返回，否则调用_astar_route函数计算这两个点之间的最短路径和路径成本，并将结果存入缓存中以供后续查询。这个方法是为了避免在训练过程中重复计算同一对点之间的路径成本，从而提高效率。
        path, cost = _astar_route(self.route_grid, a, b, turn_penalty=self.config.turn_penalty)
        if path is None:
            cost = float("inf")
        self._pair_cost_cache[key] = float(cost)
        return self._pair_cost_cache[key]

    def compute_terminal_routing(self) -> dict[str, Any]:
        """
        Run the current scheme-1 routing pipeline on the active room state.

        Returns a compact summary useful for debugging training convergence.
        """
        switches = _mask_positions(self.original_switch_mask)
        lamps = _mask_positions(self.original_lamp_mask)
        nodes = switches + lamps
        if len(nodes) <= 1:
            return {"node_count": len(nodes), "edge_count": 0, "total_cost": 0.0, "routes": []}

        routing_grid = self.current_encoded_matrix()
        edges = _build_edge_candidates(routing_grid, nodes, turn_penalty=self.config.turn_penalty)
        mst_edges = _build_mst(edges, len(nodes))
        oriented = _orient_edges_from_switch(mst_edges, nodes, switch_idx=0)
        return {
            "node_count": len(nodes),
            "edge_count": len(oriented),
            "total_cost": float(sum(edge_cost for _, _, _, edge_cost in oriented)),
            "routes": [path for _, _, path, _ in oriented],
        }

    def _is_placeable(self, point: GridPoint) -> bool:
        """Check whether a lamp can be placed on the given original-grid point."""
        r, c = point
        return bool(self.original_placeable_mask[r, c] and not self.original_lamp_mask[r, c])

    @property
    def lamp_count(self) -> int:
        """Current number of placed lamps."""
        return int(self.original_lamp_mask.sum())


def _mask_positions(mask: np.ndarray) -> list[GridPoint]:
    """Convert a boolean mask into a list of grid points."""
    coords = np.argwhere(mask)
    return [tuple(map(int, coord)) for coord in coords]
