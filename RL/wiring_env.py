"""
单房间布线RL环境

任务：给定已完成灯具布局的房间，RL决定灯具的连接顺序，A*负责实际路径规划。

连接方式（Steiner-like）：
  - 维护 tree_cells: set[GridPoint]，记录树占据的所有格子（节点+线路）
  - 每步选择一个未连接灯具，找到该灯具到 tree_cells 中最近的格子作为接入点
  - A* 从灯具到接入点规划路径，路径上所有格子加入 tree_cells
  - 后续灯具可以从线路中间分支出去，而不必回到节点

观察空间（6通道 H×W）：
  ch0: 房间可通行区域掩码
  ch1: 已连接的线路路径掩码（已铺设的线）
  ch2: 已连接节点掩码（已接入树的灯具/开关）
  ch3: 未连接节点掩码（尚未接入的灯具）
  ch4: 开关位置
  ch5: 连接进度 = 已连接灯具数 / 总灯具数（标量广播）

动作空间（离散）：
  0..N-1 → 选择第 i 个灯具接入
  动作掩码：已连接的灯具被屏蔽
奖励设计：
    每步奖励：新增线路长度的负值（归一化）
    终局奖励：相对MST的线长改善
        - MST改善：与预计算的 MST baseline 成本比较，正值=比MST好，负值=比MST差
"""
from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.wiring_layout import _astar_route

GridPoint = tuple[int, int]


@dataclass
class WiringEnvConfig:
    """布线环境配置。"""
    padded_size: int = 48       # 观察空间的填充尺寸
    turn_penalty: float = 0.2   # A* 转弯惩罚
    # 每步奖励：路径增量成本
    step_cost_coef: float = 0.3          # 步进成本归一化系数（step_reward_fixed>0时不生效）
    invalid_action_penalty: float = 1.0  # 非法动作惩罚
    # 终局奖励系数
    terminal_length_coef: float = 1.0   # 相对MST的线长改善得分
    # 固定步进奖励（>0时替代步进代价惩罚）
    step_reward_fixed: float = 0.0       # >0 时启用固定步进奖励
    step_reward_weight: float = 1.0      # 固定步进奖励权重，实际奖励 = weight / n_lamps


class WiringEnv:
    """
    单房间布线RL环境。

    输入：已完成灯具布局的房间数据（matrix 中灯具编码为 4，开关为 3）。
    每步：选择一个未连接灯具 → A* 连接到已有树的最近格子 → 更新线路掩码。
    终止：所有灯具都已连接。
    """

    def __init__(self, room_data: dict[str, Any], config: WiringEnvConfig | None = None) -> None:
        self.config = config or WiringEnvConfig()
        self.room_name = str(room_data.get("room_name", "unknown"))
        self._room_data = room_data  # 保存用于 MST 预计算

        matrix = np.asarray(room_data["matrix"], dtype=np.int32)
        self.grid_rows, self.grid_cols = matrix.shape
        self.padded_size = self.config.padded_size

        if self.grid_rows > self.padded_size or self.grid_cols > self.padded_size:
            raise ValueError(
                f"Room {self.room_name} shape {matrix.shape} exceeds padded_size {self.padded_size}."
            )

        self.row_offset = (self.padded_size - self.grid_rows) // 2
        self.col_offset = (self.padded_size - self.grid_cols) // 2

        # 静态掩码（从 matrix 解析）
        self.room_mask = matrix != 0                  # 房间内部
        self.door_mask = matrix == 2                  # 门
        self.switch_mask = matrix == 3                # 开关
        self.lamp_mask_orig = matrix == 4             # 灯具（布局结果）

        # 路由网格：房间内部可通行，门阻断
        self.route_grid = np.where(matrix == 0, 0, 1).astype(np.int32)
        self.route_grid[self.door_mask] = 0

        # 灯具位置列表（固定顺序，作为动作索引）
        lamp_coords = np.argwhere(self.lamp_mask_orig)
        self.lamp_positions: list[GridPoint] = [tuple(map(int, c)) for c in lamp_coords]
        self.n_lamps = len(self.lamp_positions)

        # 开关位置（树的根节点）
        switch_coords = np.argwhere(self.switch_mask)
        if len(switch_coords) == 0:
            raise ValueError(f"Room {self.room_name} has no switch (value=3).")
        self.switch_pos: GridPoint = tuple(map(int, switch_coords[0]))

        # 房间对角线长度，用于奖励归一化
        self.room_diagonal = float(np.sqrt(self.grid_rows ** 2 + self.grid_cols ** 2))

        # 路由缓存
        self._route_cache: dict[tuple[GridPoint, GridPoint], tuple[list[GridPoint] | None, float]] = {}

        # 预计算 MST cost，结果缓存在 room_data 字典上避免同一房间重复计算
        _cache_key = f"_mst_cost_{self.config.turn_penalty}"
        if _cache_key not in self._room_data:
            self._room_data[_cache_key] = max(
                compute_mst_baseline(self._room_data, turn_penalty=self.config.turn_penalty)["total_cost"],
                1.0,
            )
        self._mst_cost: float = self._room_data[_cache_key]

        self.episode_index = 0
        self.reset()

    @classmethod
    def from_json(
        cls,
        json_path: str | Path,
        *,
        room_name: str | None = None,
        config: WiringEnvConfig | None = None,
    ) -> "WiringEnv":
        """从 JSON 文件创建环境。"""
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"Expected non-empty dict in {path}")
        # 支持单房间格式（直接含 matrix）和多房间格式（key→room_data）
        if "matrix" in payload:
            room_data = payload
        else:
            if room_name is None:
                _, room_data = next(iter(payload.items()))
            else:
                room_data = payload[room_name]
        return cls(room_data, config=config)

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """重置环境，开关作为树的初始节点。"""
        self.episode_index += 1
        self.connected: list[bool] = [False] * self.n_lamps  # 每个灯具是否已连接
        self.wire_mask = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)  # 已铺设线路
        self.connected_node_mask = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)  # 已连接节点

        # 开关是树的根，初始加入 tree_cells
        self.tree_cells: set[GridPoint] = {self.switch_pos}
        self.connected_node_mask[self.switch_pos] = True

        # 记录每步路径，用于终局统计
        self.step_paths: list[list[GridPoint]] = []
        self.step_costs: list[float] = []

        self.done = False
        return self.observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        执行一步：选择第 action 个灯具接入布线树。

        Returns:
            obs, reward, done, info
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset() first.")

        # 动作合法性检查
        if action < 0 or action >= self.n_lamps:
            return self.observation(), -self.config.invalid_action_penalty, self.done, {"invalid": True, "reason": "out_of_range"}
        if self.connected[action]:
            return self.observation(), -self.config.invalid_action_penalty, self.done, {"invalid": True, "reason": "already_connected"}

        lamp_pos = self.lamp_positions[action]

        # 找到 tree_cells 中距离 lamp_pos 最近的格子作为接入点
        entry_point = self._find_nearest_tree_cell(lamp_pos)
        if entry_point is None:
            return self.observation(), -self.config.invalid_action_penalty, self.done, {"invalid": True, "reason": "unreachable"}

        # A* 从灯具到接入点规划路径
        path, cost = self._cached_route(lamp_pos, entry_point)
        if path is None:
            return self.observation(), -self.config.invalid_action_penalty, self.done, {"invalid": True, "reason": "no_path"}

        # 更新树：路径上所有格子加入 tree_cells
        for cell in path:
            self.tree_cells.add(cell)
            self.wire_mask[cell] = True

        # 标记灯具节点已连接
        self.connected[action] = True
        self.connected_node_mask[lamp_pos] = True
        self.step_paths.append(path)
        self.step_costs.append(cost)

        # 计算步骤奖励（势能下降）
        reward = self._step_reward()

        # 检查终止
        n_connected = sum(self.connected)
        self.done = n_connected >= self.n_lamps

        terminal_info: dict[str, Any] = {}
        if self.done:
            terminal_reward, terminal_info = self._terminal_reward()
            reward += terminal_reward

        info: dict[str, Any] = {
            "lamp_idx": action,
            "lamp_pos": lamp_pos,
            "entry_point": entry_point,
            "path_length": len(path),
            "route_cost": cost,
            "n_connected": n_connected,
            "n_total": self.n_lamps,
            **terminal_info,
        }
        return self.observation(), reward, self.done, info

    def observation(self) -> np.ndarray:
        """
        构建 6 通道观察张量（填充到 padded_size × padded_size）。

        ch0: 房间可通行区域
        ch1: 已连接线路掩码
        ch2: 已连接节点掩码（开关 + 已连接灯具）
        ch3: 未连接灯具掩码
        ch4: 开关位置
        ch5: 连接进度（标量广播）
        """
        obs = np.zeros((6, self.padded_size, self.padded_size), dtype=np.float32)
        sr = slice(self.row_offset, self.row_offset + self.grid_rows)
        sc = slice(self.col_offset, self.col_offset + self.grid_cols)

        obs[0, sr, sc] = self.room_mask.astype(np.float32)
        obs[1, sr, sc] = self.wire_mask.astype(np.float32)
        obs[2, sr, sc] = self.connected_node_mask.astype(np.float32)

        # 未连接灯具掩码
        unconnected_mask = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)
        for i, pos in enumerate(self.lamp_positions):
            if not self.connected[i]:
                unconnected_mask[pos] = True
        obs[3, sr, sc] = unconnected_mask.astype(np.float32)

        obs[4, sr, sc] = self.switch_mask.astype(np.float32)

        # 连接进度
        progress = sum(self.connected) / max(self.n_lamps, 1)
        obs[5, sr, sc] = progress

        return obs

    def action_mask(self) -> np.ndarray:
        """返回合法动作掩码，shape=[n_lamps]，True 表示可选。"""
        return np.array([not c for c in self.connected], dtype=bool)

    # ------------------------------------------------------------------
    # 奖励计算
    # ------------------------------------------------------------------

    def _step_reward(self) -> float:
        """每步奖励：固定奖励或步进代价惩罚（取决于配置）。

        固定模式：reward = step_reward_weight / n_lamps
        代价模式：reward = -step_cost_coef * last_route_cost / room_diagonal
        """
        if self.config.step_reward_fixed > 0.0:
            return self.config.step_reward_weight / max(self.n_lamps, 1)
        if not self.step_costs:
            return 0.0
        last_cost = self.step_costs[-1]
        return -self.config.step_cost_coef * last_cost / max(self.room_diagonal, 1.0)

    def _terminal_reward(self) -> tuple[float, dict[str, Any]]:
        """终局奖励：相对MST的线长改善。正值=比MST好，负值=比MST差。"""
        total_cost = sum(self.step_costs)
        mst_relative = (self._mst_cost - total_cost) / self._mst_cost
        length_reward = self.config.terminal_length_coef * mst_relative
        info = {
            "terminal_reward": length_reward,
            "mst_relative": mst_relative,
            "mst_cost": self._mst_cost,
            "length_score": mst_relative,   # 保持 key 兼容性
            "sharing_score": 0.0,           # 保持 key 兼容性
            "max_depth": 0,                 # 保持 key 兼容性
            "total_cost": total_cost,
        }
        return length_reward, info

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _find_nearest_tree_cell(self, target: GridPoint) -> GridPoint | None:
        """
        BFS 从 target 出发，找到最近的 tree_cells 中的格子作为接入点。
        在路由网格上搜索（门阻断）。
        """
        tr, tc = target
        if not (0 <= tr < self.grid_rows and 0 <= tc < self.grid_cols):
            return None

        # 如果 target 本身在 tree_cells 中，直接返回
        if target in self.tree_cells:
            return target

        visited = set()
        queue: deque[GridPoint] = deque([target])
        visited.add(target)

        while queue:
            r, c = queue.popleft()
            if (r, c) in self.tree_cells:
                return (r, c)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in visited:
                    continue
                if not (0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols):
                    continue
                if self.route_grid[nr, nc] == 0 and (nr, nc) not in self.tree_cells:
                    continue
                visited.add((nr, nc))
                queue.append((nr, nc))

        return None

    def _cached_route(
        self, start: GridPoint, end: GridPoint
    ) -> tuple[list[GridPoint] | None, float]:
        """带缓存的 A* 路径规划。"""
        key = (start, end) if start <= end else (end, start)
        if key not in self._route_cache:
            path, cost = _astar_route(
                self.route_grid, start, end, turn_penalty=self.config.turn_penalty
            )
            self._route_cache[key] = (path, float(cost))
        cached_path, cached_cost = self._route_cache[key]
        # 如果缓存的是 end→start 方向，需要反转
        if key != (start, end) and cached_path is not None:
            return list(reversed(cached_path)), cached_cost
        return cached_path, cached_cost

    @property
    def n_connected(self) -> int:
        return sum(self.connected)


# ------------------------------------------------------------------
# 独立辅助函数
# ------------------------------------------------------------------

def compute_mst_baseline(
    room_data: dict[str, Any],
    turn_penalty: float = 0.2,
) -> dict[str, Any]:
    """
    计算 MST baseline 的布线结果，用于与 RL 对比。
    复用 preprocess/wiring_layout.py 中的 MST 算法。
    """
    from preprocess.wiring_layout import _build_edge_candidates, _build_mst, _orient_edges_from_switch

    matrix = np.asarray(room_data["matrix"], dtype=np.int32)
    route_grid = np.where(matrix == 0, 0, 1).astype(np.int32)
    route_grid[matrix == 2] = 0  # 门阻断

    switch_coords = np.argwhere(matrix == 3)
    lamp_coords = np.argwhere(matrix == 4)

    if len(switch_coords) == 0 or len(lamp_coords) == 0:
        return {"total_cost": 0.0, "routes": [], "n_nodes": 0}

    switch_pos = tuple(map(int, switch_coords[0]))
    lamp_positions = [tuple(map(int, c)) for c in lamp_coords]
    nodes = [switch_pos] + lamp_positions

    edges = _build_edge_candidates(route_grid, nodes, turn_penalty=turn_penalty)
    mst_edges = _build_mst(edges, len(nodes))
    oriented = _orient_edges_from_switch(mst_edges, nodes, switch_idx=0)

    total_cost = sum(cost for _, _, _, cost in oriented)
    routes = [path for _, _, path, _ in oriented]

    return {
        "total_cost": total_cost,
        "routes": routes,
        "n_nodes": len(nodes),
    }
