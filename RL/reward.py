from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import inf
from typing import Callable, Iterable, Mapping

import numpy as np


GridPoint = tuple[int, int]
PairCostCallable = Callable[[GridPoint, GridPoint], float]
PairCostMapping = Mapping[tuple[GridPoint, GridPoint], float]
PairCostProvider = PairCostCallable | PairCostMapping | None


@dataclass
class RewardConfig:
    """
    Reward hyper-parameters derived from `布局布线优化.md`.

    The main formula is:
        R = 照度均匀性奖励 + 照度场重心奖励 + 对齐奖励 + 布线奖励
    """

    uniformity_coef: float = 2.0  ## 照度分布均匀性分数的最终系数
    illum_centroid_coef: float = 1.5  ## 照度场重心分数的最终系数
    alignment_coef: float = 1.5  ## 行列对齐分数的最终系数
    wiring_coef: float = 1.0  ## 布线分数的最终系数
    invalid_action_penalty: float = 10.0  ## 非法动作的惩罚，例如试图在墙上放灯
    light_height_cells: float = 1.0  ## 灯具到计算平面的等效高度，用于反平方衰减模型
    target_lamp_count: int | None = None


@dataclass
class RoomState:
    """
    Structured room state used by reward computation.

    Attributes:
        room_mask: Cells belonging to the room interior.
        lamp_mask: Currently placed lamp cells.
        switch_mask: Fixed switch cells.
        door_mask: Door cells.
        placeable_mask: Cells where lamps are allowed to be placed.

    `placeable_mask` is intentionally separated from `room_mask`, because doors
    and switch locations are inside the room but are not valid lamp positions.
    """

    room_mask: np.ndarray
    lamp_mask: np.ndarray
    switch_mask: np.ndarray
    door_mask: np.ndarray
    placeable_mask: np.ndarray

    @classmethod
    def from_encoded_matrix(cls, matrix: list[list[int]] | np.ndarray) -> "RoomState":
        """
        Build a room state from the repo's encoded integer matrix.

        Encodings:
            0 -> invalid / wall
            1 -> placeable
            2 -> door
            3 -> switch
            4 -> lamp

        Note:
            When using only an encoded matrix, lamps are assumed to have been
            placed on originally placeable cells. For stricter legality checks
            during training, prefer `from_channels(...)` and keep the original
            placeable mask untouched.
        """
        grid = np.asarray(matrix, dtype=np.int32)
        room_mask = grid != 0
        lamp_mask = grid == 4
        switch_mask = grid == 3
        door_mask = grid == 2
        placeable_mask = (grid == 1) | (grid == 4)
        return cls(
            room_mask=room_mask,
            lamp_mask=lamp_mask,
            switch_mask=switch_mask,
            door_mask=door_mask,
            placeable_mask=placeable_mask,
        )

    @classmethod
    def from_channels(
        cls,
        room_mask: np.ndarray,
        lamp_mask: np.ndarray,
        switch_mask: np.ndarray,
        door_mask: np.ndarray,
    ) -> "RoomState":
        """Build a room state from explicit binary masks."""
        room = np.asarray(room_mask, dtype=bool)
        lamp = np.asarray(lamp_mask, dtype=bool)
        switch = np.asarray(switch_mask, dtype=bool)
        door = np.asarray(door_mask, dtype=bool)
        placeable = room & ~switch & ~door
        return cls(room_mask=room, lamp_mask=lamp, switch_mask=switch, door_mask=door, placeable_mask=placeable)

    @property
    def shape(self) -> tuple[int, int]:
        return self.room_mask.shape

    @property
    def lamp_positions(self) -> list[GridPoint]:
        return _mask_positions(self.lamp_mask)

    @property
    def switch_positions(self) -> list[GridPoint]:
        return _mask_positions(self.switch_mask)

    @property
    def door_positions(self) -> list[GridPoint]:
        return _mask_positions(self.door_mask)

    @property
    def room_area_m2(self) -> float:
        return float(self.room_mask.sum())


@dataclass
class RewardBreakdown:  ## 是一个数据类，用于详细描述每一步或者终局评估的时候的奖励构成
    """Detailed reward report for one step or terminal evaluation."""

    total: float
    illum_centroid: float
    uniformity: float
    rules: float
    wiring: float
    cost: float
    invalid_action: float
    terminal: float
    diagnostics: dict[str, float | int | bool | list[GridPoint]]


class RewardCalculator:
    """
    Reward calculator for RL lamp placement.

    This class implements the reward design described in
    `summary_docs/布局布线优化.md`:
        - illuminance uniformity score in [0, 1]
        - illuminance-centroid score in [0, 1]
        - alignment score in [0, 1]
        - wiring score in [0, 1]

    Terminal evaluation now only contributes diagnostics and success flags.
    """

    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def calculate_step_reward(
        self,
        state: RoomState,
        *,
        invalid_action: bool = False,
        pair_cost_provider: PairCostProvider = None,
    ) -> RewardBreakdown:
        """Compute the weighted step reward for the current room state."""
        illuminance_map = self.illuminance_map(state)  ## map存储房间每个格子的照度值
        uniformity_score = self.uniformity_score(state, illuminance_map=illuminance_map)  ## 最低照度与平均照度的比值，衡量照度分布的均匀程度，越接近1越好
        illum_centroid_score, lamp_center_distance = self.illum_centroid_score(  ## 照度重心和房间几何中心的距离，经过归一化后得到分数，越接近1越好
            state,
            illuminance_map=illuminance_map,
        )
        alignment_score = self.alignment_score(state.lamp_positions)  ## 对齐分数，衡量灯具是否在同一行或同一列，越接近1越好
        wiring_score, mst_cost = self.wiring_score(state, pair_cost_provider=pair_cost_provider)  ## 布线分数，基于最小生成树的成本与参考成本的比值，越接近1越好
        invalid_penalty = -self.config.invalid_action_penalty if invalid_action else 0.0

        total = (
            self.config.uniformity_coef * uniformity_score
            + self.config.illum_centroid_coef * illum_centroid_score
            + self.config.alignment_coef * alignment_score
            + self.config.wiring_coef * wiring_score
            + invalid_penalty
        )

        diagnostics: dict[str, float | int | bool | list[GridPoint]] = {
            "lamp_count": len(state.lamp_positions),
            "mst_cost": mst_cost,
            "invalid_action": invalid_action,
            "uniformity_score": uniformity_score,
            "illum_centroid_score": illum_centroid_score,
            "lamp_center_distance": lamp_center_distance,
            "alignment_score": alignment_score,
            "wiring_score": wiring_score,
            "invalid_penalty": invalid_penalty,
        }

        return RewardBreakdown(
            total=total,
            illum_centroid=self.config.illum_centroid_coef * illum_centroid_score,
            uniformity=self.config.uniformity_coef * uniformity_score,
            rules=self.config.alignment_coef * alignment_score + invalid_penalty,
            wiring=self.config.wiring_coef * wiring_score,
            cost=0.0,
            invalid_action=invalid_penalty,
            terminal=0.0,
            diagnostics=diagnostics,
        )

    def calculate_terminal_reward(
        self,
        state: RoomState,
        *,
        pair_cost_provider: PairCostProvider = None,
    ) -> RewardBreakdown:
        """
        Compute terminal diagnostics without adding extra reward terms.

        The reward is now fully composed of:
            - illuminance uniformity
            - illuminance-field centroid
            - alignment
            - wiring cost
        """
        step = self.calculate_step_reward(state, pair_cost_provider=pair_cost_provider)
        success = (  ## 判断当前状态是否满足成功条件，即灯具数量是否达到了目标数量（如果有设定的话）。如果没有设定目标数量，那么只要有灯具就算成功。这是为了决定是否给予终局成功奖励。
            int(step.diagnostics.get("lamp_count", 0)) == int(self.config.target_lamp_count)
            if self.config.target_lamp_count is not None
            else int(step.diagnostics.get("lamp_count", 0)) > 0
        )
        step.terminal = 0.0
        step.total = 0.0
        step.diagnostics["terminal_success"] = success
        return step

    def illuminance_map(self, state: RoomState) -> np.ndarray:
        """
        Estimate the illuminance field over the whole room grid.

        A simple inverse-square decay model is used:
        简化的点光源衰减模型，粗略估计房间中每个格子的照度值，假设每个灯具都是一个点光源，照度随着距离的平方衰减
            E_ij = 1 / (dx^2 + dy^2 + h^2)

        where h is the equivalent light height in grid units.
        """
        lamps = state.lamp_positions
        illuminance = np.zeros(state.shape, dtype=np.float32) ## 照度分布网格，大小和房间相同，初始值为0
        if not lamps:
            return illuminance

        rows, cols = np.indices(state.shape, dtype=np.float32)
        h_sq = float(self.config.light_height_cells) ** 2
        for lamp_r, lamp_c in lamps:
            dist_sq = (rows - float(lamp_r)) ** 2 + (cols - float(lamp_c)) ** 2 + h_sq  ## 计算每个网格点到灯具的距离平方加上灯具高度的平方，避免除以零
            illuminance += 1.0 / np.maximum(dist_sq, 1e-6)

        illuminance *= state.room_mask.astype(np.float32)  ## 只保留房间内部的照度值，墙外的照度为0
        return illuminance

    def uniformity_score(
        self,
        state: RoomState,
        *,
        illuminance_map: np.ndarray | None = None,
    ) -> float:
        """
        Use normalized illuminance variance as the uniformity score.

        Let room-cell illuminance values be E. We first compute:
            normalized_variance = Var(E) / mean(E)^2

        Then convert it to a bounded score:
            score = 1 / (1 + normalized_variance)

        Smaller variance means more uniform lighting, so the final score is
        larger when the variance is smaller. The score lies in (0, 1].
        """
        if illuminance_map is None:
            illuminance_map = self.illuminance_map(state)

        room_values = illuminance_map[state.room_mask]  ## 提取房间内部的照度值，得到一个一维数组
        if room_values.size == 0:
            return 0.0

        avg_illum = float(np.mean(room_values))  ## 计算房间内部的平均照度，作为归一化方差分母的一部分。如果平均照度非常小，则直接返回0分，避免除以零或者得到不稳定结果。
        if avg_illum <= 1e-6:
            return 0.0

        variance = float(np.var(room_values))  ## 房间内所有格子的照度方差，越小表示越均匀
        normalized_variance = variance / max(avg_illum * avg_illum, 1e-6)  ## 用平均照度平方归一化，消除亮度绝对尺度影响
        score = 1.0 / (1.0 + normalized_variance)  ## 方差越小，score越接近1；方差越大，score越接近0
        return float(np.clip(score, 0.0, 1.0))

    def illum_centroid_score(
        self,
        state: RoomState,
        *,
        illuminance_map: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """
        Score the illuminance-field centroid for staying near the room centroid.

        Returns:
            score: normalized centroid-closeness score in [0, 1]
            distance: Euclidean distance between illuminance centroid and room centroid
        """
        if illuminance_map is None:
            illuminance_map = self.illuminance_map(state)

        room_cells = np.argwhere(state.room_mask)  ## 获取房间内部所有有效网格的坐标，得到一个二维数组，每行是一个网格的行列坐标
        if room_cells.size == 0:
            return 0.0, inf

        room_center = room_cells.mean(axis=0)  ## 计算房间内部网格坐标的平均值，得到房间的几何中心坐标，作为理想的照度重心位置。这个位置通常位于房间的中间，如果房间形状规则的话。
        room_weights = illuminance_map[state.room_mask]  ## 提取房间内部网格的照度值，得到一个一维数组，作为每个网格点的权重。照度越高的网格点对重心位置的影响越大。
        total_illum = float(np.sum(room_weights))
        if total_illum <= 1e-6:
            return 0.0, inf

        illum_centroid = np.sum(room_cells * room_weights[:, None], axis=0) / total_illum  ## 计算照度重心的坐标，方法是对每个网格点的坐标乘以其照度权重，然后求和后除以总照度。这样得到的重心位置会偏向于照度较高的区域。
        distance = float(np.linalg.norm(illum_centroid - room_center))  ## 计算照度重心和房间几何中心之间的欧几里得距离，作为重心偏离程度的度量。距离越小说明照度重心越接近房间中心，分数应该越高。
        farthest_room_distance = float(np.max(np.linalg.norm(room_cells - room_center, axis=1)))  ## 计算房间内部所有网格点到房间中心的最大距离，作为距离归一化的参考值。这个值反映了房间的大小和形状，距离越大说明房间越大或者越不规则。
        normalization = max(farthest_room_distance, 1.0)  ## 归一化因子，避免除以零或者得到过大的分数。至少为1.0，确保合理的分数范围。
        score = max(0.0, 1.0 - distance / normalization)  ## 计算最终的重心分数，方法是用1减去距离与归一化因子的比值。距离越小分数越接近1，距离越大分数越接近0。通过这种方式，重心位置越接近房间中心，分数越高。
        return score, distance

    def legacy_spacing_uniformity(self, state: RoomState) -> float:
        """
        Legacy nearest-neighbor spacing metric kept only for debugging/reference.
        """
        lamps = state.lamp_positions
        if len(lamps) < 2:
            return 0.0

        nn_dists = []
        for idx, lamp in enumerate(lamps):
            others = lamps[:idx] + lamps[idx + 1 :]
            nn_dists.append(min(_manhattan(lamp, other) for other in others))

        nn_array = np.asarray(nn_dists, dtype=np.float32)
        variance = float(np.var(nn_array))  ## 计算每个灯具最邻近距离的方差，方差越小表示灯具分布越均匀
        mean = float(np.mean(nn_array)) if nn_array.size > 0 else 1.0  ## 计算每个灯具最邻近距离的平均值，平均值越大表示灯具之间的距离越远
        normalized_variance = variance / max(mean * mean, 1.0)  ## 将方差除以平均值的平方进行归一化，避免不同规模的房间之间的比较受到距离尺度的影响
        return -normalized_variance

    def alignment_score(self, lamps: Iterable[GridPoint]) -> float:
        """
        Measure whether lamps form shared rows/columns.

        The score is in [0, 1]:
            - 1.0 means every lamp shares both a row and a column with others
            - 0.0 means lamps are completely scattered diagonally
        """
        lamp_list = list(lamps)
        if len(lamp_list) < 2:
            return 1.0

        row_counts = Counter(r for r, _ in lamp_list)
        col_counts = Counter(c for _, c in lamp_list)
        row_shared = sum(1 for r, _ in lamp_list if row_counts[r] > 1) / len(lamp_list)
        col_shared = sum(1 for _, c in lamp_list if col_counts[c] > 1) / len(lamp_list)
        return float(0.5 * (row_shared + col_shared))

    def wiring_score(
        self,
        state: RoomState,
        *,
        pair_cost_provider: PairCostProvider = None,
    ) -> tuple[float, float]:
        """
        Compute a normalized wiring score from the MST cost over switch + lamp terminals.

        If a precomputed pair-cost matrix `M` exists, pass it through
        `pair_cost_provider`. Otherwise the fallback is Manhattan distance.
        """
        terminals = state.switch_positions + state.lamp_positions
        if len(terminals) <= 1:
            return 1.0, 0.0

        mst_cost = self._mst_total_cost(terminals, pair_cost_provider=pair_cost_provider)
        if mst_cost == inf:
            return 0.0, inf

        reference_cost = self._reference_wiring_cost(state, terminal_count=len(terminals))
        score = max(0.0, 1.0 - float(mst_cost) / max(reference_cost, 1e-6))
        return score, mst_cost

    def _mst_total_cost(
        self,
        terminals: list[GridPoint],
        *,
        pair_cost_provider: PairCostProvider = None,
    ) -> float:
        """Compute MST cost using Kruskal over terminal complete graph."""
        parent = list(range(len(terminals)))
        rank = [0] * len(terminals)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> bool:
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
            return True

        edges: list[tuple[float, int, int]] = []
        for i in range(len(terminals)):
            for j in range(i + 1, len(terminals)):
                cost = self._pair_cost(terminals[i], terminals[j], pair_cost_provider)
                edges.append((cost, i, j))

        total_cost = 0.0
        edges_used = 0
        for cost, i, j in sorted(edges, key=lambda item: item[0]):
            if union(i, j):
                total_cost += cost
                edges_used += 1
                if edges_used == len(terminals) - 1:
                    return total_cost
        return inf

    def _pair_cost(
        self,
        a: GridPoint,
        b: GridPoint,
        pair_cost_provider: PairCostProvider,
    ) -> float:
        """Lookup or estimate the routing cost between two terminal cells."""
        if pair_cost_provider is None:
            return float(_manhattan(a, b))

        if callable(pair_cost_provider):
            return float(pair_cost_provider(a, b))

        direct_key = (a, b)
        reverse_key = (b, a)
        if direct_key in pair_cost_provider:
            return float(pair_cost_provider[direct_key])
        if reverse_key in pair_cost_provider:
            return float(pair_cost_provider[reverse_key])
        return float(_manhattan(a, b))

    def _reference_wiring_cost(self, state: RoomState, *, terminal_count: int) -> float:
        """
        Build a coarse upper-bound cost used to normalize MST routing cost.

        The bound is proportional to room diameter times the number of required
        tree edges, which keeps the resulting wiring score in [0, 1].
        """
        room_cells = np.argwhere(state.room_mask)
        if room_cells.size == 0:
            rows, cols = state.shape
            room_diameter = float(rows + cols)
        else:
            min_row, min_col = room_cells.min(axis=0)
            max_row, max_col = room_cells.max(axis=0)
            room_diameter = float((max_row - min_row) + (max_col - min_col) + 1.0)
        return max(1.0, room_diameter * max(terminal_count - 1, 1))


def _mask_positions(mask: np.ndarray) -> list[GridPoint]:
    """Convert a boolean mask to a row/col coordinate list."""
    coords = np.argwhere(mask)
    return [tuple(map(int, coord)) for coord in coords]


def _manhattan(a: GridPoint, b: GridPoint) -> int:
    """Compute Manhattan distance between two grid points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _demo() -> None:
    """
    Minimal smoke test for the reward module.

    The demo uses an encoded matrix with:
        - one door
        - one switch
        - two lamps
    """
    matrix = np.array(
        [
            [2, 1, 1, 1],
            [3, 1, 4, 1],
            [1, 1, 1, 1],
            [1, 4, 1, 1],
        ],
        dtype=np.int32,
    )
    state = RoomState.from_encoded_matrix(matrix)
    config = RewardConfig()
    calculator = RewardCalculator(config)
    step = calculator.calculate_step_reward(state)
    terminal = calculator.calculate_terminal_reward(state)
    print("step total:", round(step.total, 4), step.diagnostics)
    print("terminal total:", round(terminal.total, 4), terminal.diagnostics)


if __name__ == "__main__":
    _demo()
