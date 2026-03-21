from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import inf
from typing import Callable, Iterable, Mapping

import numpy as np


GridPoint = tuple[int, int]
PairCostCallable = Callable[[GridPoint, GridPoint], float]
PairCostMapping = Mapping[tuple[GridPoint, GridPoint], float]
PairCostProvider = PairCostCallable | PairCostMapping | None


@dataclass
class RewardWeights:
    """Top-level weights for the layered reward components."""

    uniformity: float = 1.0
    rules: float = 1.0
    wiring: float = 1.0
    terminal: float = 1.0


@dataclass
class RewardConfig:
    """
    Reward hyper-parameters derived from `布局布线优化.md`.

    The main formula is:
        R = 均匀性奖励 + 规范奖励 + 布线奖励 + 终局奖励
    """

    uniformity_scale: float = 1.0  ## 均匀性奖励的缩放因子，控制其在总奖励中的影响力
    alignment_reward_scale: float = 1.5  ## 对齐奖励的缩放因子，鼓励灯具布局更加均匀
    wiring_cost_scale: float = 0.01  ## 布线奖励的缩放因子，将MST成本转换为奖励值
    invalid_action_penalty: float = 10.0  ## 非法动作的惩罚，例如试图在墙上放灯

    terminal_wiring_scale: float = 0.05  ## 终局布线奖励的缩放因子，通常比step中的更高以强调最终布局质量
    terminal_success_bonus: float = 5.0  ## 满足所有约束的终局奖励，鼓励智能体找到完全合法且高质量的布局
    terminal_centering_scale: float = 5.0  ## 终局时鼓励灯具布局中心靠近房间中心的奖励系数
    target_lamp_count: int | None = None

    weights: RewardWeights = field(default_factory=RewardWeights)


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
class RewardBreakdown:
    """Detailed reward report for one step or terminal evaluation."""

    total: float
    illumination: float
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
        - uniformity reward
        - rule/constraint reward
        - wiring reward from MST cost
        - optional terminal reward
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
        uniformity_reward = self.uniformity_reward(state)
        rule_reward, rule_diagnostics = self.rule_reward(state)
        wiring_reward, mst_cost = self.wiring_reward(state, pair_cost_provider=pair_cost_provider)
        invalid_penalty = -self.config.invalid_action_penalty if invalid_action else 0.0
        rules_total = rule_reward + invalid_penalty

        total = (
            self.config.weights.uniformity * uniformity_reward
            + self.config.weights.rules * rules_total
            + self.config.weights.wiring * wiring_reward
        )

        diagnostics: dict[str, float | int | bool | list[GridPoint]] = {
            "lamp_count": len(state.lamp_positions),
            "mst_cost": mst_cost,
            "invalid_action": invalid_action,
        }
        diagnostics.update(rule_diagnostics)

        return RewardBreakdown(
            total=total,
            illumination=0.0,
            uniformity=uniformity_reward,
            rules=rules_total,
            wiring=wiring_reward,
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
        Compute an end-of-episode reward.

        The doc describes terminal reward as running `MST + A*` once more.
        Here that is represented as:
            - a final wiring reward term based on the MST cost
            - a centering reward encouraging the lamp-layout centroid to stay
              near the room centroid
            - an optional success bonus when constraints are met
        """
        step = self.calculate_step_reward(state, pair_cost_provider=pair_cost_provider) ## 先计算一次step reward，主要是为了获得当前状态的MST成本和诊断信息，这些信息对于计算终局奖励是必要的。
        success = (  ## 判断当前状态是否满足成功条件，即灯具数量是否达到了目标数量（如果有设定的话）。如果没有设定目标数量，那么只要有灯具就算成功。这是为了决定是否给予终局成功奖励。
            int(step.diagnostics.get("lamp_count", 0)) == int(self.config.target_lamp_count)
            if self.config.target_lamp_count is not None
            else int(step.diagnostics.get("lamp_count", 0)) > 0
        )
        centering_score, lamp_center_distance = self.layout_centering_score(state)
        terminal_reward = (  ## 计算终局奖励，主要包括两个部分：一个是基于MST成本的布线奖励，另一个是满足成功条件时的额外奖励。MST成本越高，布线奖励越低（因为它是负值），而满足成功条件则会增加一个固定的奖励值。
            -self.config.terminal_wiring_scale * float(step.diagnostics["mst_cost"])
            + self.config.terminal_centering_scale * centering_score
            + (self.config.terminal_success_bonus if success else 0.0)
        )
        total = step.total + self.config.weights.terminal * terminal_reward
        step.total = total
        step.terminal = terminal_reward
        step.diagnostics["terminal_success"] = success
        step.diagnostics["centering_score"] = centering_score
        step.diagnostics["lamp_center_distance"] = lamp_center_distance
        return step

    def uniformity_reward(self, state: RoomState) -> float:
        """
        Use nearest-neighbor distance variance as a uniformity reward.

        Smaller variance means lamps are spaced more evenly, so the reward is the
        negative normalized variance.
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
        return -self.config.uniformity_scale * normalized_variance  ## 最终奖励是负的归一化方差，表示方差越小（即分布越均匀），奖励越高。同时乘以一个缩放因子uniformity_scale来调整这个奖励在总奖励中的权重。

    def rule_reward(self, state: RoomState) -> tuple[float, dict[str, float | int | bool | list[GridPoint]]]:
        """
        Compute rule reward.

        The current soft constraint keeps row/column alignment preference.
        """
        lamps = state.lamp_positions

        alignment_score = self.alignment_score(lamps)
        alignment_reward = self.config.alignment_reward_scale * alignment_score

        reward = alignment_reward
        diagnostics: dict[str, float | int | bool | list[GridPoint]] = {
            "alignment_score": alignment_score,
        }
        return reward, diagnostics

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

    def layout_centering_score(self, state: RoomState) -> tuple[float, float]:
        """
        Measure how close the lamp-layout centroid is to the room centroid.

        Returns:
            score: normalized score in [0, 1], higher is better
            distance: Euclidean distance between lamp centroid and room centroid
        """
        lamps = state.lamp_positions
        if not lamps:
            return 0.0, inf

        room_cells = np.argwhere(state.room_mask)
        if room_cells.size == 0:
            return 0.0, inf

        room_center = room_cells.mean(axis=0)
        lamp_center = np.asarray(lamps, dtype=np.float32).mean(axis=0)
        distance = float(np.linalg.norm(lamp_center - room_center))

        farthest_room_distance = float(np.max(np.linalg.norm(room_cells - room_center, axis=1)))  ## 计算房间内所有单元格到房间中心的最大距离，作为距离归一化的基准。这样可以确保无论房间大小如何，距离得分都在0到1之间，其中0表示灯具布局中心与房间中心重合，1表示灯具布局中心位于房间内最远的点上。
        normalization = max(farthest_room_distance, 1.0)
        score = max(0.0, 1.0 - distance / normalization)
        return score, distance

    def wiring_reward(
        self,
        state: RoomState,
        *,
        pair_cost_provider: PairCostProvider = None,
    ) -> tuple[float, float]:
        """
        Compute the wiring reward from the MST cost over switch + lamp terminals.

        If a precomputed pair-cost matrix `M` exists, pass it through
        `pair_cost_provider`. Otherwise the fallback is Manhattan distance.
        """
        terminals = state.switch_positions + state.lamp_positions
        if len(terminals) <= 1:
            return 0.0, 0.0

        mst_cost = self._mst_total_cost(terminals, pair_cost_provider=pair_cost_provider)
        if mst_cost == inf:
            return -self.config.invalid_action_penalty, inf
        return -self.config.wiring_cost_scale * mst_cost, mst_cost

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
