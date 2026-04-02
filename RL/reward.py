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
    """Single-layer reward coefficients for lamp-placement RL."""

    potential_coef: float = 2.0
    alignment_coef: float = 2.0  ## 提高默认值，强化对齐优先级
    wiring_coef: float = 1.0
    grid_regularity_coef: float = 0.0  ## 默认禁用，避免与对齐目标竞争
    alignment_tolerance: float = 2.0  ## 软对齐容忍度：差1格得0.5分，差2格得0分
    invalid_action_penalty: float = 10.0
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
    """

    room_mask: np.ndarray
    lamp_mask: np.ndarray
    switch_mask: np.ndarray
    door_mask: np.ndarray
    placeable_mask: np.ndarray

    @classmethod
    def from_encoded_matrix(cls, matrix: list[list[int]] | np.ndarray) -> "RoomState":
        """Build a room state from the repo's encoded integer matrix."""
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


@dataclass
class RewardBreakdown:
    """奖励分解项"""
    total: float
    potential_reduction_normalized: float
    potential_reduction_item: float
    invalid_penalty: float
    alignment_normalized: float
    alignment_term: float
    wiring_normalized: float
    wiring_term: float
    mst_cost: float
    terminal_bonus: float


class RewardCalculator:
    """
    Reward calculator for RL lamp placement.

    Step reward:
        normalized potential reduction + invalid-action penalty

    Terminal reward:
        normalized alignment reward + normalized wiring reward
    """

    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def calculate_step_reward(
        self,
        state: RoomState,
        *,
        prev_potential: float,
        invalid_action: bool = False,
    ) -> RewardBreakdown:
        """Compute the per-step reward from potential reduction only."""
        current_potential = self.potential(state)
        reduction = float(prev_potential) - current_potential
        normalized_reduction = float(np.clip(reduction / prev_potential, -1.0, 1.0))

        invalid_penalty = -self.config.invalid_action_penalty if invalid_action else 0.0
        potential_term = self.config.potential_coef * normalized_reduction
        total = potential_term + invalid_penalty

        return RewardBreakdown(
            total=total,
            potential_reduction_normalized=normalized_reduction,
            potential_reduction_item=potential_term,
            invalid_penalty=invalid_penalty,
            alignment_normalized=0.0,
            alignment_term=0.0,
            wiring_normalized=0.0,
            wiring_term=0.0,
            mst_cost=0.0,
            terminal_bonus=0.0,
        )

    def calculate_terminal_reward(
        self,
        state: RoomState,
        *,
        pair_cost_provider: PairCostProvider = None,
        initial_potential: float | None = None,
        potential_quality_threshold: float = 0.15,
    ) -> RewardBreakdown:
        """Compute terminal reward from alignment and wiring only.

        If `initial_potential` is provided, the terminal bonus is gated by a
        quality threshold: the current potential must have dropped below
        `potential_quality_threshold * initial_potential` before alignment and
        wiring rewards are granted.  This prevents the agent from collecting
        terminal bonuses by placing lamps quickly in poor positions.
        设置这个阈值可以鼓励代理在达到终端奖励之前先进行有效的潜力降低，从而引导其学习更合理的布局策略，而不是仅仅追求快速放置灯具。
        """
        current_potential = self.potential(state)
        quality_ok = (
            initial_potential is None
            or initial_potential <= 0.0
            or current_potential <= potential_quality_threshold * initial_potential
        )

        if not quality_ok:
            return RewardBreakdown(
                total=0.0,
                potential_reduction_normalized=0.0,
                potential_reduction_item=0.0,
                invalid_penalty=0.0,
                alignment_normalized=0.0,
                alignment_term=0.0,
                wiring_normalized=0.0,
                wiring_term=0.0,
                mst_cost=0.0,
                terminal_bonus=0.0,
            )

        alignment_normalized = self.soft_alignment_score(state.lamp_positions)
        wiring_normalized, mst_cost = self.wiring_score(state, pair_cost_provider=pair_cost_provider)
        grid_regularity_normalized = self.grid_regularity_score(state)
        alignment_term = self.config.alignment_coef * alignment_normalized
        wiring_term = self.config.wiring_coef * wiring_normalized
        grid_regularity_term = self.config.grid_regularity_coef * grid_regularity_normalized
        total = alignment_term + wiring_term + grid_regularity_term

        return RewardBreakdown(
            total=total,
            potential_reduction_normalized=0.0,
            potential_reduction_item=0.0,
            invalid_penalty=0.0,
            alignment_normalized=alignment_normalized,
            alignment_term=alignment_term,
            wiring_normalized=wiring_normalized,
            wiring_term=wiring_term,
            mst_cost=mst_cost,
            terminal_bonus=total,
        )

    def initial_potential(self, state: RoomState) -> float:
        """
        Return the finite initial potential for the empty-lamp layout.

        With no lamps, the "distance to nearest lamp" is undefined. We use the
        squared room diagonal as a finite upper bound for every placeable cell,
        producing a deterministic baseline used both by PPO and GA.
        """
        rows, cols = state.shape
        max_dist_sq = float(max((rows - 1) ** 2 + (cols - 1) ** 2, 1))
        placeable_count = int(np.count_nonzero(state.placeable_mask))
        return float(placeable_count) * max_dist_sq

    def potential(self, state: RoomState) -> float:
        """
        Compute layout potential:
            sum over all placeable cells of squared Euclidean distance to the
            nearest lamp.
        """
        placeable_cells = np.argwhere(state.placeable_mask)
        if placeable_cells.size == 0:
            return 0.0

        lamps = state.lamp_positions
        if not lamps:
            return self.initial_potential(state)

        lamp_coords = np.asarray(lamps, dtype=np.float32)
        cell_coords = placeable_cells.astype(np.float32)
        diff = cell_coords[:, None, :] - lamp_coords[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        nearest_dist_sq = np.min(dist_sq, axis=1)
        return float(np.sum(nearest_dist_sq))

    def alignment_score(self, lamps: Iterable[GridPoint]) -> float:
        """
        Measure whether lamps form shared rows/columns (hard binary metric).

        The score is normalized to [0, 1].
        """
        lamp_list = list(lamps)
        if len(lamp_list) < 2:
            return 1.0

        row_counts = Counter(r for r, _ in lamp_list)
        col_counts = Counter(c for _, c in lamp_list)
        row_shared = sum(1 for r, _ in lamp_list if row_counts[r] > 1) / len(lamp_list)
        col_shared = sum(1 for _, c in lamp_list if col_counts[c] > 1) / len(lamp_list)
        return float(np.clip(0.5 * (row_shared + col_shared), 0.0, 1.0))

    def soft_alignment_score(self, lamps: Iterable[GridPoint]) -> float:
        """
        Soft alignment: reward lamp pairs that are close to sharing a row or column.

        For each pair (i, j):
            dr = |row_i - row_j|
            dc = |col_i - col_j|
            row_close = max(0, 1 - dr / tolerance)
            col_close = max(0, 1 - dc / tolerance)
            pair_score = max(row_close, col_close)  # aligned in either axis counts

        Final score = mean of all pair scores, in [0, 1].

        With tolerance=2:
            exact same row/col  -> 1.0
            1 cell apart        -> 0.5
            2+ cells apart      -> 0.0
        """
        lamp_list = list(lamps)
        n = len(lamp_list)
        if n < 2:
            return 1.0

        tolerance = self.config.alignment_tolerance
        total = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                dr = abs(lamp_list[i][0] - lamp_list[j][0])
                dc = abs(lamp_list[i][1] - lamp_list[j][1])
                row_close = max(0.0, 1.0 - dr / tolerance)
                col_close = max(0.0, 1.0 - dc / tolerance)
                total += max(row_close, col_close)
                pairs += 1

        return float(total / pairs)

    def grid_regularity_score(self, state: RoomState) -> float:
        """
        Measure how well lamps snap to a regular grid.

        Finds the best-fit uniform grid (spacing derived from lamp count and
        room size), then scores each lamp by how close it is to the nearest
        grid point.  Score = 1 - mean_normalized_offset, in [0, 1].

        Intuition: 4 lamps in a 20x20 room → ideal spacing ~10 cells.
        A lamp sitting exactly on a grid point contributes 0 offset; one
        displaced by half a spacing contributes 0.5.
        """
        lamps = state.lamp_positions
        n = len(lamps)
        if n < 2:
            return 1.0

        rows, cols = state.shape
        # Estimate ideal grid spacing from lamp count and room area
        area = float(rows * cols)
        spacing = float(np.sqrt(area / n))
        if spacing < 1.0:
            return 1.0

        lamp_arr = np.asarray(lamps, dtype=np.float32)
        # For each lamp, compute its offset from the nearest grid point
        # Grid points: multiples of `spacing` offset by the grid origin.
        # We optimise the grid origin (ox, oy) in [0, spacing) to minimise
        # total offset — a simple 1-D problem solved by taking the median
        # fractional position.
        frac_r = (lamp_arr[:, 0] % spacing) / spacing  # in [0, 1)
        frac_c = (lamp_arr[:, 1] % spacing) / spacing

        # Circular median: best origin minimises sum of min(frac, 1-frac)
        # Equivalent to: offset = min(frac, 1-frac) for each lamp
        offset_r = np.minimum(frac_r, 1.0 - frac_r)
        offset_c = np.minimum(frac_c, 1.0 - frac_c)
        mean_offset = float(np.mean(offset_r + offset_c)) / 2.0  # normalise to [0, 0.5]

        score = 1.0 - 2.0 * mean_offset  # maps [0, 0.5] → [1, 0]
        return float(np.clip(score, 0.0, 1.0))

    def wiring_score(
        self,
        state: RoomState,
        *,
        pair_cost_provider: PairCostProvider = None,
    ) -> tuple[float, float]:
        """
        Normalize MST routing cost into a [0, 1] reward score.

        Uses relative normalization against the worst-case MST estimate
        (n_edges * room_diagonal) so the score stays meaningful for large rooms.

        `mst_cost` is the raw wiring cost used for diagnostics.
        `wiring_score` is normalized to [0, 1] used in the reward.
        """
        terminals = state.switch_positions + state.lamp_positions
        if len(terminals) <= 1:
            return 1.0, 0.0

        mst_cost = self._mst_total_cost(terminals, pair_cost_provider=pair_cost_provider)
        if mst_cost == inf:
            return 0.0, inf

        rows, cols = state.shape
        diagonal = float(np.sqrt((rows - 1) ** 2 + (cols - 1) ** 2)) if rows > 1 or cols > 1 else 1.0
        max_mst = (len(terminals) - 1) * diagonal
        score = 1.0 - float(mst_cost) / max_mst if max_mst > 0 else 1.0
        return float(np.clip(score, 0.0, 1.0)), float(mst_cost)

    def _mst_total_cost(
        self,
        terminals: list[GridPoint],
        *,
        pair_cost_provider: PairCostProvider = None,
    ) -> float:
        """Compute MST cost using Kruskal over the terminal complete graph."""
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
    """Minimal smoke test for the reward module."""
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
    config = RewardConfig(target_lamp_count=2)
    calculator = RewardCalculator(config)
    initial_potential = calculator.initial_potential(state)
    step = calculator.calculate_step_reward(state, prev_potential=initial_potential)
    terminal = calculator.calculate_terminal_reward(state)
    print("step total:", round(step.total, 4), "pot_norm:", round(step.potential_reduction_normalized, 4))
    print("terminal total:", round(terminal.total, 4), "align:", round(terminal.alignment_normalized, 4), "wire:", round(terminal.wiring_normalized, 4))


if __name__ == "__main__":
    _demo()
