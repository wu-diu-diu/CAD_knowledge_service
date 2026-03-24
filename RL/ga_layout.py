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
import yaml

from env import EnvironmentConfig, SingleRoomLightingEnv
from reward import RewardBreakdown, RewardConfig, RoomState
from visualize import save_room_grid_image, save_padded_room_image


matplotlib.use("Agg")
import matplotlib.pyplot as plt


GridPoint = tuple[int, int]


@dataclass
class GAConfig:
    """Genetic algorithm hyper-parameters for fixed-count lamp placement."""

    population_size: int = 64  ## 种群规模，每个种群表示一个完整的灯具布局方案
    generations: int = 200  ## 最大迭代次数，算法会在达到这个代数后停止
    tournament_size: int = 3  ## 锦标赛选择的竞争者数量，较小的值增加选择压力，较大的值增加多样性
    elite_count: int = 4  ## 每代保留的精英个体数量，直接进入下一代，保持优秀基因
    # --- 变异退火参数（取代旧的 mutation_rate / mutation_count 标量） ---
    mutation_rate_start: float = 0.5   ## 初始变异概率（早期大探索）
    mutation_rate_end: float = 0.05    ## 终止变异概率（后期微调）
    mutation_count_start: int = 2      ## 初始每次变异替换的基因数
    mutation_count_end: int = 1        ## 终止每次变异替换的基因数
    # --- 多样性保护 ---
    diversity_min_hamming: int = 1     ## 子代与所有精英的最小汉明距离阈值，低于此值则替换为随机个体
    patience: int = 30  ## 早停耐心值，如果连续这么多代没有显著改进，就停止算法
    min_delta: float = 1e-4  ## 最小改进阈值，只有当新最佳适应度比当前最佳适应度高出至少这个值时，才认为是改进
    seed: int = 42  ## 随机种子，确保结果可复现
    log_every_generations: int = 10 ## 每隔多少代打印一次日志


@dataclass
class RoomConfig:
    """Room-selection settings loaded from config.yaml."""

    json_path: str = "RL/test_room/origin_room/unregular.json"
    room_name: str = "热量室准备间"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level mapping in {config_path}")
    return payload


def create_timestamped_output_dir(base_output_dir: Path) -> Path:
    base_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = base_output_dir / f"ga_{timestamp}"
    suffix = 1
    while run_output_dir.exists():
        run_output_dir = base_output_dir / f"ga_{timestamp}_{suffix:02d}"
        suffix += 1
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return run_output_dir


def plot_fitness_curve(history: list[dict[str, Any]], output_path: Path) -> Path:
    """Plot best and mean fitness over generations."""
    generations = [int(item["generation"]) for item in history]
    best_values = np.asarray([float(item["best_fitness"]) for item in history], dtype=np.float32)
    mean_values = np.asarray([float(item["mean_fitness"]) for item in history], dtype=np.float32)

    plt.figure(figsize=(10, 5))
    plt.plot(generations, best_values, label="best_fitness", linewidth=2.0)
    plt.plot(generations, mean_values, label="mean_fitness", linewidth=2.0)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GA Layout Fitness Curve")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_score_curve(history: list[dict[str, Any]], output_path: Path) -> Path:
    """Plot best-individual raw score trends over generations."""
    generations = [int(item["generation"]) for item in history]
    potential = np.asarray([float(item["potential_reduction_score"]) for item in history], dtype=np.float32)
    alignment = np.asarray([float(item["alignment_score"]) for item in history], dtype=np.float32)
    wiring = np.asarray([float(item["wiring_score"]) for item in history], dtype=np.float32)
    mst_cost = np.asarray([float(item["mst_cost"]) for item in history], dtype=np.float32)

    plt.figure(figsize=(11, 5.5))
    plt.plot(generations, potential, label="potential_reduction_score", linewidth=2.0)
    plt.plot(generations, alignment, label="alignment_score", linewidth=2.0)
    plt.plot(generations, wiring, label="wiring_score", linewidth=2.0)
    plt.plot(generations, mst_cost, label="mst_cost", linewidth=2.0)
    plt.xlabel("Generation")
    plt.ylabel("Score / Cost")
    plt.title("GA Layout Score Trends")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


class GALayoutOptimizer:
    """Fixed-count lamp layout optimizer using GA and RL reward as fitness."""

    def __init__(
        self,
        env: SingleRoomLightingEnv,
        ga_config: GAConfig,
        output_dir: Path,
    ) -> None:
        self.env = env
        self.ga_config = ga_config
        self.output_dir = output_dir
        self.reward_calculator = env.reward_calculator
        self.target_lamp_count = int(env.config.target_lamp_count)
        self.legal_points = self._legal_points()
        self.legal_ids = [self._point_to_id(point) for point in self.legal_points]
        if len(self.legal_ids) < self.target_lamp_count:
            raise ValueError("Not enough legal cells to place the target number of lamps.")

    def _legal_points(self) -> list[GridPoint]:
        coords = np.argwhere(self.env.original_placeable_mask)
        return [(int(coord[0]), int(coord[1])) for coord in coords]

    def _point_to_id(self, point: GridPoint) -> int:
        return int(point[0] * self.env.grid_cols + point[1])

    def _id_to_point(self, point_id: int) -> GridPoint:
        return int(point_id // self.env.grid_cols), int(point_id % self.env.grid_cols)

    def _random_genome(self) -> tuple[int, ...]:
        chosen = random.sample(self.legal_ids, self.target_lamp_count)
        chosen.sort()
        return tuple(chosen)

    def _repair_genome(self, genome: list[int] | tuple[int, ...]) -> tuple[int, ...]:
        unique: list[int] = []
        used = set()
        for gene in genome:
            if gene in self.legal_ids and gene not in used:
                unique.append(gene)
                used.add(gene)

        available = [gene for gene in self.legal_ids if gene not in used]
        random.shuffle(available)
        while len(unique) < self.target_lamp_count:
            if not available:
                raise RuntimeError("Unable to repair genome: insufficient legal cells.")
            gene = available.pop()
            unique.append(gene)
            used.add(gene)

        unique = unique[: self.target_lamp_count]
        unique.sort()
        return tuple(unique)

    def _build_state(self, genome: tuple[int, ...]) -> RoomState:
        lamp_mask = np.zeros_like(self.env.original_placeable_mask, dtype=bool)  ## 创造一个全为 False 的灯具掩码，形状和房间网格相同，placeable_mask中True为可放置位置，False为不可放置位置
        for gene in genome:
            lamp_mask[self._id_to_point(gene)] = True
        return RoomState.from_channels(
            room_mask=self.env.original_room_mask,
            lamp_mask=lamp_mask,
            switch_mask=self.env.original_switch_mask,
            door_mask=self.env.original_door_mask,
        )
    def _potential_score(self, state: RoomState) -> float:
        """势能奖励：布置后势能越低，得分越高，归一化到 [0, 1]。"""
        initial_potential = self.reward_calculator.initial_potential(state)
        if initial_potential <= 0.0:
            return 1.0
        current_potential = self.reward_calculator.potential(state)
        return float(np.clip(1.0 - current_potential / initial_potential, 0.0, 1.0))

    ## 计算一个基因组的适应度。
    ## fitness = 势能奖励（一次性）+ 终局对齐/布线奖励
    def evaluate_genome(self, genome: tuple[int, ...]) -> tuple[float, RewardBreakdown]:
        state = self._build_state(genome)
        initial_potential = self.reward_calculator.initial_potential(state)

        potential_normalized = self._potential_score(state)
        potential_term = self.reward_calculator.config.potential_coef * potential_normalized

        terminal_bd = self.reward_calculator.calculate_terminal_reward(
            state,
            pair_cost_provider=self.env.pair_cost,
            initial_potential=initial_potential,
        )
        total_fitness = potential_term + terminal_bd.total

        bd = RewardBreakdown(
            total=total_fitness,
            potential_reduction_normalized=potential_normalized,
            potential_reduction_item=potential_term,
            invalid_penalty=0.0,
            alignment_normalized=terminal_bd.alignment_normalized,
            alignment_term=terminal_bd.alignment_term,
            wiring_normalized=terminal_bd.wiring_normalized,
            wiring_term=terminal_bd.wiring_term,
            mst_cost=terminal_bd.mst_cost,
            terminal_bonus=terminal_bd.terminal_bonus,
        )
        return total_fitness, bd

    def _tournament_select(self, scored_population: list[tuple[tuple[int, ...], float, RewardBreakdown]]) -> tuple[int, ...]:
        competitors = random.sample(scored_population, k=min(self.ga_config.tournament_size, len(scored_population)))
        winner = max(competitors, key=lambda item: item[1])
        return winner[0]

    def _crossover(self, parent_a: tuple[int, ...], parent_b: tuple[int, ...]) -> tuple[int, ...]:
        """顺序交叉（OX）变体：随机从parent_a取一段区间基因，剩余从parent_b按顺序补充。
        真正继承两个父代的空间信息，而不是退化为随机填充。"""
        n = self.target_lamp_count
        # 随机选取parent_a中的一段区间
        lo = random.randint(0, n - 1)
        hi = random.randint(lo + 1, n)
        child_genes: list[int] = list(parent_a[lo:hi])
        child_set = set(child_genes)
        # 按parent_b的顺序补充不重复的基因
        for gene in parent_b:
            if gene not in child_set:
                child_genes.append(gene)
                child_set.add(gene)
            if len(child_genes) >= n:
                break
        return self._repair_genome(child_genes)

    def _mutate(self, genome: tuple[int, ...], *, annealing_t: float = 0.0) -> tuple[int, ...]:
        """带退火的变异：annealing_t ∈ [0, 1]，0=早期大探索，1=后期微调。
        变异率和变异步长均随 t 线性衰减。"""
        cfg = self.ga_config
        rate = cfg.mutation_rate_start * (1.0 - annealing_t) + cfg.mutation_rate_end * annealing_t
        if random.random() >= rate:
            return genome

        count = max(1, round(cfg.mutation_count_start * (1.0 - annealing_t) + cfg.mutation_count_end * annealing_t))
        replace_count = min(count, self.target_lamp_count)
        genes = list(genome)
        mutate_indices = random.sample(range(self.target_lamp_count), k=replace_count)
        used = set(genes)
        available = [gene for gene in self.legal_ids if gene not in used]
        random.shuffle(available)
        for idx in mutate_indices:
            if not available:
                break
            used.discard(genes[idx])
            genes[idx] = available.pop()
            used.add(genes[idx])
        return self._repair_genome(genes)

    def _initialize_population(self) -> list[tuple[int, ...]]:
        return [self._random_genome() for _ in range(self.ga_config.population_size)]

    def _save_best_layout_image(
        self,
        genome: tuple[int, ...],
        breakdown: RewardBreakdown,
        output_path: Path,
    ) -> Path:
        encoded = np.zeros_like(self.env.original_matrix, dtype=np.int32)
        encoded[self.env.original_placeable_mask] = 1
        encoded[self.env.original_door_mask] = 2
        encoded[self.env.original_switch_mask] = 3
        for gene in genome:
            encoded[self._id_to_point(gene)] = 4
        room_title = (
            f"{self.env.room_name} | fitness={breakdown.total:.3f} "
            f"p={breakdown.potential_reduction_normalized:.2f} "
            f"a={breakdown.alignment_normalized:.2f} "
            f"w={breakdown.wiring_normalized:.2f}"
        )
        return save_room_grid_image(encoded, output_path, cell_size=32, room_name=room_title)

    def run(self) -> dict[str, Any]:
        population = self._initialize_population()
        history: list[dict[str, Any]] = []
        best_genome: tuple[int, ...] | None = None
        best_breakdown: RewardBreakdown | None = None
        best_fitness = float("-inf")
        stagnant_generations = 0  ## 连续未改进的代数计数器

        for generation in range(1, self.ga_config.generations + 1):
            scored_population = []
            for genome in population:
                fitness, breakdown = self.evaluate_genome(genome)
                scored_population.append((genome, fitness, breakdown))

            scored_population.sort(key=lambda item: item[1], reverse=True)
            generation_best_genome, generation_best_fitness, generation_best_breakdown = scored_population[0]
            mean_fitness = float(np.mean([item[1] for item in scored_population]))

            history.append(
                {
                    "generation": generation,
                    "best_fitness": generation_best_fitness,
                    "mean_fitness": mean_fitness,
                    "potential_reduction_score": float(generation_best_breakdown.potential_reduction_normalized),
                    "alignment_score": float(generation_best_breakdown.alignment_normalized),
                    "wiring_score": float(generation_best_breakdown.wiring_normalized),
                    "mst_cost": float(generation_best_breakdown.mst_cost),
                }
            )

            if generation_best_fitness > best_fitness + self.ga_config.min_delta:
                best_fitness = generation_best_fitness
                best_genome = generation_best_genome
                best_breakdown = generation_best_breakdown
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            if generation % self.ga_config.log_every_generations == 0 or generation == 1:
                print(
                    f"[ga] generation={generation:04d} "
                    f"best={generation_best_fitness:8.3f} "
                    f"mean={mean_fitness:8.3f} "
                    f"p={generation_best_breakdown.potential_reduction_normalized:5.2f} "
                    f"a={generation_best_breakdown.alignment_normalized:5.2f} "
                    f"w={generation_best_breakdown.wiring_normalized:5.2f}"
                )

            if stagnant_generations >= self.ga_config.patience:
                break

            # 计算当前代的退火进度 t ∈ [0, 1]
            annealing_t = (generation - 1) / max(self.ga_config.generations - 1, 1)

            elites = [item[0] for item in scored_population[: self.ga_config.elite_count]]
            next_population = list(elites)
            while len(next_population) < self.ga_config.population_size:
                parent_a = self._tournament_select(scored_population)
                parent_b = self._tournament_select(scored_population)
                child = self._crossover(parent_a, parent_b)
                child = self._mutate(child, annealing_t=annealing_t)

                # 多样性保护：若子代与所有精英的汉明距离均低于阈值，则替换为随机个体
                min_hamming = self.ga_config.diversity_min_hamming
                if min_hamming > 0 and elites:
                    child_set = set(child)
                    too_close = all(
                        len(child_set.symmetric_difference(set(elite))) < min_hamming
                        for elite in elites
                    )
                    if too_close:
                        child = self._random_genome()

                next_population.append(child)
            population = next_population

        if best_genome is None or best_breakdown is None:
            raise RuntimeError("GA finished without producing a best individual.")

        best_layout_path = self._save_best_layout_image(best_genome, best_breakdown, self.output_dir / "ga_best_layout.png")
        fitness_curve_path = plot_fitness_curve(history, self.output_dir / "ga_fitness_curve.png")
        score_curve_path = plot_score_curve(history, self.output_dir / "ga_score_curve.png")

        summary = {
            "ga_config": asdict(self.ga_config),
            "room_name": self.env.room_name,
            "best_fitness": best_fitness,
            "best_genome": list(best_genome),
            "best_positions": [self._id_to_point(gene) for gene in best_genome],
            "best_diagnostics": {
                "potential_reduction_score": best_breakdown.potential_reduction_normalized,
                "alignment_score": best_breakdown.alignment_normalized,
                "wiring_score": best_breakdown.wiring_normalized,
                "mst_cost": best_breakdown.mst_cost,
            },
            "history": history,
            "best_layout_path": str(best_layout_path),
            "fitness_curve_path": str(fitness_curve_path),
            "score_curve_path": str(score_curve_path),
        }
        return summary


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "RL" / "config.yaml"
    config_payload = load_yaml_config(config_path)

    ga_config = GAConfig(**config_payload.get("ga", {}))
    room_cfg = RoomConfig(**config_payload.get("room", {}))
    reward_cfg = RewardConfig(**config_payload.get("reward", {}))
    env_cfg = EnvironmentConfig(
        **config_payload.get("environment", {}),
        reward_config=reward_cfg,
    )
    reward_cfg.target_lamp_count = env_cfg.target_lamp_count

    output_dir = create_timestamped_output_dir(repo_root / "RL" / "GA")
    shutil.copy2(config_path, output_dir / "config.yaml")
    set_seed(ga_config.seed)

    env = SingleRoomLightingEnv.from_json(
        repo_root / room_cfg.json_path,
        room_name=room_cfg.room_name,
        config=env_cfg,
    )

    # # Save padded room visualization
    # padded_room_path = output_dir / "padded_room.png"
    # save_padded_room_image(
    #     env.original_matrix,
    #     env.padded_size,
    #     padded_room_path,
    #     cell_size=32,
    #     room_name=env.room_name,
    # )
    # print(f"[ga] padded room visualization saved to {padded_room_path}")

    optimizer = GALayoutOptimizer(env=env, ga_config=ga_config, output_dir=output_dir)
    summary = optimizer.run()
    summary["config_path"] = str(config_path)
    summary["output_dir"] = str(output_dir)

    summary_path = output_dir / "ga_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[ga] run output dir: {output_dir}")
    print(f"[ga] summary saved to {summary_path}")
    print(f"[ga] best_fitness={summary['best_fitness']:.3f}")
    print(f"[ga] best_positions={summary['best_positions']}")


if __name__ == "__main__":
    main()
