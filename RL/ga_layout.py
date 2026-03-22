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
from visualize import save_room_grid_image


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
    mutation_rate: float = 0.3  ## 每个子代进行变异的概率，较高的值增加探索，较低的值增加稳定性
    mutation_count: int = 1  ## 每次变异中替换的基因数量，较大的值增加变异幅度，较小的值增加微调能力
    patience: int = 30  ## 早停耐心值，如果连续这么多代没有显著改进，就停止算法
    min_delta: float = 1e-4  ## 最小改进阈值，只有当新最佳适应度比当前最佳适应度高出至少这个值时，才认为是改进
    seed: int = 42  ## 随机种子，确保结果可复现
    log_every_generations: int = 10 ## 每隔多少代打印一次日志


@dataclass
class RoomConfig:
    """Room-selection settings loaded from config.yaml."""

    json_path: str = "RL/test_room/test_room.json"
    room_name: str = "办公室1"


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
    uniformity = np.asarray([float(item["uniformity_score"]) for item in history], dtype=np.float32)
    illum_centroid = np.asarray([float(item["illum_centroid_score"]) for item in history], dtype=np.float32)
    alignment = np.asarray([float(item["alignment_score"]) for item in history], dtype=np.float32)
    wiring = np.asarray([float(item["wiring_score"]) for item in history], dtype=np.float32)

    plt.figure(figsize=(11, 5.5))
    plt.plot(generations, uniformity, label="uniformity_score", linewidth=2.0)
    plt.plot(generations, illum_centroid, label="illum_centroid_score", linewidth=2.0)
    plt.plot(generations, alignment, label="alignment_score", linewidth=2.0)
    plt.plot(generations, wiring, label="wiring_score", linewidth=2.0)
    plt.xlabel("Generation")
    plt.ylabel("Best Individual Raw Score")
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
        return [tuple(map(int, coord)) for coord in coords]

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
        lamp_mask = np.zeros_like(self.env.original_placeable_mask, dtype=bool)
        for gene in genome:
            lamp_mask[self._id_to_point(gene)] = True
        return RoomState.from_channels(
            room_mask=self.env.original_room_mask,
            lamp_mask=lamp_mask,
            switch_mask=self.env.original_switch_mask,
            door_mask=self.env.original_door_mask,
        )
    ## 计算一个基因组的适应度，首先根据基因组构建房间状态，然后调用reward_calculator计算奖励分数，并返回总分和详细的奖励分解
    def evaluate_genome(self, genome: tuple[int, ...]) -> tuple[float, RewardBreakdown]:
        state = self._build_state(genome)
        breakdown = self.reward_calculator.calculate_step_reward(
            state,
            invalid_action=False,
            pair_cost_provider=self.env.pair_cost,
        )
        return float(breakdown.total), breakdown

    def _tournament_select(self, scored_population: list[tuple[tuple[int, ...], float, RewardBreakdown]]) -> tuple[int, ...]:
        competitors = random.sample(scored_population, k=min(self.ga_config.tournament_size, len(scored_population)))
        winner = max(competitors, key=lambda item: item[1])
        return winner[0]

    def _crossover(self, parent_a: tuple[int, ...], parent_b: tuple[int, ...]) -> tuple[int, ...]:
        shared = [gene for gene in parent_a if gene in parent_b]
        child = list(shared)

        pool = [gene for gene in parent_a + parent_b if gene not in child]
        random.shuffle(pool)
        for gene in pool:
            if gene not in child:
                child.append(gene)
            if len(child) >= self.target_lamp_count:
                break

        return self._repair_genome(child)

    def _mutate(self, genome: tuple[int, ...]) -> tuple[int, ...]:
        genes = list(genome)
        if random.random() >= self.ga_config.mutation_rate:
            return genome

        replace_count = min(self.ga_config.mutation_count, self.target_lamp_count)
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
            f"u={breakdown.diagnostics.get('uniformity_score', 0.0):.2f} "
            f"c={breakdown.diagnostics.get('illum_centroid_score', 0.0):.2f} "
            f"a={breakdown.diagnostics.get('alignment_score', 0.0):.2f} "
            f"w={breakdown.diagnostics.get('wiring_score', 0.0):.2f}"
        )
        return save_room_grid_image(encoded, output_path, cell_size=32, room_name=room_title)

    def run(self) -> dict[str, Any]:
        population = self._initialize_population()
        history: list[dict[str, Any]] = []
        best_genome: tuple[int, ...] | None = None
        best_breakdown: RewardBreakdown | None = None
        best_fitness = float("-inf")
        stagnant_generations = 0

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
                    "uniformity_score": float(generation_best_breakdown.diagnostics.get("uniformity_score", 0.0)),
                    "illum_centroid_score": float(
                        generation_best_breakdown.diagnostics.get("illum_centroid_score", 0.0)
                    ),
                    "alignment_score": float(generation_best_breakdown.diagnostics.get("alignment_score", 0.0)),
                    "wiring_score": float(generation_best_breakdown.diagnostics.get("wiring_score", 0.0)),
                    "mst_cost": float(generation_best_breakdown.diagnostics.get("mst_cost", 0.0)),
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
                    f"u={generation_best_breakdown.diagnostics.get('uniformity_score', 0.0):5.2f} "
                    f"c={generation_best_breakdown.diagnostics.get('illum_centroid_score', 0.0):5.2f} "
                    f"a={generation_best_breakdown.diagnostics.get('alignment_score', 0.0):5.2f} "
                    f"w={generation_best_breakdown.diagnostics.get('wiring_score', 0.0):5.2f}"
                )

            if stagnant_generations >= self.ga_config.patience:
                break

            elites = [item[0] for item in scored_population[: self.ga_config.elite_count]]
            next_population = list(elites)
            while len(next_population) < self.ga_config.population_size:
                parent_a = self._tournament_select(scored_population)
                parent_b = self._tournament_select(scored_population)
                child = self._crossover(parent_a, parent_b)
                child = self._mutate(child)
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
            "best_diagnostics": best_breakdown.diagnostics,
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

    output_dir = create_timestamped_output_dir(repo_root / "RL" / "output")
    shutil.copy2(config_path, output_dir / "config.yaml")
    set_seed(ga_config.seed)

    env = SingleRoomLightingEnv.from_json(
        repo_root / room_cfg.json_path,
        room_name=room_cfg.room_name,
        config=env_cfg,
    )
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
