"""
绘制 A* + Prim 最小生成树过程示意图，横向4个子图。

子图顺序：
  1. 将已布置元件作为终端点，构建完全图
  2. 使用A*算法计算完全图中每条边的路径成本
  3. 使用Prim算法找到总成本最低、无环、连通的最小树
  4. 此时最小树的总成本即视为线路成本
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties


ARTICLE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ARTICLE_DIR / "results"
OUTPUT_PATH = RESULTS_DIR / "astar_prim_mst.png"

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CN_FONT: FontProperties | None = None
if os.path.exists(FONT_PATH):
    CN_FONT = FontProperties(fname=FONT_PATH)
rcParams["axes.unicode_minus"] = False

# 4 个终端点（正方形布局）
POINTS = np.array([
    [0.5, 0.5],   # 左下
    [3.5, 0.5],   # 右下
    [0.5, 3.5],   # 左上
    [3.5, 3.5],   # 右上
])

N = len(POINTS)
ALL_EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]

# 各边 A* 路径成本：四条边为3.0，两条对角线为4.2
EDGE_COSTS = {
    (0, 1): 3.0,
    (0, 2): 3.0,
    (0, 3): 4.2,
    (1, 2): 4.2,
    (1, 3): 3.0,
    (2, 3): 3.0,
}

# Prim 选出的 MST（3 条边，舍去对角线和一条边）
MST_EDGES = {(0, 1), (0, 2), (1, 3)}

MST_TOTAL_COST = sum(EDGE_COSTS[e] for e in MST_EDGES)

COMPLETE_COLOR = "#aaaaaa"
COMPLETE_LW = 1.4
MST_COLOR = "#111111"
MST_LW = 3.0
POINT_COLOR = "#111111"
POINT_SIZE = 140
COST_FONTSIZE = 16
TITLE_SIZE = 17


def _fp(size: int) -> FontProperties | None:
    if CN_FONT is None:
        return None
    fp = CN_FONT.copy()
    fp.set_size(size)
    return fp


def _draw_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(-0.6, 4.6)


def _draw_points(ax: plt.Axes) -> None:
    ax.scatter(POINTS[:, 0], POINTS[:, 1], s=POINT_SIZE, c=POINT_COLOR, zorder=5)


def _draw_edge(ax: plt.Axes, i: int, j: int, color: str, lw: float, zorder: int = 1) -> None:
    ax.plot(
        [POINTS[i, 0], POINTS[j, 0]],
        [POINTS[i, 1], POINTS[j, 1]],
        color=color,
        linewidth=lw,
        zorder=zorder,
    )


def _draw_edge_label(ax: plt.Axes, i: int, j: int, text: str, fs: int = COST_FONTSIZE) -> None:
    mx = (POINTS[i, 0] + POINTS[j, 0]) / 2
    my = (POINTS[i, 1] + POINTS[j, 1]) / 2
    dx = POINTS[j, 0] - POINTS[i, 0]
    dy = POINTS[j, 1] - POINTS[i, 1]
    offset_x = -dy * 0.1
    offset_y = dx * 0.1
    ax.text(
        mx + offset_x, my + offset_y, text,
        ha="center", va="center",
        fontsize=fs,
        fontweight="bold",
        fontproperties=_fp(fs),
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75),
    )


def plot_complete_graph(ax: plt.Axes) -> None:
    for i, j in ALL_EDGES:
        _draw_edge(ax, i, j, COMPLETE_COLOR, COMPLETE_LW)
    _draw_points(ax)
    _draw_axes(ax)


def plot_astar_costs(ax: plt.Axes) -> None:
    for i, j in ALL_EDGES:
        _draw_edge(ax, i, j, COMPLETE_COLOR, COMPLETE_LW)
        _draw_edge_label(ax, i, j, f"{EDGE_COSTS[(i, j)]:.1f}")
    _draw_points(ax)
    _draw_axes(ax)


def plot_prim_mst(ax: plt.Axes) -> None:
    for i, j in ALL_EDGES:
        if (i, j) in MST_EDGES or (j, i) in MST_EDGES:
            _draw_edge(ax, i, j, MST_COLOR, MST_LW, zorder=3)
        else:
            _draw_edge(ax, i, j, COMPLETE_COLOR, COMPLETE_LW, zorder=1)
    _draw_points(ax)
    _draw_axes(ax)


def plot_final_cost(ax: plt.Axes) -> None:
    for i, j in ALL_EDGES:
        if (i, j) in MST_EDGES or (j, i) in MST_EDGES:
            _draw_edge(ax, i, j, MST_COLOR, MST_LW, zorder=3)
            _draw_edge_label(ax, i, j, f"{EDGE_COSTS[(i, j)]:.1f}", fs=COST_FONTSIZE)
    _draw_points(ax)
    _draw_axes(ax)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1, 4,
        figsize=(18, 5.5),
        facecolor="white",
    )

    plot_complete_graph(axes[0])
    plot_astar_costs(axes[1])
    plot_prim_mst(axes[2])
    plot_final_cost(axes[3])

    plt.subplots_adjust(wspace=0.08)
    plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    print(f"saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
