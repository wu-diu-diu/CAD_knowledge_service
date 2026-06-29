"""
绘制 5x5 简化通道示意图（中间几个红色网格，其余绿色，无数字）。
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties


ARTICLE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ARTICLE_DIR / "results"
OUTPUT_PATH = RESULTS_DIR / "channel_simple.png"

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CN_FONT: FontProperties | None = None
if os.path.exists(FONT_PATH):
    CN_FONT = FontProperties(fname=FONT_PATH)
rcParams["axes.unicode_minus"] = False

CHANNEL_ZERO = "#d62728"   # 红色
CHANNEL_ONE = "#2ca02c"    # 绿色
GRID_LINE = (1, 1, 1, 0.35)

# 5x5 网格，0=红色，1=绿色（中间几个红色）
GRID = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
    ],
    dtype=np.int32,
)


def draw_channel(ax: plt.Axes, channel: np.ndarray) -> None:
    rows, cols = channel.shape

    for r in range(rows):
        for c in range(cols):
            y = rows - r - 1
            val = int(channel[r, c])
            color = CHANNEL_ONE if val > 0 else CHANNEL_ZERO
            rect = mpatches.Rectangle(
                (c, y), 1, 1,
                facecolor=color,
                edgecolor=GRID_LINE,
                linewidth=1.2,
            )
            ax.add_patch(rect)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    draw_channel(ax, GRID)

    plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    print(f"saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
