"""
绘制布线任务的 6 通道状态示意图。

图像布局：
- 左侧：当前房间状态
- 右侧：6 个 observation 通道

示例场景：
- 5x5 矩形房间
- 4 盏灯均匀分布
- 有门、有开关
- 开关到最近灯具已连通一条线

输出：
- article/results/wiring_channels_example.png
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
import numpy as np
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties


ARTICLE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ARTICLE_DIR / "results"
OUTPUT_PATH = RESULTS_DIR / "wiring_channels_example.png"

ROOM_TITLE_SIZE = 20
CHANNEL_TITLE_SIZE = 20
TITLE_FONT_WEIGHT = "bold"

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CN_FONT: FontProperties | None = None
if os.path.exists(FONT_PATH):
    CN_FONT = FontProperties(fname=FONT_PATH)
rcParams["axes.unicode_minus"] = False


def _title_font(size: int) -> FontProperties | None:
    """Create a Chinese title font with explicit size/weight so constants always take effect."""
    if CN_FONT is None:
        return None
    fp = CN_FONT.copy()
    fp.set_size(size)
    fp.set_weight(TITLE_FONT_WEIGHT)
    return fp


# ── 示例房间状态 ──────────────────────────────────────────────────────────────

ROOM_STATE = np.array(
    [
        [2, 1, 1, 1, 1],
        [3, 4, 1, 4, 1],
        [1, 1, 1, 1, 1],
        [1, 4, 1, 4, 1],
        [1, 1, 1, 1, 1],
    ],
    dtype=np.int32,
)

WIRE_PATH = [(1, 0), (1, 1)]

CHANNELS = [
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    ),
    np.full((5, 5), 0.25, dtype=np.float32),
]


# ── 颜色配置 ─────────────────────────────────────────────────────────────────

ROOM_COLORS = {
    0: "#d62728",   # 红色：房间外/不可用
    1: "#2ca02c",   # 绿色：普通房间区域
    2: "#4f83d1",   # 蓝色：门
    3: "#f0d43a",   # 黄色：开关
    4: "#111111",   # 黑色：灯具
}

CHANNEL_ZERO = "#d62728"
CHANNEL_ONE = "#2ca02c"
GRID_LINE = (1, 1, 1, 0.35)
WIRE_COLOR = "#d62728"


def _fmt_value(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def draw_room(ax: plt.Axes, matrix: np.ndarray, path: list[tuple[int, int]]) -> None:
    rows, cols = matrix.shape

    for r in range(rows):
        for c in range(cols):
            y = rows - r - 1
            val = int(matrix[r, c])
            rect = mpatches.Rectangle(
                (c, y), 1, 1,
                facecolor=ROOM_COLORS[val],
                edgecolor=GRID_LINE,
                linewidth=1.2,
            )
            ax.add_patch(rect)

            text_color = "white" if val in (2, 4) else "black"
            ax.text(
                c + 0.5,
                y + 0.5,
                str(val),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=text_color,
            )

    if len(path) >= 2:
        xs = [c + 0.5 for _, c in path]
        ys = [rows - r - 0.5 for r, _ in path]
        ax.plot(xs, ys, color=WIRE_COLOR, linewidth=6, solid_capstyle="round", zorder=5)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "房间状态",
        pad=18,
        fontproperties=_title_font(ROOM_TITLE_SIZE),
    )


def draw_channel(ax: plt.Axes, channel: np.ndarray, title: str) -> None:
    rows, cols = channel.shape

    for r in range(rows):
        for c in range(cols):
            y = rows - r - 1
            val = float(channel[r, c])
            color = CHANNEL_ONE if val > 0 else CHANNEL_ZERO
            rect = mpatches.Rectangle(
                (c, y), 1, 1,
                facecolor=color,
                edgecolor=GRID_LINE,
                linewidth=1.2,
            )
            ax.add_patch(rect)
            ax.text(
                c + 0.5,
                y + 0.5,
                _fmt_value(val),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="black",
            )

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        title,
        pad=14,
        fontproperties=_title_font(CHANNEL_TITLE_SIZE),
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    gs = gridspec.GridSpec(
        2, 4,
        figure=fig,
        width_ratios=[1.05, 1, 1, 1],
        height_ratios=[1, 1],
        wspace=0.18,
        hspace=0.18,
    )

    room_ax = fig.add_subplot(gs[:, 0])
    draw_room(room_ax, ROOM_STATE, WIRE_PATH)

    positions = [
        (0, 1), (0, 2), (0, 3),
        (1, 1), (1, 2), (1, 3),
    ]
    for idx, (gr, gc) in enumerate(positions):
        ax = fig.add_subplot(gs[gr, gc])
        draw_channel(ax, CHANNELS[idx], f"通道{idx}")

    plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    print(f"saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
