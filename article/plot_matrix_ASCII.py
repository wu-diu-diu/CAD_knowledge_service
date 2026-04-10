"""
# 示例：agent 中的房间矩阵到 ASCII 棋盘转换
#
# 负责读取该视图的工具入口是：
#   LightingTools.tool_read_matrix_state(...)
#
# 实际执行矩阵 -> ASCII 棋盘转换的核心函数是：
#   RoomAgentState.to_ascii_board(...)
#
# 转换规则：
#   - 若某格在 placements["switches"] 中，则输出 "S"
#   - 否则若某格在 placements["lamps"] 中，则输出 "L"
#   - 否则若 matrix 值为 2，则输出 "D"
#   - 否则若 matrix 值为 1，则输出 "."
#   - 其余情况输出 "#"
#
# 示例输入：
#   matrix =
#   [
#     [0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 2, 1, 0],
#     [0, 1, 1, 1, 1, 0],
#     [0, 1, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0, 0],
#   ]
#
#   placements = {
#     "switches": [[1, 1]],
#     "lamps": [[2, 3], [3, 4]],
#   }
#
# 对应 ASCII 棋盘（compress=False）：
#   ######
#   #S.D.#
#   #..L.#
#   #...L#
#   ######
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.font_manager import FontProperties, fontManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.state import RoomAgentState

matplotlib.use("Agg")


FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    fontManager.addfont(FONT_PATH)
    CN_FONT = FontProperties(fname=FONT_PATH)
    plt.rcParams["font.family"] = CN_FONT.get_name()
except Exception:
    CN_FONT = None


COLOR_OBS = "#c73b2a"
COLOR_FREE = "#bfe4b8"
COLOR_DOOR = "#6ca8d7"
COLOR_SWITCH = "#f4cf49"
COLOR_LAMP = "#1d1d1d"
GRID_COLOR = "#d0d0d0"
ASCII_BORDER = "#666666"


def _example_state() -> tuple[np.ndarray, dict[str, list[list[int]]], str]:
    matrix = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 2, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    placements = {
        "switches": [[1, 1]],
        "lamps": [[2, 3], [3, 4]],
    }
    state = RoomAgentState(
        room_name="示例房间",
        area_m2=12.0,
        matrix=matrix,
        placements=placements,
    )
    ascii_board = state.to_ascii_board(
        max_rows=matrix.shape[0],
        max_cols=matrix.shape[1],
        compress=False,
    )
    return matrix, placements, ascii_board


def _draw_matrix(ax: plt.Axes, matrix: np.ndarray, placements: dict[str, list[list[int]]]) -> None:
    rows, cols = matrix.shape
    switch_cells = {tuple(p) for p in placements.get("switches", [])}
    lamp_cells = {tuple(p) for p in placements.get("lamps", [])}

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    for r in range(rows):
        for c in range(cols):
            value = int(matrix[r, c])
            if value == 0:
                face = COLOR_OBS
            elif value == 1:
                face = COLOR_FREE
            elif value == 2:
                face = COLOR_DOOR
            else:
                face = COLOR_FREE

            rect = patches.Rectangle((c, r), 1, 1, facecolor=face, edgecolor=GRID_COLOR, linewidth=1.0)
            ax.add_patch(rect)

            if (r, c) in switch_cells:
                overlay = patches.Rectangle((c + 0.08, r + 0.08), 0.84, 0.84, facecolor=COLOR_SWITCH, edgecolor="none")
                ax.add_patch(overlay)
                ax.text(c + 0.5, r + 0.5, "S", ha="center", va="center", fontsize=13, fontweight="bold", color="black")
            elif (r, c) in lamp_cells:
                overlay = patches.Rectangle((c + 0.08, r + 0.08), 0.84, 0.84, facecolor=COLOR_LAMP, edgecolor="none")
                ax.add_patch(overlay)
                ax.text(c + 0.5, r + 0.5, "L", ha="center", va="center", fontsize=13, fontweight="bold", color="white")
            else:
                ax.text(c + 0.5, r + 0.5, str(value), ha="center", va="center", fontsize=11, color="black")

    ax.set_title("Matrix + Placements", fontsize=15, fontweight="bold", pad=10)


def _char_style(ch: str) -> tuple[str, str]:
    if ch == "#":
        return COLOR_OBS, "white"
    if ch == ".":
        return COLOR_FREE, "black"
    if ch == "D":
        return COLOR_DOOR, "black"
    if ch == "S":
        return COLOR_SWITCH, "black"
    if ch == "L":
        return COLOR_LAMP, "white"
    return "white", "black"


def _draw_ascii(ax: plt.Axes, ascii_board: str) -> None:
    lines = ascii_board.splitlines()
    rows = len(lines)
    cols = len(lines[0]) if lines else 0

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            face, text_color = _char_style(ch)
            rect = patches.Rectangle((c, r), 1, 1, facecolor=face, edgecolor=ASCII_BORDER, linewidth=1.0)
            ax.add_patch(rect)
            ax.text(
                c + 0.5,
                r + 0.5,
                ch,
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=text_color,
                family="monospace",
            )

    ax.set_title("ASCII Board", fontsize=15, fontweight="bold", pad=10)


def _draw_arrow(fig: plt.Figure, ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    box_l = ax_left.get_position()
    box_r = ax_right.get_position()
    x0 = box_l.x1 + 0.015
    x1 = box_r.x0 - 0.015
    y = (box_l.y0 + box_l.y1) / 2
    arrow = patches.FancyArrowPatch(
        (x0, y),
        (x1, y),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.0,
        color="#2c6a73",
    )
    fig.add_artist(arrow)
    fig.text((x0 + x1) / 2, y + 0.03, "to_ascii_board()", ha="center", va="center", fontsize=12, fontweight="bold")


def main() -> None:
    matrix, placements, ascii_board = _example_state()

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    _draw_matrix(axes[0], matrix, placements)
    _draw_ascii(axes[1], ascii_board)
    _draw_arrow(fig, axes[0], axes[1])

    fig.suptitle("Matrix-to-ASCII Conversion Example", fontsize=17, fontweight="bold", y=0.97)
    plt.subplots_adjust(left=0.05, right=0.97, top=0.86, bottom=0.08, wspace=0.34)

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    png_path = results_dir / "matrix_ascii.png"
    pdf_path = results_dir / "matrix_ascii.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


if __name__ == "__main__":
    main()
