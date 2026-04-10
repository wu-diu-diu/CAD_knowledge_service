"""
Draw the mask-sampling pipeline with matplotlib.

Each 1-D vector is rendered as a vertical grid column with centered numbers.
Output:
  - article/results/mask_sampling_matplotlib.png
  - article/results/mask_sampling_matplotlib.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


ARTICLE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ARTICLE_DIR / "results"


def draw_vector_column(
    ax,
    *,
    x: float,
    y_top: float,
    values: list[str],
    facecolors: list[str],
    title: str,
    cell_w: float = 1.55,
    cell_h: float = 0.72,
    text_size: int = 12,
) -> tuple[float, float, float, float]:
    """Draw one vertical vector column and return center_y plus left/right edges."""
    n = len(values)
    total_h = n * cell_h
    y_bottom = y_top - total_h

    for i, (val, fc) in enumerate(zip(values, facecolors)):
        y = y_top - (i + 1) * cell_h
        rect = Rectangle(
            (x, y),
            cell_w,
            cell_h,
            facecolor=fc,
            edgecolor="black",
            linewidth=1.3,
        )
        ax.add_patch(rect)
        ax.text(
            x + cell_w / 2,
            y + cell_h / 2,
            val,
            ha="center",
            va="center",
            fontsize=text_size,
            color="black",
            fontweight="bold",
        )

    ax.text(
        x + cell_w / 2,
        y_bottom - 0.55,
        title,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    center_y = (y_top + y_bottom) / 2
    return x + cell_w / 2, center_y, x, x + cell_w


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float], label: str) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color="#1f5f74",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)
    ax.text(
        (start[0] + end[0]) / 2,
        start[1] + 0.35,
        label,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="black",
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logits = ["2.1", "0.4", "1.7", "-0.2", "3.0", "0.8"]
    mask = ["1", "0", "1", "0", "1", "1"]
    masked = ["2.1", "-inf", "1.7", "-inf", "3.0", "0.8"]
    probs = ["0.20", "0.00", "0.14", "0.00", "0.53", "0.13"]
    sample = ["0", "0", "0", "0", "1", "0"]

    c_logit = "#cfe1ff"
    c_mask1 = "#cfeecf"
    c_mask0 = "#f7c9c9"
    c_masked = "#dddddd"
    c_prob = "#ead7f7"
    c_sample1 = "#ffd089"
    c_sample0 = "#efefef"

    cols = [
        ("Logits", logits, [c_logit] * 6),
        ("Action Mask", mask, [c_mask1, c_mask0, c_mask1, c_mask0, c_mask1, c_mask1]),
        ("Masked Logits", masked, [c_logit, c_masked, c_logit, c_masked, c_logit, c_logit]),
        ("Softmax Probs", probs, [c_prob] * 6),
        ("Sampled Action", sample, [c_sample0, c_sample0, c_sample0, c_sample0, c_sample1, c_sample0]),
    ]

    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    y_top = 5.8
    x0 = 0.8
    gap = 1.25
    cols_geom: list[tuple[float, float, float, float]] = []

    for idx, (title, values, fills) in enumerate(cols):
        x = x0 + idx * (1.55 + gap)
        geom = draw_vector_column(ax, x=x, y_top=y_top, values=values, facecolors=fills, title=title)
        cols_geom.append(geom)

    for i, label in enumerate(["Mask", "Apply", "Softmax", "Sample"]):
        _, cy1, _, right1 = cols_geom[i]
        _, cy2, left2, _ = cols_geom[i + 1]
        draw_arrow(ax, (right1, cy1), (left2, cy2), label)

    ax.set_xlim(0, x0 + len(cols) * (1.55 + gap) + 0.5)
    ax.set_ylim(-0.3, 6.7)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    png_path = RESULTS_DIR / "mask_sampling_matplotlib.png"
    pdf_path = RESULTS_DIR / "mask_sampling_matplotlib.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_mask_sampling_matplotlib] saved -> {png_path}")
    print(f"[plot_mask_sampling_matplotlib] saved -> {pdf_path}")


if __name__ == "__main__":
    main()
