"""
绘制 LightingActorCritic 模型架构图（UNet 论文风格）。

用法:
    python RL/visualize_model.py --output RL/model_arch.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


# ── 颜色 ──────────────────────────────────────────────────────────────────────
C_ENCODER    = "#5B9BD5"   # 蓝：编码器
C_BOTTLENECK = "#ED7D31"   # 橙：瓶颈
C_DECODER    = "#A9D18E"   # 绿：解码器
C_HEAD_POL   = "#FF6B6B"   # 红：policy head
C_HEAD_VAL   = "#4ECDC4"   # 青：value head
C_SKIP       = "#7F7F7F"   # 灰：skip connection 箭头
C_DOWN       = "#5B9BD5"   # 下采样箭头
C_UP         = "#A9D18E"   # 上采样箭头
C_BG         = "#FFFFFF"
EDGE_COLOR   = "#FFFFFF"


def draw_block(ax, cx, cy, width, height, color, label, sublabel="",
               fontsize=7.5, text_color="white", alpha=1.0):
    """在 (cx, cy) 为中心绘制一个竖向矩形方块。"""
    rect = plt.Rectangle(
        (cx - width / 2, cy - height / 2), width, height,
        facecolor=color, edgecolor=EDGE_COLOR,
        linewidth=1.5, alpha=alpha, zorder=3,
    )
    ax.add_patch(rect)
    # 主标签（竖排文字）
    ax.text(cx, cy + (height * 0.12 if sublabel else 0),
            label, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold", zorder=4,
            rotation=90 if height > 0.6 else 0)
    if sublabel:
        ax.text(cx, cy - height * 0.22, sublabel,
                ha="center", va="center",
                fontsize=fontsize - 1, color=text_color,
                alpha=0.9, zorder=4,
                rotation=90 if height > 0.6 else 0)


def draw_arrow(ax, x0, y0, x1, y1, color="#555", lw=1.5,
               label="", label_side="right"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=10),
        zorder=5,
    )
    if label:
        mx = (x0 + x1) / 2 + (0.06 if label_side == "right" else -0.06)
        my = (y0 + y1) / 2
        ax.text(mx, my, label, fontsize=6.5, color=color,
                va="center", ha="left" if label_side == "right" else "right",
                zorder=6)


def draw_skip(ax, x0, y, x1, color=C_SKIP, lw=1.3, label=""):
    """水平 skip connection 箭头（带弧度）。"""
    ax.annotate(
        "", xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            mutation_scale=9,
            connectionstyle="arc3,rad=-0.25",
        ),
        zorder=5,
    )
    if label:
        ax.text((x0 + x1) / 2, y + 0.12, label,
                ha="center", fontsize=6.5, color=color, zorder=6)


def draw_legend(ax, items, x, y, spacing=0.32):
    for i, (color, label) in enumerate(items):
        iy = y - i * spacing
        rect = plt.Rectangle((x, iy - 0.09), 0.22, 0.18,
                              facecolor=color, edgecolor="#aaa",
                              linewidth=0.8, zorder=6)
        ax.add_patch(rect)
        ax.text(x + 0.28, iy, label, fontsize=7.5, va="center",
                color="#333", zorder=7)


def draw_architecture(output_path: str | Path) -> None:
    # ── 画布 ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # ── 标题 ──────────────────────────────────────────────────────────────────
    ax.text(9, 8.65, "LightingActorCritic  —  PPO Actor-Critic (UNet Backbone)",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#222")
    ax.text(9, 8.3, "Input: [B, 6, H, W]   H=W=padded_size (e.g. 48)",
            ha="center", va="center", fontsize=9, color="#666")

    # ── 布局参数 ──────────────────────────────────────────────────────────────
    # 每个 stage 的方块：(cx, cy, width, height, color, label, sublabel)
    # 高度正比于 feature map 空间分辨率（H），宽度正比于通道数（对数）
    # 编码器从左到右，解码器从中间向右展开

    BW = 0.30   # 基础宽度单位
    BH = 0.10   # 基础高度单位（每个空间单元）

    # 空间分辨率：H, H/2, H/4, H/8
    # 通道数：    6, 32,  64,  128, 256
    # 高度用 H 的倍数表示（H=8 单元）
    H_UNITS = 8.0

    def ch_width(ch):
        """通道数 → 方块宽度（对数缩放）。"""
        return BW * (1 + np.log2(ch / 6) * 0.55)

    def res_height(divisor):
        """分辨率 → 方块高度。"""
        return BH * H_UNITS / divisor

    # ── 输入 ──────────────────────────────────────────────────────────────────
    inp_cx, inp_cy = 1.2, 4.5
    inp_w, inp_h = ch_width(6), res_height(1)
    draw_block(ax, inp_cx, inp_cy, inp_w, inp_h,
               "#888888", "Input", "[B,6,H,W]", fontsize=7)

    # ── SharedEncoder ─────────────────────────────────────────────────────────
    # stage1: [B,32,H,W]   stage2: [B,64,H/2,W/2]
    # stage3: [B,128,H/4,W/4]  stage4(bottleneck): [B,256,H/8,W/8]
    enc_stages = [
        # (cx,   cy,   ch,  div, color,        label,    sublabel)
        (2.4,  4.5,  32,  1,   C_ENCODER,    "Stage1",  "32×H×W"),
        (3.6,  3.75, 64,  2,   C_ENCODER,    "Stage2",  "64×H/2"),
        (4.8,  3.0,  128, 4,   C_ENCODER,    "Stage3",  "128×H/4"),
        (6.0,  2.25, 256, 8,   C_BOTTLENECK, "Stage4",  "256×H/8"),
    ]

    enc_blocks = []
    for cx, cy, ch, div, color, label, sublabel in enc_stages:
        w = ch_width(ch)
        h = res_height(div)
        draw_block(ax, cx, cy, w, h, color, label, sublabel, fontsize=7.5)
        enc_blocks.append((cx, cy, w, h))

    # encoder 标签
    ax.text(4.2, 7.9, "SharedEncoder", ha="center", fontsize=10,
            fontweight="bold", color=C_ENCODER)
    ax.add_patch(FancyBboxPatch((1.6, 1.2), 5.2, 6.4,
                                boxstyle="round,pad=0.1",
                                facecolor="#EBF3FB", edgecolor=C_ENCODER,
                                linewidth=1.5, alpha=0.3, zorder=1))

    # input → stage1
    draw_arrow(ax, inp_cx + inp_w/2, inp_cy,
               enc_blocks[0][0] - enc_blocks[0][2]/2, enc_blocks[0][1],
               color="#888", lw=1.5)

    # stage1→2, 2→3, 3→4（下采样箭头，斜向右下）
    for i in range(len(enc_blocks) - 1):
        cx0, cy0, w0, h0 = enc_blocks[i]
        cx1, cy1, w1, h1 = enc_blocks[i + 1]
        draw_arrow(ax, cx0 + w0/2, cy0 - h0/2,
                   cx1 - w1/2, cy1 + h1/2,
                   color=C_DOWN, lw=1.6, label="MaxPool2d")

    # ── PolicyDecoder (Actor) ─────────────────────────────────────────────────
    pol_stages = [
        # (cx,   cy,   ch,  div, label,    sublabel)
        (7.8,  3.0,  128, 4,   "UpBlock3", "128×H/4"),
        (9.2,  3.75, 64,  2,   "UpBlock2", "64×H/2"),
        (10.6, 4.5,  32,  1,   "UpBlock1", "32×H×W"),
    ]

    pol_blocks = []
    for cx, cy, ch, div, label, sublabel in pol_stages:
        w = ch_width(ch)
        h = res_height(div)
        draw_block(ax, cx, cy, w, h, C_DECODER, label, sublabel, fontsize=7.5)
        pol_blocks.append((cx, cy, w, h))

    ax.text(9.2, 7.9, "PolicyDecoder  (Actor)", ha="center", fontsize=10,
            fontweight="bold", color="#4A7C3F")
    ax.add_patch(FancyBboxPatch((7.0, 1.2), 4.4, 6.4,
                                boxstyle="round,pad=0.1",
                                facecolor="#EDF7E6", edgecolor=C_DECODER,
                                linewidth=1.5, alpha=0.3, zorder=1))

    # bottleneck → UpBlock3
    cx0, cy0, w0, h0 = enc_blocks[3]
    cx1, cy1, w1, h1 = pol_blocks[0]
    draw_arrow(ax, cx0 + w0/2, cy0 + h0/2,
               cx1 - w1/2, cy1 - h1/2,
               color=C_BOTTLENECK, lw=2.0, label="ConvTranspose2d")

    # UpBlock3→2→1
    for i in range(len(pol_blocks) - 1):
        cx0, cy0, w0, h0 = pol_blocks[i]
        cx1, cy1, w1, h1 = pol_blocks[i + 1]
        draw_arrow(ax, cx0 + w0/2, cy0 + h0/2,
                   cx1 - w1/2, cy1 - h1/2,
                   color=C_UP, lw=1.6, label="ConvTranspose2d")

    # skip connections: stage3→UpBlock3, stage2→UpBlock2, stage1→UpBlock1
    skip_pairs = [
        (enc_blocks[2], pol_blocks[0], "skip s3"),
        (enc_blocks[1], pol_blocks[1], "skip s2"),
        (enc_blocks[0], pol_blocks[2], "skip s1"),
    ]
    for (ecx, ecy, ew, eh), (pcx, pcy, pw, ph), lbl in skip_pairs:
        # 从 encoder 方块右侧顶部 → decoder 方块左侧顶部（同高度）
        y_skip = max(ecy + eh/2 - 0.1, pcy + ph/2 - 0.1)
        draw_skip(ax, ecx + ew/2, y_skip, pcx - pw/2, label=lbl)

    # ── Policy Heads ──────────────────────────────────────────────────────────
    # spatial head
    sp_cx, sp_cy = 12.0, 5.5
    sp_w, sp_h = 0.55, res_height(1) * 0.5
    draw_block(ax, sp_cx, sp_cy, sp_w, sp_h, C_HEAD_POL,
               "spatial\nhead", "Conv1×1\n→[B,H*W]", fontsize=6.5)

    # stop head
    st_cx, st_cy = 12.0, 3.5
    st_w, st_h = 0.55, 0.55
    draw_block(ax, st_cx, st_cy, st_w, st_h, C_HEAD_POL,
               "stop\nhead", "AvgPool\n→Linear\n→[B,1]", fontsize=6.5)

    # UpBlock1 → spatial head
    cx1, cy1, w1, h1 = pol_blocks[2]
    draw_arrow(ax, cx1 + w1/2, cy1 + h1*0.2,
               sp_cx - sp_w/2, sp_cy,
               color=C_HEAD_POL, lw=1.4)

    # bottleneck → stop head（直接从 stage4）
    cx0, cy0, w0, h0 = enc_blocks[3]
    draw_arrow(ax, cx0 + w0/2, cy0,
               st_cx - st_w/2, st_cy,
               color=C_HEAD_POL, lw=1.4)

    # concat + mask
    cat_cx, cat_cy = 13.3, 4.5
    cat_w, cat_h = 0.7, 0.55
    draw_block(ax, cat_cx, cat_cy, cat_w, cat_h, "#C0392B",
               "cat+mask", "[B,H*W+1]", fontsize=7)
    draw_arrow(ax, sp_cx + sp_w/2, sp_cy, cat_cx - cat_w/2, cat_cy + 0.1,
               color=C_HEAD_POL, lw=1.3)
    draw_arrow(ax, st_cx + st_w/2, st_cy, cat_cx - cat_w/2, cat_cy - 0.1,
               color=C_HEAD_POL, lw=1.3)

    # policy output
    out_pol_cx, out_pol_cy = 14.6, 4.5
    out_pol_w, out_pol_h = 0.8, 0.55
    draw_block(ax, out_pol_cx, out_pol_cy, out_pol_w, out_pol_h, C_HEAD_POL,
               "Policy", "Categorical\naction/log_p", fontsize=7)
    draw_arrow(ax, cat_cx + cat_w/2, cat_cy,
               out_pol_cx - out_pol_w/2, out_pol_cy,
               color=C_HEAD_POL, lw=1.5)

    # ── ValueDecoder (Critic) ─────────────────────────────────────────────────
    ax.text(16.2, 7.9, "ValueDecoder  (Critic)", ha="center", fontsize=10,
            fontweight="bold", color="#0E6655")
    ax.add_patch(FancyBboxPatch((15.0, 1.2), 2.5, 6.4,
                                boxstyle="round,pad=0.1",
                                facecolor="#D1F2EB", edgecolor=C_HEAD_VAL,
                                linewidth=1.5, alpha=0.3, zorder=1))

    val_pools = [
        (16.2, 5.8, "AvgPool\nstage2", "[B,64]"),
        (16.2, 4.5, "AvgPool\nstage3", "[B,128]"),
        (16.2, 3.2, "AvgPool\nstage4", "[B,256]"),
    ]
    pool_blocks = []
    for cx, cy, label, sublabel in val_pools:
        w, h = 0.65, 0.50
        draw_block(ax, cx, cy, w, h, C_HEAD_VAL, label, sublabel, fontsize=6.5)
        pool_blocks.append((cx, cy, w, h))

    # encoder stage2/3/4 → value pools（长箭头跨越 decoder 区域）
    for enc_idx, (pcx, pcy, pw, ph) in zip([1, 2, 3], pool_blocks):
        ecx, ecy, ew, eh = enc_blocks[enc_idx]
        ax.annotate(
            "", xy=(pcx - pw/2, pcy), xytext=(ecx + ew/2, ecy),
            arrowprops=dict(
                arrowstyle="-|>", color=C_HEAD_VAL, lw=1.2,
                mutation_scale=8,
                connectionstyle="arc3,rad=0.35",
            ),
            zorder=5,
        )

    # concat
    vcat_cx, vcat_cy = 16.2, 2.1
    vcat_w, vcat_h = 0.65, 0.45
    draw_block(ax, vcat_cx, vcat_cy, vcat_w, vcat_h, C_HEAD_VAL,
               "cat", "[B,448]", fontsize=7)
    for pcx, pcy, pw, ph in pool_blocks:
        draw_arrow(ax, pcx, pcy - ph/2, vcat_cx, vcat_cy + vcat_h/2,
                   color=C_HEAD_VAL, lw=1.1)

    # MLP
    mlp_items = [
        (16.2, 1.45, "Linear 448→256", "ReLU"),
        (16.2, 0.85, "Linear 256→128", "ReLU"),
    ]
    prev_cy = vcat_cy - vcat_h/2
    for cx, cy, label, act in mlp_items:
        w, h = 0.65, 0.38
        draw_block(ax, cx, cy, w, h, C_HEAD_VAL, label, act, fontsize=6)
        draw_arrow(ax, cx, prev_cy, cx, cy + h/2, color=C_HEAD_VAL, lw=1.1)
        prev_cy = cy - h/2

    # value output
    ax.text(16.2, 0.45, "Value  [B,1]",
            ha="center", fontsize=9, color=C_HEAD_VAL,
            fontweight="bold")
    draw_arrow(ax, 16.2, prev_cy, 16.2, 0.58, color=C_HEAD_VAL, lw=1.3)

    # ── 图例 ─────────────────────────────────────────────────────────────────
    legend_items = [
        (C_ENCODER,    "Encoder ConvBlock"),
        (C_BOTTLENECK, "Bottleneck (stage4)"),
        (C_DECODER,    "Actor UpBlock"),
        (C_HEAD_POL,   "Policy head / output"),
        (C_HEAD_VAL,   "Value decoder"),
        (C_SKIP,       "Skip connection"),
    ]
    draw_legend(ax, legend_items, x=0.15, y=7.5)

    # ── 输入通道注释 ──────────────────────────────────────────────────────────
    ch_info = [
        "ch0: placeable_mask",
        "ch1: placed_lamps",
        "ch2: switch_mask",
        "ch3: lamp_progress",
        "ch4: room_mask",
        "ch5: target_lamp_count",
    ]
    ax.text(0.15, 6.2, "Input channels:", fontsize=7.5,
            color="#444", fontweight="bold")
    for i, txt in enumerate(ch_info):
        ax.text(0.15, 5.9 - i * 0.27, txt, fontsize=7,
                color="#555", fontfamily="monospace")

    # ── 保存 ─────────────────────────────────────────────────────────────────
    plt.tight_layout(pad=0.3)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualize_model] saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "results" / "rl_model_arch.png"),
    )
    args = parser.parse_args()
    draw_architecture(args.output)
