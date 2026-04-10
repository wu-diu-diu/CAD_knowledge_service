"""
绘制 RL 动作流程图：
  房间输入 → 神经网络 → 空间动作 H×W + Stop动作 → 掩码 → 采样
使用 PlotNeuralNet + 原生 TikZ 节点混合绘制。
输出: article/results/action_flow.png
"""
import sys
import os

sys.path.append('/home/chen/punchy/PlotNeuralNet')
from pycore.tikzeng import to_head, to_cor, to_begin, to_end, to_generate


ARTICLE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ARTICLE_DIR, "results")


def box(name, offset, to, w, h, d, fill=r'\ConvColor'):
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r"""
    {Box={
        name=""" + name + r""",
        caption= ,
        xlabel={{ , }},
        zlabel= ,
        fill=""" + fill + r""",
        height=""" + str(h) + r""",
        width=""" + str(w) + r""",
        depth=""" + str(d) + r"""
        }
    };
"""


def label(node, text, anchor="south", yshift="-0.5cm", xshift="0cm"):
    return (
        r"\node[below=" + yshift + r" of " + node + "-" + anchor
        + r", font=\bfseries\normalsize, align=center, xshift=" + xshift + r"] {"
        + text + r"};" + "\n"
    )


def arrow(a, b):
    return (
        r"\draw[-stealth, line width=1.5pt, draw=\edgecolor] ("
        + a + r") -- (" + b + r");" + "\n"
    )


def arrow_label(a, b, txt, pos="above"):
    return (
        r"\draw[-stealth, line width=1.5pt, draw=\edgecolor] ("
        + a + r") -- node[" + pos + r", font=\bfseries\small, text=black] {"
        + txt + r"} (" + b + r");" + "\n"
    )


# ── 颜色 ──────────────────────────────────────────────────────────────────────
C_INPUT   = r'\ConvColor'           # 黄色系 – 输入
C_NET     = r'\ConvReluColor'       # 橙色系 – 网络层
C_SPATIAL = r'rgb:green,4;white,3'  # 绿色   – 空间动作图
C_STOP    = r'rgb:blue,4;white,3'   # 蓝色   – stop 动作
C_MASK    = r'rgb:red,3;white,4'    # 红色   – 掩码
C_SAMPLE  = r'rgb:magenta,4;white,3'# 紫色   – 采样结果

arch = [
    to_head('/home/chen/punchy/PlotNeuralNet'),
    to_cor(),
    r"\usetikzlibrary{calc,shapes,fit,backgrounds}",
    to_begin(),

    # ── 自定义颜色 ────────────────────────────────────────────────────────────
    r"\def\InputColor{" + C_INPUT   + r"}",
    r"\def\NetColor{"   + C_NET     + r"}",
    r"\def\SpatColor{"  + C_SPATIAL + r"}",
    r"\def\StopColor{"  + C_STOP    + r"}",
    r"\def\MaskColor{"  + C_MASK    + r"}",
    r"\def\SampColor{"  + C_SAMPLE  + r"}",

    # ════════════════════════════════════════════════════════════════════════
    # 1. 输入：房间状态 H×W×C
    # ════════════════════════════════════════════════════════════════════════
    box('inp', "(0,0,0)", "(0,0,0)", 2, 28, 28, r'\InputColor'),
    r"\node[below=1.5cm of inp-south, font=\bfseries\normalsize, align=center]"
    r" {Room State\\$H \times W \times C$};" + "\n",

    # ════════════════════════════════════════════════════════════════════════
    # 2. 神经网络（三层示意）
    # ════════════════════════════════════════════════════════════════════════
    box('fc1', "(3,0,0)", "(inp-east)",  3, 22, 6, r'\NetColor'),
    box('fc2', "(2,0,0)", "(fc1-east)",  3, 16, 6, r'\NetColor'),
    box('fc3', "(2,0,0)", "(fc2-east)",  3, 12, 6, r'\NetColor'),

    arrow('inp-east',  'fc1-west'),
    arrow('fc1-east',  'fc2-west'),
    arrow('fc2-east',  'fc3-west'),

    r"\node[below=1.15cm of fc2-south, font=\bfseries\normalsize, align=center]"
    r" {Neural Network};" + "\n",

    # ════════════════════════════════════════════════════════════════════════
    # 3. 输出头：空间动作图 + Stop
    # ════════════════════════════════════════════════════════════════════════

    # 空间动作图 H×W（上方）
    box('spat', "(5, 4,0)", "(fc3-east)", 2, 28, 28, r'\SpatColor'),
    r"\node[below=1.5cm of spat-south, font=\bfseries\normalsize, align=center]"
    r" {Spatial Logits\\$H \times W$};" + "\n",

    # Stop 动作（下方）
    box('stop', "(5,-4,0)", "(fc3-east)", 2,  6,  6, r'\StopColor'),
    r"\node[below=0.75cm of stop-south, font=\bfseries\normalsize, align=center]"
    r" {Stop Logit\\$1$};" + "\n",

    # fc3 → spat / stop（折线箭头）
    r"""
\draw[-stealth, line width=1.5pt, draw=\edgecolor]
    (fc3-north) -- (fc3-north |- spat-west) -- (spat-west);
\draw[-stealth, line width=1.5pt, draw=\edgecolor]
    (fc3-south) -- (fc3-south |- stop-west) -- (stop-west);
""",

    # ════════════════════════════════════════════════════════════════════════
    # 4. 掩码：与 spat 同一水平线，放在 spat 上方（y 偏移对齐）
    # ════════════════════════════════════════════════════════════════════════
    box('mask', "(5,0,0)", "(spat-west)", 2, 28, 28, r'\MaskColor'),
    r"\node[below=1.5cm of mask-south, font=\bfseries\normalsize, align=center]"
    r" {Placeable Mask\\$H \times W$};" + "\n",

    # spat → mask（水平箭头）
    r"""
\draw[-stealth, line width=1.5pt, draw=\edgecolor]
    (spat-east) -- node[above, font=\bfseries\small, text=black] {Mask} (mask-west);
""",

    # ════════════════════════════════════════════════════════════════════════
    # 5. Masked Logits → Softmax → Sample
    # ════════════════════════════════════════════════════════════════════════

    box('soft', "(10,0,0)", "(spat-east)", 2, 28, 28, r'\SampColor'),
    r"\node[below=1.5cm of soft-south, font=\bfseries\normalsize, align=center]"
    r" {Softmax\\Prob. Map};" + "\n",

    r"""
\draw[-stealth, line width=1.5pt, draw=\edgecolor]
    (mask-east) -- node[above, font=\bfseries\small, text=black] {Softmax} (soft-west);
""",

    # Stop sigmoid
    box('sigs', "(5,0,0)", "(stop-east)", 2,  6,  6, r'\SampColor'),
    r"\node[below=0.75cm of sigs-south, font=\bfseries\normalsize, align=center]"
    r" {Sigmoid\\Stop Prob.};" + "\n",
    r"""
\draw[-stealth, line width=1.5pt, draw=\edgecolor]
    (stop-east) -- node[above, font=\bfseries\small, text=black] {Sigmoid} (sigs-west);
""",

    # ════════════════════════════════════════════════════════════════════════
    # 6. 采样结果
    # ════════════════════════════════════════════════════════════════════════
    r"""
\node[draw, fill=\SampColor, rounded corners=4pt,
      minimum width=2.2cm, minimum height=1.0cm,
      font=\bfseries\normalsize, align=center,
      right=3.5cm of soft-east] (act_pos) {Position\\$(r, c)$};

\node[draw, fill=\SampColor, rounded corners=4pt,
      minimum width=2.2cm, minimum height=1.0cm,
      font=\bfseries\normalsize, align=center,
      right=3.5cm of sigs-east] (act_stop) {Stop\\$\{0,1\}$};

\draw[-stealth, line width=1.5pt, draw=\edgecolor]
    (soft-east) -- node[above, font=\bfseries\small, text=black] {Sample} (act_pos.west);
\draw[-stealth, line width=1.5pt, draw=\edgecolor]
    (sigs-east) -- node[above, font=\bfseries\small, text=black] {Sample} (act_stop.west);
""",

    to_end(),
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_tex = os.path.join(RESULTS_DIR, 'action_flow.tex')
    to_generate(arch, out_tex)
    print(f"[plot_action_flow] generated → {out_tex}")


if __name__ == '__main__':
    main()
