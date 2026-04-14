"""
使用 PlotNeuralNet 绘制 SharedEncoder + ValueDecoder 架构图。
- Encoder: 6×48×48 → 256×6×6（与 PolicyDecoder 共享）
- ValueDecoder:
    Stage2(64×24×24) → AvgPool → 1×64  ┐
    Stage3(128×12×12) → AvgPool → 1×128 ├→ Concat → 1×448 → FC → 256 → FC → 128 → FC → 1
    Stage4(256×6×6)  → AvgPool → 1×256 ┘
  三个 pooling 输出在 x 方向对齐，y 方向均匀分布

用法:
    cd /home/chen/punchy/CAD_knowledge_service/article
    bash compile_value.sh
"""
import sys
import os

sys.path.append('/home/chen/punchy/PlotNeuralNet')
from pycore.tikzeng import to_head, to_cor, to_begin, to_end, to_generate


ARTICLE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ARTICLE_DIR, "results")


def to_tensor_block(name, offset, to, width, height, depth, fill=r'\ConvColor'):
    """单层方块，表示一个数据张量。"""
    return r"""
\pic[shift={""" + offset + r"""}] at """ + to + r"""
    {Box={
        name=""" + name + r""",
        caption= ,
        xlabel={{ , }},
        zlabel= ,
        fill=""" + fill + r""",
        height=""" + str(height) + r""",
        width=""" + str(width) + r""",
        depth=""" + str(depth) + r"""
        }
    };
"""


def to_label_below(node_name, text, xshift="-0.5cm", yshift="1.5cm"):
    """在方块正下方标注张量维度。"""
    return r"""
\node[below=""" + yshift + r""" of """ + node_name + r"""-south, font=\DimFont, align=center, xshift=""" + xshift + r"""]
    {""" + text + r"""};
"""


def to_arrow_with_label(from_east, to_west, label):
    """带操作名标注的箭头。"""
    return r"""
\draw [-stealth, line width=1.5pt, draw=\edgecolor] (""" + from_east + r"""-east) -- node[above, font=\LabelFont, text=black] {""" + label + r"""} (""" + to_west + r"""-west);
"""


def to_arrow(from_east, to_west):
    """仅绘制连接箭头。"""
    return r"""
\draw [-stealth, line width=1.5pt, draw=\edgecolor] (""" + from_east + r"""-east) -- (""" + to_west + r"""-west);
"""


def to_legend_entry(name, at_expr, fill, text, swatch_w="1.1cm", swatch_h="0.7cm"):
    """绘制图例色块及其文字说明。"""
    return r"""
\node[draw, rounded corners=2pt, fill=""" + fill + r""", minimum width=""" + swatch_w + r""", minimum height=""" + swatch_h + r"""] (""" + name + r""") at """ + at_expr + r""" {};
\node[anchor=west, font=\LabelFont] at ($(""" + name + r""".east)+(0.35cm,0)$) {""" + text + r"""};
"""


# ── 三个 pooling 向量的 y 方向间距和 x 位置 ──
# p2(1×64) 在上，p3(1×128) 在中，p4(1×256) 在下
# 用 \coordinate 定位，然后用 \pic 画方块

arch = [
    to_head('/home/chen/punchy/PlotNeuralNet'),
    to_cor(),
    r"\usepackage{fontspec}",
    r"\usetikzlibrary{calc}",
    to_begin(),
    # 自定义颜色
    r"\def\EncoderColor{\ConvColor}",
    r"\def\BottleneckColor{\ConvReluColor}",
    r"\def\ValueColor{rgb:blue,3;green,1;white,4}",
    r"\newfontfamily\SongtiFont{Noto Serif CJK SC}",
    r"\newcommand{\LabelFont}{\SongtiFont\fontsize{22pt}{24pt}\selectfont}",
    r"\newcommand{\DimFont}{\SongtiFont\fontsize{22pt}{24pt}\selectfont\boldmath}",

    # ════════════════════════════════════════════════════════════════════════
    # ENCODER
    # ════════════════════════════════════════════════════════════════════════

    to_tensor_block('t0', offset="(0,0,0)", to="(0,0,0)",
                    width=1, height=40, depth=40,
                    fill=r'\EncoderColor'),
    to_label_below('t0', r'', xshift="-1.5cm", yshift="1.8cm"),

    to_tensor_block('t1', offset="(1,0,0)", to="(t0-east)",
                    width=3, height=40, depth=40,
                    fill=r'\EncoderColor'),
    to_label_below('t1', r'$32 \times 48 \times 48$', xshift="-1.2cm", yshift="1.8cm"),
    to_arrow('t0', 't1'),

    to_tensor_block('t2', offset="(1,0,0)", to="(t1-east)",
                    width=3, height=32, depth=32,
                    fill=r'\EncoderColor'),
    to_label_below('t2', r'', xshift="-1.2cm", yshift="1.5cm"),
    to_arrow('t1', 't2'),

    to_tensor_block('t3', offset="(1,0,0)", to="(t2-east)",
                    width=4, height=32, depth=32,
                    fill=r'\EncoderColor'),
    to_label_below('t3', r'$64 \times 24 \times 24$', xshift="-1cm", yshift="1.5cm"),
    to_arrow('t2', 't3'),

    to_tensor_block('t4', offset="(1,0,0)", to="(t3-east)",
                    width=4, height=24, depth=24,
                    fill=r'\EncoderColor'),
    to_label_below('t4', r'', xshift="-0.8cm", yshift="1.2cm"),
    to_arrow('t3', 't4'),

    to_tensor_block('t5', offset="(1,0,0)", to="(t4-east)",
                    width=5, height=24, depth=24,
                    fill=r'\EncoderColor'),
    to_label_below('t5', r'$128 \times 12 \times 12$', xshift="-0.6cm", yshift="1.2cm"),
    to_arrow('t4', 't5'),

    to_tensor_block('t6', offset="(1,0,0)", to="(t5-east)",
                    width=5, height=16, depth=16,
                    fill=r'\EncoderColor'),
    to_label_below('t6', r'', xshift="-0.5cm", yshift="1cm"),
    to_arrow('t5', 't6'),

    to_tensor_block('t7', offset="(1,0,0)", to="(t6-east)",
                    width=6, height=16, depth=16,
                    fill=r'\EncoderColor'),
    to_label_below('t7', r'$256 \times 6 \times 6$', xshift="-0.5cm", yshift="1cm"),
    to_arrow('t6', 't7'),

    # ════════════════════════════════════════════════════════════════════════
    # VALUE DECODER
    # 三个 AvgPool 输出竖向排列（x对齐，y均匀分布），汇聚到 Concat
    # ════════════════════════════════════════════════════════════════════════

    # 用原始 TikZ 定位三个 pooling 向量方块
    # p3 (1×128) 在中间，与 encoder 同一水平线
    # p2 (1×64) 在上方，p4 (1×256) 在下方
    # 间距 5cm
    r"""
% --- 定位三个 pooling 输出，x 对齐在 t7 右侧，y 均匀分布 ---
% p3 居中（与 encoder 同高），p2 在上，p4 在下
\coordinate (pool_center) at ($(t7-east) + (4,0,0)$);
""",

    # p2: 1×64（上方）
    to_tensor_block('p2', offset="(0,5,0)", to="(pool_center)",
                    width=2, height=6, depth=1,
                    fill=r'\ValueColor'),
    to_label_below('p2', r'$1 \times 64$', xshift="0cm", yshift="0.6cm"),

    # p4: 1×256（中间）
    to_tensor_block('p4', offset="(0,0,0)", to="(pool_center)",
                    width=2, height=12, depth=1,
                    fill=r'\ValueColor'),
    to_label_below('p4', r'$1 \times 256$', xshift="0cm", yshift="0.8cm"),

    # p3: 1×128（下方）
    to_tensor_block('p3', offset="(0,-5,0)", to="(pool_center)",
                    width=2, height=8, depth=1,
                    fill=r'\ValueColor'),
    to_label_below('p3', r'$1 \times 128$', xshift="0cm", yshift="0.6cm"),

    # --- 从 encoder 各 stage 到对应 pooling 输出的箭头（标注 AvgPool）---
    r"""
% t3 → p2: 先向上，再直接向右到 p2-west，仅一个直角拐弯
\draw [-stealth, line width=1.5pt, draw=\edgecolor]
    (t3-north)
    -- (t3-north |- p2-west)
    -- node[midway, above, font=\LabelFont, text=black] {平均池化} (p2-west);
""",

    r"""
% t5 → p3: 先向下，再直接向右到 p3-west，仅一个直角拐弯
\draw [-stealth, line width=1.5pt, draw=\edgecolor]
    (t5-south)
    -- (t5-south |- p3-west)
    -- node[midway, below, font=\LabelFont, text=black] {平均池化} (p3-west);
""",

    r"""
% t7 → p4: 水平向右到 p4
\draw [-stealth, line width=1.5pt, draw=\edgecolor]
    (t7-east) -- node[above, font=\LabelFont, text=black] {平均池化} (p4-west);
""",

    # ── Concat: 64+128+256 = 448 ──
    to_tensor_block('vcat', offset="(4,0,0)", to="(p4-east)",
                    width=3, height=16, depth=1,
                    fill=r'\ValueColor'),
    to_label_below('vcat', r'$1 \times 448$', xshift="0cm", yshift="0.8cm"),

    # 三个 pooling → vcat 的箭头
    r"""
% p2 → vcat
\draw [-stealth, line width=1.5pt, draw=\edgecolor]
    (p2-east) -- (p2-east -| vcat-north) -- (vcat-north);
% p4 → vcat (Concat 标注在这条上，中间那条)
\draw [-stealth, line width=1.5pt, draw=\edgecolor]
    (p4-east) -- node[above, font=\LabelFont, text=black] {拼接} (vcat-west);
% p3 → vcat
\draw [-stealth, line width=1.5pt, draw=\edgecolor]
    (p3-east) -- (p3-east -| vcat-south) -- (vcat-south);
""",

    # ── FC: 448 → 256 ──
    to_tensor_block('v1', offset="(4,0,0)", to="(vcat-east)",
                    width=2, height=12, depth=1,
                    fill=r'\ValueColor'),
    to_label_below('v1', r'$256$', xshift="0cm", yshift="0.8cm"),
    to_arrow_with_label('vcat', 'v1', '线性映射'),

    # ── FC: 256 → 128 ──
    to_tensor_block('v2', offset="(4,0,0)", to="(v1-east)",
                    width=2, height=8, depth=1,
                    fill=r'\ValueColor'),
    to_label_below('v2', r'$128$', xshift="0cm", yshift="0.8cm"),
    to_arrow_with_label('v1', 'v2', '线性映射'),

    # ── FC: 128 → 1 ──
    to_tensor_block('v3', offset="(4,0,0)", to="(v2-east)",
                    width=1, height=4, depth=1,
                    fill=r'\ValueColor'),
    to_label_below('v3', r'$1$', xshift="0cm", yshift="0.6cm"),
    to_arrow_with_label('v2', 'v3', '线性映射'),

    # 图例：先固定一个基准点，避免 current bounding box 在绘制首个图例后继续变化
    r"\coordinate (legend_base) at ($(current bounding box.south)+(0,-2.2cm)$);",
    to_legend_entry('leg_enc', r"($(legend_base)+(-5.5cm,0)$)", r'\EncoderColor', '编码器'),
    to_legend_entry('leg_val', r"($(legend_base)+(6.2cm,0)$)", r'\ValueColor', '价值解码器'),

    to_end(),
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_tex = os.path.join(RESULTS_DIR, 'model_value.tex')
    to_generate(arch, output_tex)
    print(f"[plot_value] generated → {output_tex}")


if __name__ == '__main__':
    main()
