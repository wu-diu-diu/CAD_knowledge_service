"""
使用 PlotNeuralNet 绘制 SharedEncoder + PolicyDecoder 架构图。
- Encoder: 6×48×48 → 256×6×6
- PolicyDecoder: 256×6×6 → spatial_head (1×48×48) + stop_head (1)

用法:
    cd /home/chen/punchy/CAD_knowledge_service/article
    python plot_policy.py
    pdflatex -interaction=nonstopmode model_policy.tex
    pdftoppm -r 300 -png model_policy.pdf model_policy_page
    mv model_policy_page-1.png model_policy.png
"""
import sys
import os

sys.path.append('/home/chen/punchy/PlotNeuralNet')
from pycore.tikzeng import to_head, to_cor, to_begin, to_end, to_generate


ARTICLE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ARTICLE_DIR, "results")


def to_tensor_block(name, offset, to, width, height, depth, fill=r'\ConvColor'):
    """单层方块，表示一个数据张量，不带任何文字标注。"""
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
    """在方块正下方标注张量维度，支持自定义左偏移和下偏移。"""
    return r"""
\node[below=""" + yshift + r""" of """ + node_name + r"""-south, font=\bfseries\large, align=center, xshift=""" + xshift + r"""]
    {""" + text + r"""};
"""


def to_arrow_with_label(from_east, to_west, label):
    """带操作名标注的箭头，标注在箭头上方。"""
    return r"""
\draw [-stealth, line width=1.5pt, draw=\edgecolor] (""" + from_east + r"""-east) -- node[above, font=\normalsize\bfseries, text=black] {""" + label + r"""} (""" + to_west + r"""-west);
"""


def to_arrow(from_east, to_west):
    """仅绘制连接箭头，不放文字。"""
    return r"""
\draw [-stealth, line width=1.5pt, draw=\edgecolor] (""" + from_east + r"""-east) -- (""" + to_west + r"""-west);
"""


def to_right_angle_arrow(from_coord, to_coord, label=None):
    """
    绘制直角拐弯箭头：先向下，然后指向目标节点。

    参数：
        from_coord: 起点坐标或节点锚点，例如 "t7-east"、"stop_head-west"
        to_coord: 终点坐标或节点锚点，例如 "stop_head-west"、"fuse1-east"
        label: 水平段中部显示的文字，默认为空

    说明：
        路径形状固定为“下 -> 右”。
        中间拐点使用 `from_coord |- to_coord` 自动计算，因此无需手动设置纵向距离。
    """
    label_tex = ""
    if label:
        label_tex = (
            r" node[midway, above, font=\Large\bfseries, text=black] {"
            + label
            + r"} "
        )

    return r"""
\draw [-stealth, line width=1.5pt, draw=\edgecolor]
    (""" + from_coord + r""")
    -- (""" + from_coord + r""" |- """ + to_coord + r""")
    --""" + label_tex + r"""(""" + to_coord + r""");
"""


def to_up_right_down_arrow(from_coord, to_coord, up_dist="2cm", label=None):
    """
    绘制折线箭头：先向上，再向右，最后向下连接到目标点。

    参数：
        from_coord: 起点坐标或节点锚点，例如 "t7-east"、"$(t7-east)+(0.5,0)$"
        to_coord: 终点坐标或节点锚点，例如 "stop_head-west"
        up_dist: 先向上的高度，例如 "2cm"、"1.2cm"
        label: 水平段中部显示的文字，默认为空

    说明：
        路径形状固定为“上 -> 右 -> 下”。
        若设置 label，则标签显示在中间水平段的正上方。
    """
    label_tex = ""
    if label:
        label_tex = (
            r" node[midway, above, font=\Large\bfseries, text=black] {"
            + label
            + r"} "
        )

    return r"""
\draw [line width=1.5pt, draw=\edgecolor]
    (""" + from_coord + r""")
    -- ++(0,""" + up_dist + r""") coordinate (urd_mid)
    --""" + label_tex + r"""(urd_mid -| """ + to_coord + r""")
    -- (""" + to_coord + r""");
"""


def to_skip_connection(from_node, to_node):
    """绘制 skip connection（虚线）。"""
    return r"""
\draw [connection, dashed] (""" + from_node + r"""-east) -- (""" + to_node + r"""-west);
"""


def to_concat_arrow(from_main, from_skip, to_node, height="1.5cm", width="2cm"):
    """
    绘制 concat 操作：
    - 两条竖线分别从两个输入节点向上
    - 一条横线连接这两条竖线
    - 一条箭头从横线中点指向目标节点

    参数：
        from_main: 主输入节点名
        from_skip: skip 输入节点名
        to_node: 目标节点名
        height: 竖线向上的高度（默认 1.5cm）
        width: 横线跨越的宽度（默认 2cm）
    """
    return r"""
\draw [line width=1.5pt, draw=\edgecolor] (""" + from_main + r"""-north) -- ++(0,""" + height + r""");
\draw [line width=1.5pt, draw=\edgecolor] (""" + from_skip + r"""-north) -- ++(0,""" + height + r""");
\draw [line width=1.5pt, draw=\edgecolor] ($(""" + from_main + r"""-north) + (0,""" + height + r""")$) -- ++(""" + width + r""",0);
\draw [-stealth, line width=1.5pt, draw=\edgecolor] ($(""" + from_main + r"""-north) + (""" + width + r"""/2,""" + height + r""")$) -- (""" + to_node + r"""-north);
"""


arch = [
    to_head('/home/chen/punchy/PlotNeuralNet'),
    to_cor(),
    r"\usetikzlibrary{calc}",
    to_begin(),
    # 自定义颜色
    r"\def\EncoderColor{\ConvColor}",
    r"\def\BottleneckColor{\ConvReluColor}",
    r"\def\DecoderColor{green!30}",

    # ════════════════════════════════════════════════════════════════════════
    # ENCODER
    # ════════════════════════════════════════════════════════════════════════

    # ── 6×48×48 ──────────────────────────────────────────────────────────────
    to_tensor_block('t0', offset="(0,0,0)", to="(0,0,0)",
                    width=1, height=40, depth=40,
                    fill=r'\EncoderColor'),
    to_label_below('t0', r'', xshift="-1.5cm", yshift="1.8cm"),

    # Stage1 ConvBlock → 32×48×48
    to_tensor_block('t1', offset="(1,0,0)", to="(t0-east)",
                    width=3, height=40, depth=40,
                    fill=r'\EncoderColor'),
    to_label_below('t1', r'$32 \times 48 \times 48$', xshift="-1.2cm", yshift="1.8cm"),
    to_arrow('t0', 't1'),

    # MaxPool → 32×24×24
    to_tensor_block('t2', offset="(1,0,0)", to="(t1-east)",
                    width=3, height=32, depth=32,
                    fill=r'\EncoderColor'),
    to_label_below('t2', r'', xshift="-1.2cm", yshift="1.5cm"),
    to_arrow('t1', 't2'),

    # Stage2 ConvBlock → 64×24×24
    to_tensor_block('t3', offset="(1,0,0)", to="(t2-east)",
                    width=4, height=32, depth=32,
                    fill=r'\EncoderColor'),
    to_label_below('t3', r'$64 \times 24 \times 24$', xshift="-1cm", yshift="1.5cm"),
    to_arrow('t2', 't3'),

    # MaxPool → 64×12×12
    to_tensor_block('t4', offset="(1,0,0)", to="(t3-east)",
                    width=4, height=24, depth=24,
                    fill=r'\EncoderColor'),
    to_label_below('t4', r'', xshift="-0.8cm", yshift="1.2cm"),
    to_arrow('t3', 't4'),

    # Stage3 ConvBlock → 128×12×12
    to_tensor_block('t5', offset="(1,0,0)", to="(t4-east)",
                    width=5, height=24, depth=24,
                    fill=r'\EncoderColor'),
    to_label_below('t5', r'$128 \times 12 \times 12$', xshift="-0.6cm", yshift="1.2cm"),
    to_arrow('t4', 't5'),

    # MaxPool → 128×6×6
    to_tensor_block('t6', offset="(1,0,0)", to="(t5-east)",
                    width=5, height=16, depth=16,
                    fill=r'\EncoderColor'),
    to_label_below('t6', r'', xshift="-0.5cm", yshift="1cm"),
    to_arrow('t5', 't6'),

    # Stage4 ConvBlock → 256×6×6
    to_tensor_block('t7', offset="(1,0,0)", to="(t6-east)",
                    width=6, height=16, depth=16,
                    fill=r'\BottleneckColor'),
    to_label_below('t7', r'$256 \times 6 \times 6$', xshift="-0.5cm", yshift="1cm"),
    to_arrow('t6', 't7'),

    # ════════════════════════════════════════════════════════════════════════
    # POLICY DECODER
    # ════════════════════════════════════════════════════════════════════════

    # ── Up3: 256×6×6 → 128×12×12 ──
    to_tensor_block('up3_out', offset="(3,0,0)", to="(t7-east)",
                    width=5, height=24, depth=24,
                    fill=r'\DecoderColor'),
    to_label_below('up3_out', r'$128 \times 12 \times 12$', xshift="-0.6cm", yshift="1.2cm"),
    to_arrow_with_label('t7', 'up3_out', 'Up'),
    to_up_right_down_arrow(from_coord="t5-north", to_coord="up3_out-north", up_dist="2cm", label="Concat"),

    # ── Concat3: 128×12×12 + 128×12×12 → 256×12×12 ──
    to_tensor_block('concat3', offset="(2,0,0)", to="(up3_out-east)",
                    width=6, height=24, depth=24,
                    fill=r'\DecoderColor'),
    to_label_below('concat3', r'$256 \times 12 \times 12$', xshift="-0.5cm", yshift="1.2cm"),
    to_arrow('up3_out', 'concat3'),

    # ── Fuse3: 256×12×12 → 128×12×12 ──
    to_tensor_block('fuse3', offset="(3,0,0)", to="(concat3-east)",
                    width=5, height=24, depth=24,
                    fill=r'\DecoderColor'),
    to_label_below('fuse3', r'$128 \times 12 \times 12$', xshift="-0.6cm", yshift="1.2cm"),
    to_arrow_with_label('concat3', 'fuse3', 'Fuse'),

    # ── Up2: 128×12×12 → 64×24×24 ──
    to_tensor_block('up2_out', offset="(3.2,0,0)", to="(fuse3-east)",
                    width=4, height=32, depth=32,
                    fill=r'\DecoderColor'),
    to_label_below('up2_out', r'$64 \times 24 \times 24$', xshift="-1cm", yshift="1.5cm"),
    to_arrow_with_label('fuse3', 'up2_out', 'Up'),

    # ── Concat2: 64×24×24 + 64×24×24 → 128×24×24 ──
    to_tensor_block('concat2', offset="(2,0,0)", to="(up2_out-east)",
                    width=5, height=32, depth=32,
                    fill=r'\DecoderColor'),
    to_label_below('concat2', r'$128 \times 24 \times 24$', xshift="-0.8cm", yshift="1.5cm"),
    to_arrow('up2_out', 'concat2'),
    to_up_right_down_arrow(from_coord="t3-north", to_coord="up2_out-north", up_dist="2cm",  label="Concat"),

    # ── Fuse2: 128×24×24 → 64×24×24 ──
    to_tensor_block('fuse2', offset="(4,0,0)", to="(concat2-east)",
                    width=4, height=32, depth=32,
                    fill=r'\DecoderColor'),
    to_label_below('fuse2', r'$64 \times 24 \times 24$', xshift="-1cm", yshift="1.5cm"),
    to_arrow_with_label('concat2', 'fuse2', 'Fuse'),

    # ── Up1: 64×24×24 → 32×48×48 ──
    to_tensor_block('up1_out', offset="(4,0,0)", to="(fuse2-east)",
                    width=3, height=40, depth=40,
                    fill=r'\DecoderColor'),
    to_label_below('up1_out', r'$32 \times 48 \times 48$', xshift="-1.2cm", yshift="1.8cm"),
    to_arrow_with_label('fuse2', 'up1_out', 'Up'),

    # ── Concat1: 32×48×48 + 32×48×48 → 64×48×48 ──
    to_tensor_block('concat1', offset="(3,0,0)", to="(up1_out-east)",
                    width=4, height=40, depth=40,
                    fill=r'\DecoderColor'),
    to_label_below('concat1', r'$64 \times 48 \times 48$', xshift="-1cm", yshift="1.8cm"),
    to_arrow('up1_out', 'concat1'),
    to_up_right_down_arrow(from_coord="t1-north", to_coord="up1_out-north", up_dist="2cm", label="Concat"),

    # ── Fuse1: 64×48×48 → 32×48×48 ──
    to_tensor_block('fuse1', offset="(4,0,0)", to="(concat1-east)",
                    width=3, height=40, depth=40,
                    fill=r'\DecoderColor'),
    to_label_below('fuse1', r'$32 \times 48 \times 48$', xshift="-1.2cm", yshift="1.8cm"),
    to_arrow_with_label('concat1', 'fuse1', 'Fuse'),

    # ── spatial_head: 32×48×48 → 1×48×48 ──
    to_tensor_block('spatial_head', offset="(3,0,0)", to="(fuse1-east)",
                    width=1, height=40, depth=40,
                    fill=r'\DecoderColor'),
    to_label_below('spatial_head', r'$1 \times 48 \times 48$', xshift="-1.5cm", yshift="1.8cm"),
    to_arrow('fuse1', 'spatial_head'),

    # ── stop_head: 256×6×6 → 1 ──
    to_tensor_block('stop_head', offset="(38,-8,0)", to="(t7-east)",
                    width=1, height=6, depth=6,
                    fill=r'\DecoderColor'),
    to_label_below('stop_head', r'$1$', xshift="-0.5cm", yshift="1cm"),
    to_right_angle_arrow('t7-south', 'stop_head-west', label='Stop-head'),
    to_end(),
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_tex = os.path.join(RESULTS_DIR, 'model_policy.tex')
    to_generate(arch, output_tex)
    print(f"[plot_policy] generated → {output_tex}")


if __name__ == '__main__':
    main()
