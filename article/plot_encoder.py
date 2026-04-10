"""
使用 PlotNeuralNet 绘制 SharedEncoder 部分架构图。
- 每个方块代表数据张量，维度标在下方
- 网络操作名（ConvBlock / MaxPool）标在箭头上
- MaxPool 不画方块，只在箭头上标注

用法:
    cd /home/chen/punchy/CAD_knowledge_service/article
    bash compile_arch.sh
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


def to_label_below(node_name, text, x_offset="0cm", y_offset="1.5cm"):
    """在方块正下方标注张量维度，支持自定义左右偏移和向下偏移。"""
    return r"""
\node[below=""" + y_offset + r""" of """ + node_name + r"""-south, font=\bfseries\Large\boldmath, align=center, xshift=""" + x_offset + r"""]
    {""" + text + r"""};
"""


def to_arrow_with_label(from_east, to_west, label):
    """带操作名标注的箭头，标注在箭头上方。"""
    return r"""
\draw [-stealth, line width=1.5pt, draw=\edgecolor] (""" + from_east + r"""-east) -- node[above, font=\normalsize\bfseries, text=black] {""" + label + r"""} (""" + to_west + r"""-west);
"""


arch = [
    to_head('/home/chen/punchy/PlotNeuralNet'),
    to_cor(),
    to_begin(),
    # 自定义颜色
    r"\def\BlockColor{\ConvColor}",
    r"\def\BottleneckColor{\ConvReluColor}",

    # ── 6×48×48 ──────────────────────────────────────────────────────────────
    to_tensor_block('t0', offset="(0,0,0)", to="(0,0,0)",
                    width=1, height=40, depth=40),
    to_label_below('t0', r'$6 \times 48 \times 48$', x_offset="-1.5cm", y_offset="1.8cm"),

    # Stage1 ConvBlock → 32×48×48
    to_tensor_block('t1', offset="(5,0,0)", to="(t0-east)",
                    width=3, height=40, depth=40),
    to_label_below('t1', r'$32 \times 48 \times 48$', x_offset="-1.2cm", y_offset="1.8cm"),
    to_arrow_with_label('t0', 't1', 'ConvBlock'),

    # MaxPool → 32×24×24
    to_tensor_block('t2', offset="(5,0,0)", to="(t1-east)",
                    width=3, height=32, depth=32),
    to_label_below('t2', r'$32 \times 24 \times 24$', x_offset="-1.2cm", y_offset="1.5cm"),
    to_arrow_with_label('t1', 't2', 'MaxPool2d'),

    # Stage2 ConvBlock → 64×24×24
    to_tensor_block('t3', offset="(4.5,0,0)", to="(t2-east)",
                    width=4, height=32, depth=32),
    to_label_below('t3', r'$64 \times 24 \times 24$', x_offset="-1cm", y_offset="1.5cm"),
    to_arrow_with_label('t2', 't3', 'ConvBlock'),

    # MaxPool → 64×12×12
    to_tensor_block('t4', offset="(4.8,0,0)", to="(t3-east)",
                    width=4, height=24, depth=24),
    to_label_below('t4', r'$64 \times 12 \times 12$', x_offset="-0.8cm", y_offset="1.2cm"),
    to_arrow_with_label('t3', 't4', 'MaxPool2d'),

    # Stage3 ConvBlock → 128×12×12
    to_tensor_block('t5', offset="(4.2,0,0)", to="(t4-east)",
                    width=5, height=24, depth=24),
    to_label_below('t5', r'$128 \times 12 \times 12$', x_offset="-0.6cm", y_offset="1.2cm"),
    to_arrow_with_label('t4', 't5', 'ConvBlock'),

    # MaxPool → 128×6×6
    to_tensor_block('t6', offset="(4,0,0)", to="(t5-east)",
                    width=5, height=16, depth=16),
    to_label_below('t6', r'$128 \times 6 \times 6$', x_offset="-0.5cm", y_offset="1cm"),
    to_arrow_with_label('t5', 't6', 'MaxPool2d'),

    # Stage4 ConvBlock → 256×6×6
    to_tensor_block('t7', offset="(4,0,0)", to="(t6-east)",
                    width=6, height=16, depth=16,
                    fill=r'\BottleneckColor'),
    to_label_below('t7', r'$256 \times 6 \times 6$', x_offset="-0.5cm", y_offset="1cm"),
    to_arrow_with_label('t6', 't7', 'ConvBlock'),

    to_end(),
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_tex = os.path.join(RESULTS_DIR, 'model_encoder.tex')
    to_generate(arch, output_tex)
    print(f"[plot_model_arch] generated → {output_tex}")


if __name__ == '__main__':
    main()
