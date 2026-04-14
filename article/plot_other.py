"""
Generate standalone PlotNeuralNet/TikZ diagrams for several RL submodules.

Outputs:
    - model_other_convblock.tex
    - model_other_upblock.tex
    - model_other_fuse.tex
    - model_other_stop_head.tex
"""

from __future__ import annotations

import os
import sys

sys.path.append("/home/chen/punchy/PlotNeuralNet")
from pycore.tikzeng import to_begin, to_cor, to_end, to_generate, to_head


PLOTNN_ROOT = "/home/chen/punchy/PlotNeuralNet"
ARTICLE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ARTICLE_DIR, "results")


def to_tensor_block(name: str, offset: str, to: str, width: int, height: int, depth: int, fill: str) -> str:
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


def to_label_below(node_name: str, text: str, xshift: str = "-0.5cm", yshift: str = "1.2cm") -> str:
    return r"""
\node[below=""" + yshift + r""" of """ + node_name + r"""-south, font=\LabelFont\boldmath, align=center, xshift=""" + xshift + r"""]
    {""" + text + r"""};
"""


def to_arrow_with_label(from_node: str, to_node: str, label: str, *, label_pos: str = "above") -> str:
    return r"""
\draw [-stealth, line width=1.5pt, draw=\edgecolor]
    (""" + from_node + r"""-east)
    -- node[""" + label_pos + r""", font=\LabelFont, text=black] {""" + label + r"""} (""" + to_node + r"""-west);
"""


def to_arrow(from_node: str, to_node: str) -> str:
    return r"""
\draw [-stealth, line width=1.5pt, draw=\edgecolor] (""" + from_node + r"""-east) -- (""" + to_node + r"""-west);
"""


def to_corner_arrow(from_coord: str, to_coord: str) -> str:
    return r"""
\draw [-stealth, line width=1.5pt, draw=\edgecolor] (""" + from_coord + r""") -| (""" + to_coord + r""");
"""


def to_title(left_anchor: str, right_anchor: str, text: str, yshift: str = "2.6cm") -> str:
    return r"""
\node[font=\LabelFont, text=black]
    at ($(""" + left_anchor + r""")!0.5!(""" + right_anchor + r""") + (0,""" + yshift + r""")$)
    {""" + text + r"""};
"""


def common_prefix() -> list[str]:
    return [
        to_head(PLOTNN_ROOT),
        to_cor(),
        r"\usepackage{fontspec}",
        r"\usetikzlibrary{calc}",
        to_begin(),
        r"\def\FeatureColor{\ConvColor}",
        r"\def\VectorColor{rgb:blue,3;green,1;white,4}",
        r"\def\ModuleColor{green!25}",
        r"\newfontfamily\SongtiFont{Noto Serif CJK SC}",
        r"\newcommand{\LabelFont}{\SongtiFont\fontsize{18pt}{20pt}\selectfont}",
    ]


def build_convblock_arch() -> list[str]:
    arch = common_prefix()
    arch.extend(
        [
            to_tensor_block("cb_in", "(0,0,0)", "(0,0,0)", width=2, height=26, depth=26, fill=r"\FeatureColor"),
            to_label_below("cb_in", r"$C_{in} \times H \times W$", xshift="-0.7cm", yshift="1.3cm"),
            to_tensor_block("cb_mid", "(8,0,0)", "(cb_in-east)", width=3, height=26, depth=26, fill=r"\FeatureColor"),
            to_label_below("cb_mid", r"$C_{out} \times H \times W$", xshift="-0.6cm", yshift="1.3cm"),
            to_arrow_with_label("cb_in", "cb_mid", "卷积(3x3)+组归一化+ReLU"),
            to_tensor_block("cb_out", "(8,0,0)", "(cb_mid-east)", width=3, height=26, depth=26, fill=r"\FeatureColor"),
            to_label_below("cb_out", r"$C_{out} \times H \times W$", xshift="-0.6cm", yshift="1.3cm"),
            to_arrow_with_label("cb_mid", "cb_out", "卷积(3x3)+组归一化+ReLU"),
            to_title("cb_in-west", "cb_out-east", "卷积块"),
            to_end(),
        ]
    )
    return arch


def build_upblock_arch() -> list[str]:
    arch = common_prefix()
    arch.extend(
        [
            to_tensor_block("up_in", "(0,0,0)", "(0,0,0)", width=2, height=18, depth=18, fill=r"\FeatureColor"),
            to_label_below("up_in", r"$C_{in} \times H \times W$", xshift="-0.7cm", yshift="1.1cm"),
            to_tensor_block("up_out", "(3,0,0)", "(up_in-east)", width=3, height=28, depth=28, fill=r"\FeatureColor"),
            to_label_below("up_out", r"$C_{out} \times 2H \times 2W$", xshift="-0.7cm", yshift="1.3cm"),
            to_arrow_with_label("up_in", "up_out", r"转置卷积(k=2,s=2)"),
            to_tensor_block("skip_in", "(5,-4.2,0)", "(up_out-east)", width=3, height=28, depth=28, fill=r"\FeatureColor"),
            to_label_below("skip_in", r"$C_{skip} \times 2H \times 2W$", xshift="-0.7cm", yshift="1.3cm"),
            to_tensor_block("cat_out", "(5,0,0)", "(up_out-east)", width=4, height=28, depth=28, fill=r"\ModuleColor"),
            to_label_below("cat_out", r"$(C_{out}+C_{skip}) \times 2H \times 2W$", xshift="-0.8cm", yshift="1.3cm"),
            to_arrow_with_label("up_out", "cat_out", "拼接"),
            to_corner_arrow("skip_in-east", "cat_out-south"),
            to_tensor_block("fuse_out", "(4,0,0)", "(cat_out-east)", width=3, height=28, depth=28, fill=r"\FeatureColor"),
            to_label_below("fuse_out", r"$C_{out} \times 2H \times 2W$", xshift="-0.7cm", yshift="1.3cm"),
            to_arrow_with_label("cat_out", "fuse_out", "融合"),
            to_title("up_in-west", "fuse_out-east", "上采样块"),
            to_end(),
        ]
    )
    return arch


def build_fuse_arch() -> list[str]:
    arch = common_prefix()
    arch.extend(
        [
            to_tensor_block("fuse_in", "(0,0,0)", "(0,0,0)", width=4, height=28, depth=28, fill=r"\ModuleColor"),
            to_label_below("fuse_in", r"$(C_{out}+C_{skip}) \times H \times W$", xshift="-0.8cm", yshift="1.3cm"),
            to_tensor_block("fuse_only_out", "(4,0,0)", "(fuse_in-east)", width=3, height=28, depth=28, fill=r"\FeatureColor"),
            to_label_below("fuse_only_out", r"$C_{out} \times H \times W$", xshift="-0.6cm", yshift="1.3cm"),
            to_arrow_with_label("fuse_in", "fuse_only_out", "卷积(3x3)+组归一化+ReLU"),
            to_title("fuse_in-west", "fuse_only_out-east", "融合"),
            to_end(),
        ]
    )
    return arch


def build_stop_head_arch() -> list[str]:
    arch = common_prefix()
    arch.extend(
        [
            to_tensor_block("sh_in", "(0,0,0)", "(0,0,0)", width=4, height=18, depth=18, fill=r"\FeatureColor"),
            to_label_below("sh_in", r"$256 \times 6 \times 6$", xshift="-0.5cm", yshift="1.1cm"),
            to_tensor_block("sh_pool", "(4,0,0)", "(sh_in-east)", width=2, height=10, depth=10, fill=r"\FeatureColor"),
            to_label_below("sh_pool", r"$256 \times 1 \times 1$", xshift="-0.5cm", yshift="0.9cm"),
            to_arrow_with_label("sh_in", "sh_pool", r"平均池化"),
            to_tensor_block("sh_flat", "(4,0,0)", "(sh_pool-east)", width=3, height=8, depth=1, fill=r"\FeatureColor"),
            to_label_below("sh_flat", r"$1 \times 256$", xshift="-0.3cm", yshift="0.7cm"),
            to_arrow_with_label("sh_pool", "sh_flat", "展平"),
            to_tensor_block("sh_hidden", "(4,0,0)", "(sh_flat-east)", width=2, height=10, depth=1, fill=r"\FeatureColor"),
            to_label_below("sh_hidden", r"$128$", xshift="-0.1cm", yshift="0.7cm"),
            to_arrow_with_label("sh_flat", "sh_hidden", r"线性层"),
            to_tensor_block("sh_out", "(4,0,0)", "(sh_hidden-east)", width=1, height=5, depth=1, fill=r"\FeatureColor"),
            to_label_below("sh_out", r"$1$", xshift="-0.1cm", yshift="0.6cm"),
            to_arrow_with_label("sh_hidden", "sh_out", r"线性层"),
            # to_title("sh_in-west", "sh_out-east", "StopHead"),
            to_end(),
        ]
    )
    return arch


def generate_all() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    figures = {
        "model_other_convblock": build_convblock_arch(),
        "model_other_upblock": build_upblock_arch(),
        "model_other_fuse": build_fuse_arch(),
        "model_other_stop_head": build_stop_head_arch(),
    }

    for basename, arch in figures.items():
        output_tex = os.path.join(RESULTS_DIR, f"{basename}.tex")
        to_generate(arch, output_tex)
        print(f"[plot_other] generated -> {output_tex}")


def main() -> None:
    generate_all()


if __name__ == "__main__":
    main()
