"""
Draw a mask-based action sampling diagram with one box per vector.

Pipeline:
  Logits -> Action Mask -> Masked Logits -> Softmax Probs -> Sampled Action

Output: article/results/mask_sampling.tex
"""
import os
import sys

sys.path.append('/home/chen/punchy/PlotNeuralNet')
from pycore.tikzeng import to_head, to_cor, to_begin, to_end, to_generate


ARTICLE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ARTICLE_DIR, "results")


def box(name, offset, to, w, h, d, fill):
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


def vector_text(node, text, color="black"):
    return (
        r"\node[font=\bfseries\small, text=" + color + r", align=center] at ("
        + node + r"-anchor) {" + text + r"};" + "\n"
    )


def label(node, text, yshift="0.95cm"):
    return (
        r"\node[below=" + yshift + r" of " + node
        + r"-south, font=\bfseries\normalsize, align=center] {"
        + text + r"};" + "\n"
    )


def arrow(a, b, txt=None):
    if txt:
        return (
            r"\draw[-stealth, line width=1.5pt, draw=\edgecolor] ("
            + a + r") -- node[above, font=\bfseries\small, text=black] {"
            + txt + r"} (" + b + r");" + "\n"
        )
    return (
        r"\draw[-stealth, line width=1.5pt, draw=\edgecolor] ("
        + a + r") -- (" + b + r");" + "\n"
    )


C_LOGIT = r'rgb:blue,3;white,6'
C_MASK  = r'rgb:green,4;white,3'
C_MSKD  = r'rgb:gray,3;white,6'
C_PROB  = r'rgb:magenta,3;white,5'
C_SAMP  = r'rgb:orange,5;yellow,2;white,4'


arch = [
    to_head('/home/chen/punchy/PlotNeuralNet'),
    to_cor(),
    r"\usetikzlibrary{calc,positioning}",
    to_begin(),

    r"\def\LogitColor{" + C_LOGIT + r"}",
    r"\def\MaskColor{" + C_MASK + r"}",
    r"\def\MaskedColor{" + C_MSKD + r"}",
    r"\def\ProbColor{" + C_PROB + r"}",
    r"\def\SampleColor{" + C_SAMP + r"}",

    box('logits', "(0,0,0)", "(0,0,0)", 7.2, 5.2, 1.0, r'\LogitColor'),
    vector_text('logits', r'$[2.1,\ 0.4,\ 1.7,\ -0.2,\ 3.0,\ 0.8]$'),
    label('logits', 'Logits'),

    box('mask', "(4.5,0,0)", "(logits-east)", 7.2, 5.2, 1.0, r'\MaskColor'),
    vector_text('mask', r'$[1,\ 0,\ 1,\ 0,\ 1,\ 1]$'),
    label('mask', 'Action Mask'),
    arrow('logits-east', 'mask-west', 'Mask'),

    box('masked', "(4.5,0,0)", "(mask-east)", 7.6, 5.2, 1.0, r'\MaskedColor'),
    vector_text('masked', r'$[2.1,\ -\infty,\ 1.7,\ -\infty,\ 3.0,\ 0.8]$'),
    label('masked', 'Masked Logits'),
    arrow('mask-east', 'masked-west', 'Apply'),

    box('prob', "(4.5,0,0)", "(masked-east)", 7.6, 5.2, 1.0, r'\ProbColor'),
    vector_text('prob', r'$[0.20,\ 0.00,\ 0.14,\ 0.00,\ 0.53,\ 0.13]$'),
    label('prob', 'Softmax Probs'),
    arrow('masked-east', 'prob-west', 'Softmax'),

    box('sample', "(4.5,0,0)", "(prob-east)", 7.0, 5.2, 1.0, r'\SampleColor'),
    vector_text('sample', r'$[0,\ 0,\ 0,\ 0,\ 1,\ 0]$'),
    label('sample', 'Sampled Action'),
    arrow('prob-east', 'sample-west', 'Sample'),

    to_end(),
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_tex = os.path.join(RESULTS_DIR, 'mask_sampling.tex')
    to_generate(arch, out_tex)
    print(f"[plot_mask_sampling] generated -> {out_tex}")


if __name__ == '__main__':
    main()
