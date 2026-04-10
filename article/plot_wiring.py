"""
绘制 WiringPolicyNet 架构图，分3张输出：
  1. wiring_encoder.png  — 共享编码器
  2. wiring_policy.png   — 编码器 + Policy Head
  3. wiring_value.png    — 编码器 + Value Head
输出: article/results/
"""
import os
import sys

sys.path.append('/home/chen/punchy/PlotNeuralNet')
from pycore.tikzeng import to_begin, to_cor, to_end, to_generate, to_head

ARTICLE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ARTICLE_DIR, 'results')

C_ENC  = r'\ConvColor'
C_NECK = r'\ConvReluColor'
C_POL  = r'rgb:green,4;white,3'
C_VAL  = r'rgb:blue,3;green,1;white,4'
C_AUX  = r'rgb:magenta,4;black,3;white,4'


# ── 基础绘图函数 ──────────────────────────────────────────────────────────────

def block(name, offset, to, w, h, d, fill):
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


def label(node, text, x="0cm", y="0.65cm"):
    return (
        r"\node[below=" + y + r" of " + node
        + r"-south, font=\bfseries\large\boldmath, align=center, xshift=" + x + r"] {"
        + text + r"};" + "\n"
    )


def arrow(a, b, lbl=None, pos="above"):
    if lbl:
        return (
            r"\draw[-stealth, line width=1.5pt, draw=\edgecolor] ("
            + a + r") -- node[" + pos
            + r", font=\bfseries\small, text=black] {" + lbl + r"} ("
            + b + r");" + "\n"
        )
    return (
        r"\draw[-stealth, line width=1.5pt, draw=\edgecolor] ("
        + a + r") -- (" + b + r");" + "\n"
    )


def elbow(a, b, lbl=None, pos="right"):
    """先垂直到 a |- b，再水平到 b（一个直角弯）。"""
    mid = a + r" |- " + b
    if lbl:
        return (
            r"\draw[-stealth, line width=1.5pt, draw=\edgecolor] ("
            + a + r") -- (" + mid + r") -- node[" + pos
            + r", font=\bfseries\small, text=black] {" + lbl + r"} ("
            + b + r");" + "\n"
        )
    return (
        r"\draw[-stealth, line width=1.5pt, draw=\edgecolor] ("
        + a + r") -- (" + mid + r") -- (" + b + r");" + "\n"
    )


def right_then_down(a, b, lbl=None):
    """先水平到 a -| b，再竖直到 b（一个直角弯）。"""
    mid = a + r" -| " + b
    if lbl:
        return (
            r"\draw[-stealth, line width=1.5pt, draw=\edgecolor] ("
            + a + r") -- node[midway, above, font=\bfseries\small, text=black] {"
            + lbl + r"} (" + mid + r") -- (" + b + r");" + "\n"
        )
    return (
        r"\draw[-stealth, line width=1.5pt, draw=\edgecolor] ("
        + a + r") -- (" + mid + r") -- (" + b + r");" + "\n"
    )


def plain_node(name, at_expr, text, fill, w="2.6cm", h="1.0cm"):
    return (
        r"\node[draw, rounded corners=3pt, fill=" + fill
        + r", minimum width=" + w + r", minimum height=" + h
        + r", font=\bfseries\small, align=center] (" + name
        + r") at " + at_expr + r" {" + text + r"};" + "\n"
    )


# ── 编码器主干（所有图共用的块序列）────────────────────────────────────────────

def encoder_blocks():
    return [
        r"\def\EncColor{"  + C_ENC  + r"}",
        r"\def\NeckColor{" + C_NECK + r"}",
        r"\def\PolColor{"  + C_POL  + r"}",
        r"\def\ValColor{"  + C_VAL  + r"}",
        r"\def\AuxColor{"  + C_AUX  + r"}",

        # 输入
        block('inp', "(0,0,0)", "(0,0,0)", 1, 30, 30, r'\EncColor'),
        label('inp', r'$6 \times 48 \times 48$', x="-0.6cm", y="1.5cm"),

        # Stage1 → 32×H×W
        block('s1', "(4,0,0)", "(inp-east)", 3, 30, 30, r'\EncColor'),
        label('s1', r'$32 \times 48 \times 48$', x="-0.7cm", y="1.5cm"),
        arrow('inp-east', 's1-west', 'ConvBlock'),

        # feat_map → 64×H/2×W/2
        block('feat', "(5,0,0)", "(s1-east)", 4, 23, 23, r'\EncColor'),
        label('feat', r'$64 \times 24 \times 24$',
              x="-0.4cm", y="1.2cm"),
        arrow('s1-east', 'feat-west', r'MaxPool2d + ConvBlock'),

        # neck → 128×H/4×W/4
        block('neck', "(5,0,0)", "(feat-east)", 5, 16, 16, r'\NeckColor'),
        label('neck', r'$128 \times 12 \times 12$',
              x="-0.3cm", y="1.0cm"),
        arrow('feat-east', 'neck-west', r'MaxPool2d + ConvBlock'),

        # global_feat → 1×128
        block('gfeat', "(5,0,0)", "(neck-east)", 4, 8, 1, r'\NeckColor'),
        label('gfeat', r'$1 \times 128$', y="0.55cm"),
        arrow('neck-east', 'gfeat-west', r'AdaptiveAvgPool2d'),
    ]


def policy_head_blocks():
    return [
        # local_feat N×64（从 feat_map 双线性插值）
        block('local', "(10,5,0)", "(feat-east)", 4, 8, 1, r'\PolColor'),
        label('local', r'$N \times 64$', y="0.55cm"),
        elbow('feat-north', 'local-west',
              r'grid\_sample', pos='above'),

        # lamp_coords 节点
        plain_node('lcoords',
                   r"($(local-north)+(0,2.2)$)",
                   r'Lamp Coords\\$N \times 2$',
                   r'\AuxColor'),
        arrow('lcoords.south', 'local-north', r'coord map', pos='left'),

        # global feature broadcast / expand -> N×128
        block('gexp', "(2,0,0)", "(gfeat-east)", 4, 8, 1, r'\PolColor'),
        label('gexp', r'$N \times 128$', y="0.55cm"),
        # elbow('gfeat-east', 'gexp-west',
        #       r'unsqueeze + expand', pos='right'),
        arrow('gfeat-east', 'gexp-west', r'Expand'),

        # combined N×192
        block('cat', "(2,0,0)", "(gexp-east)", 5, 10, 1, r'\PolColor'),
        label('cat', r'$N \times 192$', y="0.6cm"),
        right_then_down('local-east', 'cat-north', r'Concat'),
        # elbow('gexp-east', 'cat-south'),
        arrow('gexp-east', 'cat-west', r''),

        # hidden N×128
        block('mlp', "(3,0,0)", "(cat-east)", 4, 8, 1, r'\PolColor'),
        label('mlp', r'$N \times 128$', y="0.55cm"),
        arrow('cat-east', 'mlp-west', r'Linear'),

        # raw logits N×1
        block('rawlogits', "(2,0,0)", "(mlp-east)", 3, 8, 1, r'\PolColor'),
        label('rawlogits', r'$N \times 1$', y="0.55cm"),
        arrow('mlp-east', 'rawlogits-west', r'Linear'),

        # masked logits N
        block('logits', "(2,0,0)", "(rawlogits-east)", 2, 14, 1, r'\PolColor'),
        label('logits', r'logits $N$', y="0.75cm"),
        arrow('rawlogits-east', 'logits-west', r'squeeze'),

        # action mask
        plain_node('amask',
                   r"($(logits-north)+(0,1.9)$)",
                   r'Action Mask\\$N$',
                   r'\AuxColor'),
        arrow('amask.south', 'logits-north'),
    ]


def value_head_blocks():
    return [
        # vfc 1×64
        block('vfc', "(2.5,0,0)", "(gfeat-east)", 4, 8, 1, r'\ValColor'),
        label('vfc', r'$1 \times 64$', y="0.55cm"),
        # elbow('gfeat-south', 'vfc-north',
        #       r'Linear(128,64)+ReLU', pos='right'),
        arrow('gfeat-east', 'vfc-west', r'Linear+ReLU'),

        # value 1
        block('value', "(2.5,0,0)", "(vfc-east)", 2, 4, 1, r'\ValColor'),
        label('value', r'value $1$', y="0.4cm"),
        arrow('vfc-east', 'value-west', r'Linear'),
    ]


# ── 三张图的 arch 定义 ────────────────────────────────────────────────────────

def make_arch(extra_blocks):
    return (
        [to_head('/home/chen/punchy/PlotNeuralNet'),
         to_cor(),
         r"\usetikzlibrary{calc}",
         to_begin()]
        + encoder_blocks()
        + extra_blocks
        + [to_end()]
    )


arch_encoder = make_arch([])
arch_policy  = make_arch(policy_head_blocks())
arch_value   = make_arch(value_head_blocks())


# ── 生成 + 编译 ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name, arch in [
        ('wiring_encoder', arch_encoder),
        ('wiring_policy',  arch_policy),
        ('wiring_value',   arch_value),
    ]:
        out_tex = os.path.join(RESULTS_DIR, f'{name}.tex')
        to_generate(arch, out_tex)
        print(f"[plot_wiring] generated → {out_tex}")


if __name__ == '__main__':
    main()
