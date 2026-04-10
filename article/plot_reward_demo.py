"""
绘制布局奖励示意图：展示灯具从随机到均匀分布过程中势能的变化。
- 6×6 全规则房间
- 可布置区域颜色由势能决定：势能大=深蓝，势能小=浅蓝
- 灯具用黑色方块表示
- 输出: article/reward_demo.png
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties, fontManager

# 注册并使用系统 Noto Sans CJK 字体
_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fontManager.addfont(_FONT_PATH)
_fp = FontProperties(fname=_FONT_PATH)
rcParams['font.family'] = _fp.get_name()
rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from RL.reward import RoomState, RewardCalculator, RewardConfig

# ── 房间定义：6×6 全规则区域 ──────────────────────────────────────────────────
BASE_ROOM = np.ones((6, 6), dtype=np.int32)

# 4个时刻的灯具位置（4盏灯，逐步从分散到均匀）
LAMP_CONFIGS = [
    [(0, 0), (0, 5), (5, 0), (1, 3)],   # t=0: 分散，不均匀
    [(0, 1), (1, 4), (4, 0), (5, 4)],   # t=1
    [(1, 0), (0, 4), (5, 1), (4, 5)],   # t=2: 接近均匀
    [(1, 1), (1, 4), (4, 1), (4, 4)],   # t=3: 均匀四角对称
]

TITLES = ["$t = 0$", "$t = 1$", "$t = 2$", "$t = 3$"]
# 可布置区域：势能小=浅，势能大=深（绿色系）
POTENTIAL_CMAP = LinearSegmentedColormap.from_list(
    'pot', ["#c9e0f0", "#5dade2", "#03255a"], N=256
)
LAMP_COLOR = '#1c1c1c'


def build_room(lamp_positions):
    room = BASE_ROOM.copy()
    for r, c in lamp_positions:
        if room[r, c] == 1:
            room[r, c] = 4
    return room


def compute_potential_map(state: RoomState) -> np.ndarray:
    """每个可布置格子到最近灯的平方距离，wall=NaN。"""
    rows, cols = state.shape
    pot_map = np.full((rows, cols), np.nan)
    placeable_cells = np.argwhere(state.placeable_mask)
    if placeable_cells.size == 0:
        return pot_map
    lamps = state.lamp_positions
    if not lamps:
        diag_sq = float((rows - 1) ** 2 + (cols - 1) ** 2)
        for r, c in placeable_cells:
            pot_map[r, c] = diag_sq
        return pot_map
    lamp_coords = np.asarray(lamps, dtype=np.float32)
    cell_coords = placeable_cells.astype(np.float32)
    diff = cell_coords[:, None, :] - lamp_coords[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    nearest = np.min(dist_sq, axis=1)
    for idx, (r, c) in enumerate(placeable_cells):
        pot_map[r, c] = nearest[idx]
    return pot_map


def draw_room(ax, room_matrix, pot_map, title, total_potential, vmax_global):
    rows, cols = room_matrix.shape

    for r in range(rows):
        for c in range(cols):
            val = room_matrix[r, c]
            y = rows - r - 1  # 翻转 y 轴，使 row=0 在顶部

            if val == 4:
                color = LAMP_COLOR
            else:
                intensity = pot_map[r, c] / (vmax_global + 1e-9)
                intensity = float(np.clip(intensity, 0.0, 1.0))
                color = POTENTIAL_CMAP(intensity)

            rect = mpatches.FancyBboxPatch(
                (c + 0.06, y + 0.06), 0.88, 0.88,
                boxstyle="round,pad=0.04",
                facecolor=color, edgecolor='white', linewidth=1.8,
                zorder=2
            )
            ax.add_patch(rect)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(
        f"{title}\n布局势能 $\\Phi={total_potential:.1f}$",
        fontsize=12, pad=6
    )


def main():
    config = RewardConfig()
    calculator = RewardCalculator(config)

    rooms, states, pot_maps, potentials = [], [], [], []
    for lamp_pos in LAMP_CONFIGS:
        room = build_room(lamp_pos)
        state = RoomState.from_encoded_matrix(room)
        pot_map = compute_potential_map(state)
        rooms.append(room)
        states.append(state)
        pot_maps.append(pot_map)
        potentials.append(calculator.potential(state))

    vmax_global = float(np.nanmax(pot_maps[0]))

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5),
                             gridspec_kw={'bottom': 0.22})

    for ax, room, pot_map, title, pot in zip(
            axes, rooms, pot_maps, TITLES, potentials):
        draw_room(ax, room, pot_map, title, pot, vmax_global)

    # 色条
    sm = plt.cm.ScalarMappable(cmap=POTENTIAL_CMAP, norm=plt.Normalize(0, vmax_global))
    sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='horizontal',
    #                     fraction=0.025, pad=0.08, aspect=50)
    # cbar.set_label(r'网格势能 $\phi_i$（到最近灯具的距离平方）', fontsize=10)
    # cbar.set_ticks([0, vmax_global * 0.5, vmax_global])
    # cbar.set_ticklabels(['低势能（浅色）', '中', '高势能（深色）'])

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reward_demo.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"saved → {out_path}")


if __name__ == '__main__':
    main()
