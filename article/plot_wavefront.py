"""
绘制波扩展（Wavefront / BFS）路径搜索示意图。
- 10×10 网格，部分格子为障碍（红色）
- 展示 BFS 从起点扩展到终点的过程（多个时刻）
- 最终帧显示找到的最短路径
输出: article/results/wavefront.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties, fontManager
from collections import deque

_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fontManager.addfont(_FONT_PATH)
_fp = FontProperties(fname=_FONT_PATH)
rcParams['font.family'] = _fp.get_name()
rcParams['axes.unicode_minus'] = False

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# ── 网格定义（10×10，7个障碍）────────────────────────────────────────────────
GRID = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=np.int32)

START = (9, 0)
GOAL  = (0, 9)

# ── 颜色（与 A*/Dijkstra 对比图一致）─────────────────────────────────────────
C_WALL  = '#c0392b'
C_FREE  = '#ecf0f1'
C_PATH  = '#f39c12'
C_START = '#27ae60'
C_GOAL  = '#8e44ad'

WAVE_CMAP = LinearSegmentedColormap.from_list(
    'wave', ['#5dade2', '#1a3a6c'], N=256
)


# ── BFS ───────────────────────────────────────────────────────────────────────

def bfs_steps(grid, start, goal):
    rows, cols = grid.shape
    dist = np.full((rows, cols), -1, dtype=np.int32)
    prev = {}
    dist[start] = 0
    queue = deque([start])
    snapshots = [dist.copy()]

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr,nc] == 1 and dist[nr,nc] == -1:
                dist[nr,nc] = dist[r,c] + 1
                prev[(nr,nc)] = (r,c)
                queue.append((nr,nc))
                if (nr,nc) == goal:
                    snapshots.append(dist.copy())
                    path = []
                    cur = goal
                    while cur != start:
                        path.append(cur)
                        cur = prev[cur]
                    path.append(start)
                    return snapshots, list(reversed(path))
        if not queue or dist[queue[0]] > dist[r,c]:
            snapshots.append(dist.copy())

    return snapshots, []


def draw_frame(ax, grid, dist_map, path, title):
    rows, cols = grid.shape
    max_dist = dist_map.max() if dist_map.max() > 0 else 1
    path_set = set(path)

    for r in range(rows):
        for c in range(cols):
            y = rows - r - 1

            if grid[r, c] == 0:
                color = C_WALL
                text, tc = '', 'white'
            elif (r, c) == START:
                color = C_START
                text, tc = 'S', 'white'
            elif (r, c) == GOAL:
                color = C_GOAL
                text, tc = 'G', 'white'
            elif dist_map[r, c] >= 0:
                intensity = dist_map[r, c] / max_dist
                color = WAVE_CMAP(intensity)
                text = str(dist_map[r, c])
                tc = 'white' if intensity > 0.45 else '#1a3a6c'
            else:
                color = C_FREE
                text, tc = '', '#555'

            rect = mpatches.FancyBboxPatch(
                (c + 0.04, y + 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor='white', linewidth=1.2,
                zorder=2
            )
            ax.add_patch(rect)

            if text:
                ax.text(c + 0.5, y + 0.5, text,
                        ha='center', va='center',
                        fontsize=7, fontweight='bold', color=tc, zorder=4)

    if len(path) > 1:
        xs = [c + 0.5 for r, c in path]
        ys = [rows - r - 1 + 0.5 for r, c in path]
        ax.plot(xs, ys, color='#e74c3c', lw=1.5, zorder=5)
        ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5,
                                   mutation_scale=12),
                    zorder=6)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=4)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    snapshots, path = bfs_steps(GRID, START, GOAL)

    n = len(snapshots)
    # 均匀取 6 帧，最后一帧固定为找到路径
    N_COLS = 4
    indices = [int(i * (n-1) / (N_COLS-1)) for i in range(N_COLS)]
    indices[-1] = n - 1

    titles = []
    for idx in indices:
        if idx == 0:
            titles.append("初始状态")
        elif idx == n-1:
            titles.append(f"步数={snapshots[idx].max()}")
        else:
            titles.append(f"步数={snapshots[idx].max()}")

    fig, axes = plt.subplots(1, N_COLS, figsize=(16, 3.5),
                             gridspec_kw={'bottom': 0.18})

    for ax, idx, title in zip(axes, indices, titles):
        is_final = (idx == n-1)
        draw_frame(ax, GRID, snapshots[idx],
                   path if is_final else [], title)
        if is_final:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#1e8449')
                spine.set_linewidth(3)

    out = os.path.join(RESULTS_DIR, 'wavefront.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"saved → {out}")


if __name__ == '__main__':
    main()
