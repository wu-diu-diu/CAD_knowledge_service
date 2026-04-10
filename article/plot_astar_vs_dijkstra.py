"""
绘制 A* 与 Dijkstra 算法对比示意图。
- 同一个 5×5 网格，相同起点/终点
- 上排 4 帧：Dijkstra 扩展过程
- 下排 4 帧：A* 扩展过程
- 格子内显示 g 值（已走代价），A* 帧额外用色调体现 f=g+h
输出: article/results/astar_vs_dijkstra.png
"""
import os
import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties, fontManager

_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fontManager.addfont(_FONT_PATH)
_fp = FontProperties(fname=_FONT_PATH)
rcParams['font.family'] = _fp.get_name()
rcParams['axes.unicode_minus'] = False

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# 竖向障碍墙在第5列，缺口在第1行（顶部）
# 起点(9,0)左下，终点(0,9)右上
# Dijkstra 从左下均匀扩展，会大量探索左侧和底部无用区域
# A* 启发函数指向右上，优先沿对角线推进，几乎不碰左下区域
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

START = (9, 0)   # 左下
GOAL  = (0, 9)   # 右上

C_WALL   = '#c0392b'
C_FREE   = '#ecf0f1'   # 未探索：浅灰（几乎无色）
C_PATH   = '#f39c12'
C_START  = '#27ae60'
C_GOAL   = '#8e44ad'

# 已扩展（closed）：深蓝色系，距离越远越深
CLOSED_CMAP = LinearSegmentedColormap.from_list('closed', ["#5dade2", "#1a3a6c"], N=256)
# 待扩展（open）：浅蓝色系，整体比 closed 浅，表示波前边缘
OPEN_CMAP   = LinearSegmentedColormap.from_list('open',   ["#d6eaf8", "#aed6f1"], N=256)


def heuristic(a, b):
    """曼哈顿距离启发函数。"""
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def neighbors(r, c, grid):
    rows, cols = grid.shape
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr,nc] == 1:
            yield nr, nc


# ── Dijkstra ──────────────────────────────────────────────────────────────────

def run_dijkstra(grid, start, goal):
    """返回完整扩展序列和路径。"""
    INF = float('inf')
    rows, cols = grid.shape
    g = np.full((rows, cols), INF)
    g[start] = 0
    prev = {}
    closed = set()
    heap = [(0, start)]
    expand_order = []

    while heap:
        cost, node = heapq.heappop(heap)
        if node in closed:
            continue
        closed.add(node)
        expand_order.append((node, dict(
            g=g.copy(),
            closed=set(closed),
            open_set={n for _, n in heap if n not in closed},
        )))
        if node == goal:
            break
        for nr, nc in neighbors(*node, grid):
            ng = cost + 1
            if ng < g[nr, nc]:
                g[nr, nc] = ng
                prev[(nr, nc)] = node
                heapq.heappush(heap, (ng, (nr, nc)))

    path = []
    cur = goal
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    return expand_order, list(reversed(path))


# ── A* ────────────────────────────────────────────────────────────────────────

def run_astar(grid, start, goal):
    """返回完整扩展序列和路径。"""
    INF = float('inf')
    rows, cols = grid.shape
    g = np.full((rows, cols), INF)
    h = np.array([[heuristic((r, c), goal) for c in range(cols)]
                  for r in range(rows)], dtype=np.float32)
    g[start] = 0
    prev = {}
    closed = set()
    heap = [(float(h[start]), start)]
    expand_order = []

    while heap:
        f, node = heapq.heappop(heap)
        if node in closed:
            continue
        closed.add(node)
        expand_order.append((node, dict(
            g=g.copy(), h=h,
            closed=set(closed),
            open_set={n for _, n in heap if n not in closed},
        )))
        if node == goal:
            break
        r, c = node
        for nr, nc in neighbors(r, c, grid):
            ng = g[r, c] + 1
            if ng < g[nr, nc]:
                g[nr, nc] = ng
                prev[(nr, nc)] = node
                heapq.heappush(heap, (ng + h[nr, nc], (nr, nc)))

    path = []
    cur = goal
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    return expand_order, list(reversed(path))


def sample_frames(expand_order, path, goal, n_cols, is_astar=False,
                  shared_indices=None):
    """
    按 shared_indices（绝对步数）采样快照。
    若某帧超出该算法的总步数，则复用最后一帧（已找到终点）。
    """
    total = len(expand_order)
    result = []
    for idx in shared_indices:
        actual = min(idx, total - 1)
        node, state = expand_order[actual]
        found = (actual == total - 1)   # 该算法已到达终点
        frame = {
            'g': state['g'],
            'closed': state['closed'],
            'open_set': state['open_set'],
            'path': path if found else [],
            'current': node,
            'found': found,
            'steps': actual + 1,        # 已扩展节点数
        }
        if is_astar:
            frame['h'] = state['h']
        result.append(frame)
    return result


# ── 绘制单帧 ──────────────────────────────────────────────────────────────────

def draw_frame(ax, grid, frame, title, use_heuristic=False):
    rows, cols = grid.shape
    g_map    = frame['g']
    closed   = frame['closed']
    open_set = frame['open_set']
    path     = frame['path']
    path_set = set(path)
    found    = frame['found']

    finite_g = g_map[np.isfinite(g_map) & (g_map > 0)]
    g_max = finite_g.max() if finite_g.size > 0 else 1.0

    if use_heuristic:
        h_map = frame['h']
        f_map = np.where(np.isfinite(g_map), g_map + h_map, np.inf)
        finite_f = f_map[np.isfinite(f_map) & (f_map > 0)]
        f_max = finite_f.max() if finite_f.size > 0 else 1.0

    for r in range(rows):
        for c in range(cols):
            y = rows - r - 1
            pos = (r, c)

            if grid[r, c] == 0:
                color = C_WALL
                text, tc = '', 'white'
            elif pos == START:
                color = C_START
                text, tc = 'S', 'white'
            elif pos == GOAL:
                color = C_GOAL
                text, tc = 'G', 'white'
            elif pos in closed:
                intensity = g_map[r, c] / g_max if g_max > 0 else 0
                color = CLOSED_CMAP(min(intensity, 1.0))
                text, tc = '', 'white'
            elif pos in open_set and np.isfinite(g_map[r, c]):
                if use_heuristic:
                    intensity = f_map[r, c] / f_max if f_max > 0 else 0
                else:
                    intensity = g_map[r, c] / g_max if g_max > 0 else 0
                color = OPEN_CMAP(min(intensity, 1.0))
                text, tc = '', '#1a5c2a'
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
                        fontsize=9, fontweight='bold', color=tc, zorder=4)

    # 路径箭头
    if found and len(path) > 1:
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

    # 已找到终点时加绿色边框
    if found:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#1e8449')
            spine.set_linewidth(3)

    ax.set_title(title, fontsize=10, fontweight='bold', pad=4)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    N_COLS = 4   # 每行 4 帧

    dijk_order, dijk_path = run_dijkstra(GRID, START, GOAL)
    astar_order, astar_path = run_astar(GRID, START, GOAL)

    d_total = len(dijk_order)   # Dijkstra 总步数（更多）
    a_total = len(astar_order)  # A* 总步数（更少）

    # 固定采样点：初始状态(0)、第6步(5)、第12步(11)、第19步(18)
    shared = [0, 5, 11, 18]

    dijk_frames = sample_frames(dijk_order, dijk_path, GOAL,
                                N_COLS, is_astar=False, shared_indices=shared)
    astar_frames = sample_frames(astar_order, astar_path, GOAL,
                                 N_COLS, is_astar=True,  shared_indices=shared)

    fig, axes = plt.subplots(2, N_COLS, figsize=(10, 6))
    plt.subplots_adjust(wspace=0.05)

    for col in range(N_COLS):
        df = dijk_frames[col]
        af = astar_frames[col]
        step_idx = shared[col] + 1

        d_title = (f"步数={df['steps']}")
        a_title = (f"步数={af['steps']}")

        draw_frame(axes[0, col], GRID, df, d_title, use_heuristic=False)
        draw_frame(axes[1, col], GRID, af, a_title, use_heuristic=True)

    # 行标注
    for row, txt in enumerate(['Dijkstra', 'A*']):
        axes[row, 0].set_ylabel(txt, fontsize=13, fontweight='bold',
                                rotation=90, labelpad=6)
        axes[row, 0].yaxis.set_label_position('left')
        axes[row, 0].yaxis.label.set_visible(True)

    # 图例
    # legend_patches = [
    #     mpatches.Patch(color=C_WALL,           label='障碍'),
    #     mpatches.Patch(color=C_FREE,           label='未探索'),
    #     mpatches.Patch(color=CLOSED_CMAP(0.5), label='已扩展'),
    #     mpatches.Patch(color=OPEN_CMAP(0.4),   label='待扩展'),
    #     mpatches.Patch(color=C_PATH,           label='最短路径'),
    #     mpatches.Patch(color=C_START,          label='起点 S'),
    #     mpatches.Patch(color=C_GOAL,           label='终点 G'),
    # ]
    # fig.legend(handles=legend_patches, loc='lower center', ncol=7,
    #            fontsize=9, bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.text(-0.01, 0.75, 'Dijkstra', fontsize=9,
             fontweight='bold', va='center', ha='left', color='#1e8449')
    fig.text(0.01, 0.27, 'A*', fontsize=9,
             fontweight='bold', va='center', ha='left', color='#1e8449')

    plt.tight_layout(rect=[0.04, 0.06, 1, 1])
    out = os.path.join(RESULTS_DIR, 'astar_vs_dijkstra.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"saved → {out}")
    print(f"  Dijkstra 总步数: {d_total},  A* 总步数: {a_total}")


if __name__ == '__main__':
    main()
