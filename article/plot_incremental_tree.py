"""
绘制增量树扩展的状态转移示意图。
- 10×10 网格，部分格子为障碍（红色）
- 展示以下内容：1.策略模型选择了一个灯具位置。2.从被选择灯具位置出发进行BFS，找到接入点。3.使用A*算法从灯具位置出发，找到到达接入点的最短路径。
- 最终帧显示找到最短路径后的新的布线状态，即线路连接好后的状态
输出: article/results/incremental_tree.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from collections import deque
import heapq
from pathlib import Path

# 中文字体设置
_CN_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
CN_FONT = FontProperties(fname=_CN_FONT_PATH, size=22)
rcParams['axes.unicode_minus'] = False

# ── 网格定义（10×10）────────────────────────────────────────────────
GRID = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 4, 1, 1, 1, 1, 4, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 4, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
], dtype=np.int32)

ROWS, COLS = GRID.shape

# ── 颜色定义 ────────────────────────────────────────────────────────
COLOR_OBSTACLE  = '#C0392B'   # 红色：障碍
COLOR_FREE      = "#C4EBC2"   # 浅绿：可布置区域
COLOR_SWITCH    = '#F0A500'   # 橙黄：开关
COLOR_LAMP_CONN = '#2C3E50'   # 深色：已连接灯具
COLOR_LAMP_NEW  = '#E74C3C'   # 红色高亮：待连接灯具
COLOR_WIRE      = '#F39C12'   # 黄色：已有线路
COLOR_BFS       = '#5DADE2'   # 蓝色：BFS 扩展
COLOR_BFS_ENTRY = '#1A5276'   # 深蓝：接入点
COLOR_ASTAR     = '#F39C12'   # 黄色：A* 新路径
COLOR_GRID_LINE = '#BDC3C7'   # 网格线


def route_grid():
    """可通行网格：1=可走，0=障碍"""
    rg = np.where(GRID == 0, 0, 1).astype(np.int32)
    return rg


def astar(rg, start, end):
    """A* 最短路径，返回路径格子列表（含端点）"""
    def h(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    open_set = [(h(start, end), 0, start, [start])]
    visited = {}
    while open_set:
        f, g, cur, path = heapq.heappop(open_set)
        if cur in visited:
            continue
        visited[cur] = g
        if cur == end:
            return path
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cur[0]+dr, cur[1]+dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and rg[nr,nc] == 1:
                ng = g + 1
                heapq.heappush(open_set, (ng + h((nr,nc), end), ng, (nr,nc), path+[(nr,nc)]))
    return None


def bfs_nearest(rg, start, tree_cells):
    """BFS 从 start 出发，找最近的 tree_cells 中的格子，返回 (接入点, BFS扩展格子列表)"""
    if start in tree_cells:
        return start, []
    visited = {start}
    queue = deque([(start, [start])])
    bfs_order = []
    while queue:
        cur, path = queue.popleft()
        bfs_order.append(cur)
        if cur in tree_cells:
            return cur, bfs_order
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cur[0]+dr, cur[1]+dc
            nxt = (nr, nc)
            if nxt in visited:
                continue
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if rg[nr,nc] == 0 and nxt not in tree_cells:
                continue
            visited.add(nxt)
            queue.append((nxt, path+[nxt]))
    return None, bfs_order


def draw_grid(ax, highlight_bfs=None, highlight_entry=None, wire_cells=None,
              new_path=None, new_lamp=None, title=""):
    """绘制一个子图"""
    rg = route_grid()

    for r in range(ROWS):
        for c in range(COLS):
            val = GRID[r, c]
            if val == 0:
                color = COLOR_OBSTACLE
            else:
                color = COLOR_FREE
            rect = plt.Rectangle((c, ROWS-1-r), 1, 1,
                                  facecolor=color, edgecolor=COLOR_GRID_LINE, linewidth=0.5)
            ax.add_patch(rect)

    # BFS 扩展格子
    if highlight_bfs:
        for i, (r, c) in enumerate(highlight_bfs):
            alpha = 0.3 + 0.5 * (i / max(len(highlight_bfs)-1, 1))
            rect = plt.Rectangle((c, ROWS-1-r), 1, 1,
                                  facecolor=COLOR_BFS, alpha=alpha,
                                  edgecolor=COLOR_GRID_LINE, linewidth=0.5)
            ax.add_patch(rect)

    # 接入点高亮
    if highlight_entry:
        r, c = highlight_entry
        rect = plt.Rectangle((c, ROWS-1-r), 1, 1,
                              facecolor=COLOR_BFS_ENTRY, alpha=0.85,
                              edgecolor='white', linewidth=1.5)
        ax.add_patch(rect)

    # 已有线路（黄色线段）
    if wire_cells:
        wire_set = set(wire_cells)
        for (r, c) in wire_cells:
            cx, cy = c + 0.5, ROWS-1-r + 0.5
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if (nr, nc) in wire_set:
                    nx, ny = nc + 0.5, ROWS-1-nr + 0.5
                    ax.plot([cx, nx], [cy, ny], color=COLOR_WIRE, linewidth=3, solid_capstyle='round', zorder=3)

    # A* 新路径
    if new_path and len(new_path) > 1:
        xs = [c + 0.5 for (r, c) in new_path]
        ys = [ROWS-1-r + 0.5 for (r, c) in new_path]
        ax.plot(xs, ys, color=COLOR_ASTAR, linewidth=3, solid_capstyle='round', zorder=4)

    # 重新覆盖节点网格颜色（开关=黄色，灯具=黑色）
    for r in range(ROWS):
        for c in range(COLS):
            val = GRID[r, c]
            if val == 3:
                rect = plt.Rectangle((c, ROWS-1-r), 1, 1,
                                      facecolor=COLOR_SWITCH, edgecolor='white',
                                      linewidth=1.0, zorder=5)
                ax.add_patch(rect)
            elif val == 4:
                facecolor = COLOR_LAMP_NEW if new_lamp == (r, c) else COLOR_LAMP_CONN
                rect = plt.Rectangle((c, ROWS-1-r), 1, 1,
                                      facecolor=facecolor, edgecolor='white',
                                      linewidth=1.0, zorder=5)
                ax.add_patch(rect)

    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontproperties=CN_FONT, pad=8)


# ── 预计算布线状态 ───────────────────────────────────────────────────
rg = route_grid()
switch_pos = (4, 9)
lamp1 = (2, 2)   # 已连接灯具1
lamp2 = (2, 7)   # 已连接灯具2
lamp3 = (7, 7)   # 待连接灯具

# 初始树：开关
tree_cells = {switch_pos}

# 连接 lamp2：手工指定从开关到灯具2的固定路径
# 路径形状为“先向左，再向上”。
entry1 = switch_pos
path1 = [(4, 9), (4, 8), (4, 7), (3, 7), (2, 7)]
for cell in path1:
    tree_cells.add(cell)

# 连接 lamp1：手工指定从灯具2到灯具1的固定路径
# 路径形状为同一行上的水平直连。
entry2 = lamp2
path2 = [(2, 7), (2, 6), (2, 5), (2, 4), (2, 3), (2, 2)]
for cell in path2:
    tree_cells.add(cell)

# 已有线路格子（不含灯具节点本身，只含线路）
existing_wire = set(path1) | set(path2) | {switch_pos}

# 对 lamp3 做 BFS
entry3, bfs_order3 = bfs_nearest(rg, lamp3, tree_cells)
# A* 路径
path3 = astar(rg, lamp3, entry3)

# ── 绘图：4个子图 ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.patch.set_facecolor('white')

# 子图1：初始状态（两个灯具已连接，待连接灯具高亮）
draw_grid(axes[0],
          wire_cells=existing_wire,
          new_lamp=lamp3,
          title="（a）初始状态")

# 子图2：BFS 扩展过程（展示部分BFS格子）
bfs_partial = bfs_order3[:max(1, len(bfs_order3)//2)]
draw_grid(axes[1],
          highlight_bfs=bfs_partial,
          wire_cells=existing_wire,
          new_lamp=lamp3,
          title="（b）BFS 扩展")

# 子图3：找到接入点
draw_grid(axes[2],
          highlight_bfs=bfs_order3,
          highlight_entry=entry3,
          wire_cells=existing_wire,
          new_lamp=lamp3,
          title="（c）找到接入点")

# 子图4：A* 路径规划完成，更新状态
new_wire = existing_wire | set(path3)
draw_grid(axes[3],
          wire_cells=new_wire,
          title="（d）状态转移完成")

legend_elements = [
    mpatches.Patch(facecolor=COLOR_FREE, edgecolor=COLOR_GRID_LINE, label='可通行区域'),
    mpatches.Patch(facecolor=COLOR_OBSTACLE, edgecolor=COLOR_GRID_LINE, label='障碍物'),
    mpatches.Patch(facecolor=COLOR_BFS, edgecolor=COLOR_GRID_LINE, alpha=0.6, label='BFS 扩展区域'),
    mpatches.Patch(facecolor=COLOR_BFS_ENTRY, edgecolor='white', label='接入点'),
    mpatches.Patch(facecolor=COLOR_SWITCH, edgecolor='white', label='开关'),
    mpatches.Patch(facecolor=COLOR_LAMP_CONN, edgecolor='white', label='已连接灯具'),
    mpatches.Patch(facecolor=COLOR_LAMP_NEW, edgecolor='white', label='待连接灯具'),
    Line2D([0], [0], color=COLOR_WIRE, linewidth=3, label='线路路径'),
]
fig.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=4,
    bbox_to_anchor=(0.5, -0.015),
    frameon=False,
    prop=CN_FONT,
    handlelength=1.8,
    handleheight=1.0,
    columnspacing=1.4,
    handletextpad=1.6,
)
plt.tight_layout(rect=[0, 0.2, 1, 1])

out_path = Path(__file__).parent / 'results' / 'incremental_tree.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved to {out_path}")
