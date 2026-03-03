"""
步骤8: 线路拓扑与几何寻路

目标:
1) 基于最小生成树(MST)生成从开关到所有灯具的无环连线拓扑;
2) 在离散矩阵中使用 A* 寻路生成横平竖直线路;
3) 输出网格/像素/CAD三套坐标并生成按房间拆分的可视化。
"""

from __future__ import annotations

import heapq
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .coordinate_converter import pixel_to_cad
from .lighting_layout import (
    _draw_legend,
    _draw_room_grid_on_overlay,
    _grid_cell_to_pixel,
    _is_point,
    _sanitize_filename,
)


GridPoint = Tuple[int, int]


@dataclass
class _EdgeCandidate:
    cost: float
    i: int
    j: int
    path: List[GridPoint]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _normalize_grid_point(value: Any) -> Optional[GridPoint]:
    if not _is_point(value):
        return None
    return int(value[0]), int(value[1])


def _is_passable_cell(
    grid: np.ndarray,
    p: GridPoint,
    start: GridPoint,
    end: GridPoint,
) -> bool:
    r, c = p
    if r < 0 or c < 0 or r >= grid.shape[0] or c >= grid.shape[1]:
        return False
    if p == start or p == end:
        return int(grid[r, c]) != 0
    return int(grid[r, c]) == 1


def _astar_route(
    grid: np.ndarray,
    start: GridPoint,
    end: GridPoint,
    turn_penalty: float,
    step_cost: float = 1.0,
) -> Tuple[Optional[List[GridPoint]], float]:
    """
    在4邻域上进行A*:
    - 禁止对角线;
    - 通过状态(r,c,dir)实现转弯惩罚。
    """
    if start == end:
        return [start], 0.0

    rows, cols = grid.shape
    sr, sc = start
    er, ec = end
    if sr < 0 or sc < 0 or sr >= rows or sc >= cols:
        return None, math.inf
    if er < 0 or ec < 0 or er >= rows or ec >= cols:
        return None, math.inf

    dirs: List[GridPoint] = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def h(r: int, c: int) -> float:
        return float(abs(r - er) + abs(c - ec))

    # 堆元素: (f, g, r, c, dir_idx)
    heap: List[Tuple[float, float, int, int, int]] = []
    start_dir = -1
    heapq.heappush(heap, (h(sr, sc), 0.0, sr, sc, start_dir))

    best_g: Dict[Tuple[int, int, int], float] = {(sr, sc, start_dir): 0.0}
    parent: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {(sr, sc, start_dir): None}

    end_state: Optional[Tuple[int, int, int]] = None

    while heap:
        _, g, r, c, d_prev = heapq.heappop(heap)
        state = (r, c, d_prev)
        if g > best_g.get(state, math.inf) + 1e-9:
            continue
        if (r, c) == end:
            end_state = state
            break

        for d_idx, (dr, dc) in enumerate(dirs):
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not _is_passable_cell(grid, nxt, start, end):
                continue

            turn_cost = 0.0 if d_prev in (-1, d_idx) else float(turn_penalty)
            ng = g + float(step_cost) + turn_cost
            nstate = (nr, nc, d_idx)
            if ng + 1e-9 >= best_g.get(nstate, math.inf):
                continue
            best_g[nstate] = ng
            parent[nstate] = state
            heapq.heappush(heap, (ng + h(nr, nc), ng, nr, nc, d_idx))

    if end_state is None:
        return None, math.inf

    path_rev: List[GridPoint] = []
    cur: Optional[Tuple[int, int, int]] = end_state
    while cur is not None:
        path_rev.append((cur[0], cur[1]))
        cur = parent.get(cur)
    path = list(reversed(path_rev))
    return path, float(best_g[end_state])


def _build_edge_candidates(
    grid: np.ndarray,
    nodes: List[GridPoint],
    turn_penalty: float,
) -> List[_EdgeCandidate]:
    edges: List[_EdgeCandidate] = []
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            path, cost = _astar_route(grid, nodes[i], nodes[j], turn_penalty=turn_penalty)
            if path is None:
                continue
            edges.append(_EdgeCandidate(cost=cost, i=i, j=j, path=path))
    edges.sort(key=lambda x: x.cost)
    return edges


def _build_mst(edges: List[_EdgeCandidate], n_nodes: int) -> List[_EdgeCandidate]:
    uf = _UnionFind(n_nodes)
    picked: List[_EdgeCandidate] = []
    for e in edges:
        if uf.union(e.i, e.j):
            picked.append(e)
            if len(picked) == n_nodes - 1:
                break
    return picked


def _orient_edges_from_switch(
    mst_edges: List[_EdgeCandidate],
    nodes: List[GridPoint],
    switch_idx: int = 0,
) -> List[Tuple[int, int, List[GridPoint], float]]:
    """
    将无向MST边定向为“从开关出发”的树结构。
    返回: [(parent_idx, child_idx, path, cost), ...]
    """
    adj: Dict[int, List[Tuple[int, _EdgeCandidate]]] = {i: [] for i in range(len(nodes))}
    for e in mst_edges:
        adj[e.i].append((e.j, e))
        adj[e.j].append((e.i, e))

    visited = {switch_idx}
    q = deque([switch_idx])
    oriented: List[Tuple[int, int, List[GridPoint], float]] = []

    while q:
        cur = q.popleft()
        for nxt, e in adj.get(cur, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            q.append(nxt)
            if cur == e.i and nxt == e.j:
                path = e.path
            elif cur == e.j and nxt == e.i:
                path = list(reversed(e.path))
            else:
                # 理论不会出现，兜底用原始方向
                path = e.path
            oriented.append((cur, nxt, path, e.cost))
    return oriented


def _merge_unique_step_segments(routes: List[List[GridPoint]]) -> List[List[GridPoint]]:
    seg_set = set()
    merged: List[List[GridPoint]] = []
    for path in routes:
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            key = (a, b) if a <= b else (b, a)
            if key in seg_set:
                continue
            seg_set.add(key)
            merged.append([list(a), list(b)])
    return merged


def _grid_path_to_pixel_path(
    path: List[GridPoint],
    bbox_pixel: List[int],
    cell_size_px: int,
) -> List[List[float]]:
    min_x, min_y, max_x, max_y = [int(v) for v in bbox_pixel]
    room_w = max_x - min_x + 1
    room_h = max_y - min_y + 1

    pix: List[List[float]] = []
    for r, c in path:
        px, py = _grid_cell_to_pixel(
            row=int(r),
            col=int(c),
            min_x=min_x,
            min_y=min_y,
            room_w=room_w,
            room_h=room_h,
            cell_size_px=cell_size_px,
        )
        pix.append([float(px), float(py)])
    return pix


def _pixel_path_to_cad_path(
    pixel_path: List[List[float]],
    cad_params: Dict[str, float],
    image_w: int,
    image_h: int,
) -> List[List[float]]:
    cad_path: List[List[float]] = []
    for px, py in pixel_path:
        x_cad, y_cad = pixel_to_cad(
            px=px,
            py=py,
            Xmin=cad_params["Xmin"],
            Ymin=cad_params["Ymin"],
            Xmax=cad_params["Xmax"],
            Ymax=cad_params["Ymax"],
            width=image_w,
            height=image_h,
        )
        cad_path.append([float(x_cad), float(y_cad)])
    return cad_path


def _draw_wiring_legend(vis: np.ndarray) -> None:
    _draw_legend(vis)
    legend_x = 20
    legend_y = 20 + 130
    color_wire = (0, 255, 255)  # 黄色(BGR)
    cv2.rectangle(vis, (legend_x, legend_y), (legend_x + 18, legend_y + 18), color_wire, -1)
    cv2.putText(
        vis,
        "Wire route",
        (legend_x + 24, legend_y + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _save_wiring_visualization(
    image_path: str,
    room_wiring_dump: Dict[str, Dict[str, Any]],
    output_dir: str,
    alpha: float = 0.35,
) -> Optional[str]:
    img = cv2.imread(image_path)
    if img is None:
        return None

    alpha = max(0.05, min(0.95, float(alpha)))
    vis_dir = os.path.join(output_dir, "step8_room_wiring_visualization")
    os.makedirs(vis_dir, exist_ok=True)

    color_wire = (0, 255, 255)  # 黄色(BGR)
    thickness = max(1, int(os.getenv("CAD_WIRING_LINE_THICKNESS", "2")))

    for room_name, room_info in room_wiring_dump.items():
        bbox = room_info.get("bbox_pixel", [])
        matrix = np.array(room_info.get("matrix", []), dtype=np.int32)
        if len(bbox) != 4 or matrix.size == 0:
            continue
        min_x, min_y, max_x, max_y = [int(v) for v in bbox]
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img.shape[1] - 1, max_x)
        max_y = min(img.shape[0] - 1, max_y)
        if max_x <= min_x or max_y <= min_y:
            continue

        crop = img[min_y:max_y + 1, min_x:max_x + 1].copy()
        overlay = crop.copy()

        vis_room_info = {
            "matrix": room_info.get("matrix", []),
            "bbox_pixel": bbox,
            "cell_size_px": int(room_info.get("cell_size_px", 40)),
            "lamp_grid_positions": room_info.get("lamp_grid_positions", []),
            "switch_grid_positions": room_info.get("switch_grid_positions", []),
            "lamp_type": room_info.get("lamp_type", ""),
        }
        _draw_room_grid_on_overlay(
            overlay=overlay,
            room_name=room_name,
            room_info=vis_room_info,
            offset_x=min_x,
            offset_y=min_y,
            draw_label=True,
        )

        routes = room_info.get("route_paths_grid", []) or []
        for path in routes:
            if not isinstance(path, list) or len(path) < 2:
                continue
            points: List[Tuple[int, int]] = []
            for item in path:
                gp = _normalize_grid_point(item)
                if gp is None:
                    continue
                px, py = _grid_cell_to_pixel(
                    row=gp[0],
                    col=gp[1],
                    min_x=min_x,
                    min_y=min_y,
                    room_w=max_x - min_x + 1,
                    room_h=max_y - min_y + 1,
                    cell_size_px=int(room_info.get("cell_size_px", 40)),
                )
                points.append((int(round(px - min_x)), int(round(py - min_y))))
            if len(points) >= 2:
                cv2.polylines(
                    overlay,
                    [np.array(points, dtype=np.int32)],
                    isClosed=False,
                    color=color_wire,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )

        vis = cv2.addWeighted(overlay, alpha, crop, 1.0 - alpha, 0.0)
        _draw_wiring_legend(vis)

        lamp_type = str(room_info.get("lamp_type", "")).strip()
        file_base = f"{room_name}-{lamp_type}" if lamp_type else room_name
        safe_name = _sanitize_filename(file_base)
        output_path = os.path.join(vis_dir, f"{safe_name}.png")
        cv2.imwrite(output_path, vis)

    return vis_dir


def process_room_wiring_layout(
    lighting_payload: Dict[str, Any],
    image_path: str,
    cad_params: Dict[str, float],
    save_to_file: bool = True,
) -> Dict[str, Any]:
    """
    步骤8:
    - 基于步骤7输出生成线路布局:
      1) MST决定连接拓扑(无闭环)
      2) A*在离散矩阵中做横平竖直布线(含转弯惩罚)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    image_h, image_w = img.shape[:2]
    turn_penalty = float(os.getenv("CAD_WIRING_TURN_PENALTY", "0.8"))

    rooms_in = (
        lighting_payload.get("rooms_internal")
        or lighting_payload.get("rooms")
        or {}
    )
    rooms_out: Dict[str, Dict[str, Any]] = {}
    vis_dump: Dict[str, Dict[str, Any]] = {}

    for room_name, room_info in rooms_in.items():
        matrix_raw = room_info.get("matrix", [])
        matrix = np.array(matrix_raw, dtype=np.int32)
        if matrix.size == 0 or matrix.ndim != 2:
            continue

        bbox = room_info.get("bbox_pixel", [])
        if len(bbox) != 4:
            continue
        cell_size_px = int(room_info.get("cell_size_px", 40))

        lamp_positions_raw = ((room_info.get("lamps", {}) or {}).get("grid_positions", [])) or []
        lamp_points: List[GridPoint] = []
        for p in lamp_positions_raw:
            gp = _normalize_grid_point(p)
            if gp is None:
                continue
            lamp_points.append(gp)

        switch_positions_raw = []
        switches = room_info.get("switches", []) or []
        if isinstance(switches, list):
            for sw in switches:
                if not isinstance(sw, dict):
                    continue
                gp = _normalize_grid_point(sw.get("grid_position"))
                if gp is not None:
                    switch_positions_raw.append(gp)
        if not switch_positions_raw:
            sw_single = room_info.get("switch", {})
            if isinstance(sw_single, dict):
                gp = _normalize_grid_point(sw_single.get("grid_position"))
                if gp is not None:
                    switch_positions_raw.append(gp)

        if not switch_positions_raw or not lamp_points:
            rooms_out[room_name] = {
                "room_name": room_name,
                "status": "skipped",
                "reason": "missing_switch_or_lamps",
                "route_count": 0,
                "routes": [],
            }
            continue

        switch_point = switch_positions_raw[0]
        nodes: List[GridPoint] = [switch_point] + lamp_points
        node_labels = ["switch"] + [f"lamp_{i+1}" for i in range(len(lamp_points))]

        edge_candidates = _build_edge_candidates(
            grid=matrix,
            nodes=nodes,
            turn_penalty=turn_penalty,
        )
        mst_edges = _build_mst(edge_candidates, len(nodes))
        directed = _orient_edges_from_switch(mst_edges, nodes, switch_idx=0)

        routes: List[Dict[str, Any]] = []
        route_paths_grid: List[List[GridPoint]] = []
        total_cost = 0.0
        for parent_idx, child_idx, path_grid, cost in directed:
            route_paths_grid.append(path_grid)
            pixel_path = _grid_path_to_pixel_path(path_grid, bbox, cell_size_px)
            cad_path = _pixel_path_to_cad_path(pixel_path, cad_params, image_w, image_h)
            total_cost += float(cost)
            routes.append(
                {
                    "from_node": node_labels[parent_idx],
                    "to_node": node_labels[child_idx],
                    "from_grid": [int(nodes[parent_idx][0]), int(nodes[parent_idx][1])],
                    "to_grid": [int(nodes[child_idx][0]), int(nodes[child_idx][1])],
                    "path_grid": [[int(r), int(c)] for r, c in path_grid],
                    "path_pixel": pixel_path,
                    "path_cad": cad_path,
                    "cost": float(cost),
                }
            )

        unreachable_nodes = []
        reachable = {0}
        for _, child_idx, _, _ in directed:
            reachable.add(child_idx)
        for idx in range(1, len(nodes)):
            if idx not in reachable:
                unreachable_nodes.append(node_labels[idx])

        merged_segments_grid = _merge_unique_step_segments(route_paths_grid)
        merged_segments_pixel = []
        merged_segments_cad = []
        for seg in merged_segments_grid:
            p0, p1 = (tuple(seg[0]), tuple(seg[1]))
            pix = _grid_path_to_pixel_path([p0, p1], bbox, cell_size_px)
            cad = _pixel_path_to_cad_path(pix, cad_params, image_w, image_h)
            merged_segments_pixel.append(pix)
            merged_segments_cad.append(cad)

        status = "ok"
        if len(directed) < max(0, len(nodes) - 1):
            status = "partial"

        room_out = {
            "room_name": room_name,
            "status": status,
            "turn_penalty": float(turn_penalty),
            "node_count": len(nodes),
            "route_count": len(routes),
            "switch_grid_position": [int(switch_point[0]), int(switch_point[1])],
            "lamp_grid_positions": [[int(r), int(c)] for r, c in lamp_points],
            "total_route_cost": float(total_cost),
            "unreachable_nodes": unreachable_nodes,
            "routes": routes,
            "merged_segments_grid": merged_segments_grid,
            "merged_segments_pixel": merged_segments_pixel,
            "merged_segments_cad": merged_segments_cad,
        }
        rooms_out[room_name] = room_out

        vis_dump[room_name] = {
            "room_name": room_name,
            "bbox_pixel": bbox,
            "cell_size_px": cell_size_px,
            "matrix": matrix.tolist(),
            "lamp_type": ((room_info.get("lamps", {}) or {}).get("lamp_type", "")),
            "lamp_grid_positions": [[int(r), int(c)] for r, c in lamp_points],
            "switch_grid_positions": [[int(switch_point[0]), int(switch_point[1])]],
            "route_paths_grid": [[[int(r), int(c)] for r, c in p] for p in route_paths_grid],
        }

        print(
            f"[step8] room='{room_name}' status={status} routes={len(routes)} "
            f"unreachable={len(unreachable_nodes)}"
        )

    payload = {
        "image_width": int(image_w),
        "image_height": int(image_h),
        "rooms": rooms_out,
    }

    if save_to_file:
        output_dir = os.getenv("CAD_STEP_OUTPUT_DIR", "images/output")
        os.makedirs(output_dir, exist_ok=True)

        result_file = os.path.join(output_dir, "step8_wiring_layout.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        vis_alpha = float(os.getenv("CAD_WIRING_VIS_ALPHA", "0.35"))
        vis_dir = _save_wiring_visualization(
            image_path=image_path,
            room_wiring_dump=vis_dump,
            output_dir=output_dir,
            alpha=vis_alpha,
        )
        print(f"步骤8线路结果已保存: {result_file}")
        if vis_dir:
            print(f"步骤8线路可视化已保存: {vis_dir}")

    return payload
