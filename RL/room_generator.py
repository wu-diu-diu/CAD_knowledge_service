"""
Room generator for RL lamp-placement training.

Generates irregular rooms (L, U/concave, T/convex, hollow, cross, multi_cut)
within a 48×48 grid, with lamp counts derived from placeable area (~100 cells/lamp).
Each room is saved as a standalone JSON file compatible with SingleRoomLightingEnv.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────
MAX_SIZE = 48
AREA_PER_LAMP = 100
LAMP_MIN = 2
LAMP_MAX = 9

# Outer bbox row/col range per target lamp count
# Sized so that after carving, placeable ≈ lamp * AREA_PER_LAMP
_LAMP_BBOX: dict[int, tuple[int, int]] = {
    2: (14, 18),
    3: (17, 21),
    4: (20, 24),
    5: (23, 27),
    6: (26, 30),
    7: (29, 33),
    8: (32, 37),
    9: (36, 45),
}

_TEMPLATE_FIELDS = {
    "room_name": "房间",
    "cell_size_px": 40,
    "illuminance": 300,
    "lamp_type": "荧光灯",
    "lamp": {
        "model": "PHILIPS TL5 HE 35w/827",
        "lamp_power_w": 35,
        "lamp_luminous_flux_lm": 3325,
        "uf": 0.6,
        "mf": 0.8,
        "tube_count": 2,
        "total_luminous_flux_lm": 6650,
    },
}

SHAPES = ["L", "U", "T", "hollow", "cross", "multi_cut"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _lamp_count(matrix: np.ndarray) -> int:
    placeable = int(np.sum(matrix == 1))
    return int(np.clip(round(placeable / AREA_PER_LAMP), LAMP_MIN, LAMP_MAX))


def _collect_edge_runs(matrix: np.ndarray) -> list[tuple[str, int, list[int]]]:
    """
    Collect runs of 4+ consecutive value-1 cells on the four outer edges.

    Returns list of (edge, fixed_coord, [varying_coords]) where:
      edge = 'top' | 'bottom' | 'left' | 'right'
      fixed_coord = the row (top/bottom) or col (left/right) index
      varying_coords = list of col (top/bottom) or row (left/right) indices
    """
    rows, cols = matrix.shape
    results = []

    def _runs_in_seq(seq: list[int], fixed: int, edge: str) -> None:
        run: list[int] = []
        for v in seq:
            if matrix[fixed, v] == 1 if edge in ('top', 'bottom') else matrix[v, fixed] == 1:
                run.append(v)
            else:
                if len(run) >= 4:
                    results.append((edge, fixed, run.copy()))
                run = []
        if len(run) >= 4:
            results.append((edge, fixed, run.copy()))

    _runs_in_seq(list(range(cols)), 0,        'top')
    _runs_in_seq(list(range(cols)), rows - 1, 'bottom')
    _runs_in_seq(list(range(rows)), 0,        'left')
    _runs_in_seq(list(range(rows)), cols - 1, 'right')

    return results


def _place_door_switch(matrix: np.ndarray, rng: random.Random) -> bool:
    """
    Place a 4-cell door (value=2) on one of the four outer edges and one
    switch (value=3) on the interior side, same row/col as the door middle.

    Door placement rules:
      - Must be on row 0, last row, col 0, or last col (true outer boundary)
      - 4 consecutive value-1 cells along that edge

    Switch placement rules (strict):
      - If door is horizontal (top/bottom edge), switch is on the same row,
        placed at one of door's left/right sides.
        * If door center is left-biased in 48x48 (center_col < 24), choose right side.
        * Otherwise choose left side.
      - If door is vertical (left/right edge), switch is on the same column,
        placed at one of door's up/down sides.
        * If door center is upper-biased in 48x48 (center_row < 24), choose lower side.
        * Otherwise choose upper side.
    """
    rows, cols = matrix.shape
    runs = _collect_edge_runs(matrix)
    if not runs:
        return False

    edge, fixed, varying = rng.choice(runs)
    max_start = len(varying) - 4
    start = rng.randint(0, max_start)
    door_varying = varying[start:start + 4]

    # place door cells
    for v in door_varying:
        if edge in ('top', 'bottom'):
            matrix[fixed, v] = 2
        else:
            matrix[v, fixed] = 2

    switch_pos = _choose_switch_by_side_rule(matrix, edge=edge, fixed=fixed, door_varying=door_varying)
    if switch_pos is None:
        return False

    matrix[switch_pos] = 3
    return _is_switch_rule_compliant(matrix)


def _is_switch_rule_compliant(matrix: np.ndarray) -> bool:
    """
    Validate switch-door relation under the strict side rule:
      - exactly one switch cell (value=3)
      - at least one door cell (value=2)
      - horizontal door: switch shares the same row, is outside door span, and
        side is selected by 48x48 left/right bias
      - vertical door: switch shares the same column, is outside door span, and
        side is selected by 48x48 upper/lower bias
    """
    door_cells = np.argwhere(matrix == 2)
    switch_cells = np.argwhere(matrix == 3)
    if len(door_cells) == 0 or len(switch_cells) != 1:
        return False

    sr, sc = map(int, switch_cells[0])
    door_rows = sorted({int(dr) for dr, _ in door_cells.tolist()})
    door_cols = sorted({int(dc) for _, dc in door_cells.tolist()})

    # Horizontal door: same row, side based on left/right bias in 48x48.
    if len(door_rows) == 1 and len(door_cols) >= 2:
        row = door_rows[0]
        c_min, c_max = min(door_cols), max(door_cols)
        if sr != row or (c_min <= sc <= c_max):
            return False
        center_col = 0.5 * (c_min + c_max)
        prefer_right = center_col < (MAX_SIZE / 2.0)
        if prefer_right:
            return sc > c_max
        return sc < c_min

    # Vertical door: same column, side based on upper/lower bias in 48x48.
    if len(door_cols) == 1 and len(door_rows) >= 2:
        col = door_cols[0]
        r_min, r_max = min(door_rows), max(door_rows)
        if sc != col or (r_min <= sr <= r_max):
            return False
        center_row = 0.5 * (r_min + r_max)
        prefer_down = center_row < (MAX_SIZE / 2.0)
        if prefer_down:
            return sr > r_max
        return sr < r_min

    return False


def _choose_switch_by_side_rule(
    matrix: np.ndarray,
    *,
    edge: str,
    fixed: int,
    door_varying: list[int],
) -> tuple[int, int] | None:
    """
    Choose switch position by the requested side rule.

    Returns None if the preferred side has no valid value-1 cell.
    """
    rows, cols = matrix.shape

    if edge in ("top", "bottom"):
        row = fixed
        c_min, c_max = min(door_varying), max(door_varying)
        center_col = 0.5 * (c_min + c_max)
        prefer_right = center_col < (MAX_SIZE / 2.0)

        if prefer_right:
            candidates = [(row, c) for c in range(c_max + 1, cols)]
        else:
            candidates = [(row, c) for c in range(c_min - 1, -1, -1)]

    else:
        col = fixed
        r_min, r_max = min(door_varying), max(door_varying)
        center_row = 0.5 * (r_min + r_max)
        prefer_down = center_row < (MAX_SIZE / 2.0)

        if prefer_down:
            candidates = [(r, col) for r in range(r_max + 1, rows)]
        else:
            candidates = [(r, col) for r in range(r_min - 1, -1, -1)]

    for r, c in candidates:
        if matrix[r, c] == 1:
            return int(r), int(c)
    return None


def _rand(lo: int, hi: int, rng: random.Random) -> int:
    return rng.randint(lo, max(lo, hi))


# ── shape generators ──────────────────────────────────────────────────────────

def _gen_L(bbox_lo: int, bbox_hi: int, rng: random.Random) -> np.ndarray:
    R = min(_rand(bbox_lo, bbox_hi, rng), MAX_SIZE)
    C = min(_rand(bbox_lo, bbox_hi, rng), MAX_SIZE)
    m = np.ones((R, C), dtype=np.int32)
    cr = _rand(R // 3, R * 2 // 3, rng)
    cc = _rand(C // 3, C * 2 // 3, rng)
    corner = rng.randint(0, 3)
    if corner == 0:
        m[:cr, :cc] = 0
    elif corner == 1:
        m[:cr, C - cc:] = 0
    elif corner == 2:
        m[R - cr:, :cc] = 0
    else:
        m[R - cr:, C - cc:] = 0
    return m


def _gen_U(bbox_lo: int, bbox_hi: int, rng: random.Random) -> np.ndarray:
    R = min(_rand(bbox_lo, bbox_hi, rng), MAX_SIZE)
    C = min(_rand(bbox_lo, bbox_hi, rng), MAX_SIZE)
    m = np.ones((R, C), dtype=np.int32)
    side = rng.randint(0, 3)
    if side == 0:
        depth = _rand(R // 4, R // 2, rng)
        w = _rand(C // 4, C // 2, rng)
        c0 = _rand(1, max(1, C - w - 1), rng)
        m[:depth, c0:c0 + w] = 0
    elif side == 1:
        depth = _rand(R // 4, R // 2, rng)
        w = _rand(C // 4, C // 2, rng)
        c0 = _rand(1, max(1, C - w - 1), rng)
        m[R - depth:, c0:c0 + w] = 0
    elif side == 2:
        depth = _rand(C // 4, C // 2, rng)
        h = _rand(R // 4, R // 2, rng)
        r0 = _rand(1, max(1, R - h - 1), rng)
        m[r0:r0 + h, :depth] = 0
    else:
        depth = _rand(C // 4, C // 2, rng)
        h = _rand(R // 4, R // 2, rng)
        r0 = _rand(1, max(1, R - h - 1), rng)
        m[r0:r0 + h, C - depth:] = 0
    return m


def _gen_T(bbox_lo: int, bbox_hi: int, rng: random.Random) -> np.ndarray:
    R = min(_rand(bbox_lo, bbox_hi, rng), MAX_SIZE - 6)
    C = min(_rand(bbox_lo, bbox_hi, rng), MAX_SIZE - 6)
    ph = _rand(max(3, R // 5), max(4, R // 3), rng)
    pw = _rand(max(3, C // 5), max(4, C // 3), rng)
    side = rng.randint(0, 3)
    if side == 0:
        total_R, total_C = min(R + ph, MAX_SIZE), C
        m = np.zeros((total_R, total_C), dtype=np.int32)
        m[ph:, :] = 1
        c0 = _rand(1, max(1, total_C - pw - 1), rng)
        m[:ph, c0:c0 + pw] = 1
    elif side == 1:
        total_R, total_C = min(R + ph, MAX_SIZE), C
        m = np.zeros((total_R, total_C), dtype=np.int32)
        m[:R, :] = 1
        c0 = _rand(1, max(1, total_C - pw - 1), rng)
        m[R:, c0:c0 + pw] = 1
    elif side == 2:
        total_R, total_C = R, min(C + ph, MAX_SIZE)
        m = np.zeros((total_R, total_C), dtype=np.int32)
        m[:, ph:] = 1
        r0 = _rand(1, max(1, total_R - pw - 1), rng)
        m[r0:r0 + pw, :ph] = 1
    else:
        total_R, total_C = R, min(C + ph, MAX_SIZE)
        m = np.zeros((total_R, total_C), dtype=np.int32)
        m[:, :C] = 1
        r0 = _rand(1, max(1, total_R - pw - 1), rng)
        m[r0:r0 + pw, C:] = 1
    return m


def _gen_hollow(bbox_lo: int, bbox_hi: int, rng: random.Random) -> np.ndarray:
    R = min(_rand(max(bbox_lo, 12), bbox_hi, rng), MAX_SIZE)
    C = min(_rand(max(bbox_lo, 12), bbox_hi, rng), MAX_SIZE)
    m = np.ones((R, C), dtype=np.int32)
    margin = 2
    hole_r = _rand(4, max(4, R - 2 * margin - 2), rng)
    hole_c = _rand(4, max(4, C - 2 * margin - 2), rng)
    r0 = _rand(margin, max(margin, R - margin - hole_r), rng)
    c0 = _rand(margin, max(margin, C - margin - hole_c), rng)
    m[r0:r0 + hole_r, c0:c0 + hole_c] = 0
    return m


def _gen_cross(bbox_lo: int, bbox_hi: int, rng: random.Random) -> np.ndarray:
    R = min(_rand(bbox_lo, bbox_hi, rng), MAX_SIZE)
    C = min(_rand(bbox_lo, bbox_hi, rng), MAX_SIZE)
    m = np.zeros((R, C), dtype=np.int32)
    bar_h = _rand(max(3, R // 4), max(4, R // 2), rng)
    r0 = (R - bar_h) // 2
    m[r0:r0 + bar_h, :] = 1
    bar_w = _rand(max(3, C // 4), max(4, C // 2), rng)
    c0 = (C - bar_w) // 2
    m[:, c0:c0 + bar_w] = 1
    return m


def _gen_multi_cut(bbox_lo: int, bbox_hi: int, rng: random.Random) -> np.ndarray:
    R = min(_rand(bbox_lo, bbox_hi, rng), MAX_SIZE)
    C = min(_rand(bbox_lo, bbox_hi, rng), MAX_SIZE)
    m = np.ones((R, C), dtype=np.int32)
    n_cuts = rng.randint(2, 3)
    corners = rng.sample(range(4), min(n_cuts, 4))
    for corner in corners:
        cr = _rand(R // 5, R // 3, rng)
        cc = _rand(C // 5, C // 3, rng)
        if corner == 0:
            m[:cr, :cc] = 0
        elif corner == 1:
            m[:cr, C - cc:] = 0
        elif corner == 2:
            m[R - cr:, :cc] = 0
        else:
            m[R - cr:, C - cc:] = 0
    return m


_SHAPE_FN = {
    "L":         _gen_L,
    "U":         _gen_U,
    "T":         _gen_T,
    "hollow":    _gen_hollow,
    "cross":     _gen_cross,
    "multi_cut": _gen_multi_cut,
}


# ── core generator ────────────────────────────────────────────────────────────

def generate_room(
    target_lamps: int,
    shape: str,
    rng: random.Random,
    max_retries: int = 10,
) -> np.ndarray | None:
    bbox_lo, bbox_hi = _LAMP_BBOX[target_lamps]
    fn = _SHAPE_FN[shape]
    for _ in range(max_retries):
        m = fn(bbox_lo, bbox_hi, rng)
        if m.shape[0] > MAX_SIZE or m.shape[1] > MAX_SIZE:
            continue
        if int(np.sum(m == 1)) < LAMP_MIN * AREA_PER_LAMP:
            continue
        m_copy = m.copy()
        if not _place_door_switch(m_copy, rng):
            continue
        if not _is_switch_rule_compliant(m_copy):
            continue
        if _lamp_count(m_copy) != target_lamps:
            continue
        return m_copy
    return None


def _to_json(matrix: np.ndarray) -> dict:
    room = {
        **_TEMPLATE_FIELDS,
        "grid_rows": int(matrix.shape[0]),
        "grid_cols": int(matrix.shape[1]),
        "lamp_count": _lamp_count(matrix),
        "matrix": matrix.tolist(),
    }
    return {"房间": room}


# ── dataset generation ────────────────────────────────────────────────────────

def generate_dataset(count: int, output_dir: str | Path, seed: int = 42) -> None:
    """Generate `count` rooms with balanced lamp-count and shape distributions."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    np.random.seed(seed)

    lamp_counts = list(range(LAMP_MIN, LAMP_MAX + 1))  # 2..9
    per_lamp = max(1, count // len(lamp_counts))

    # Build balanced task list: round-robin over shapes within each lamp count
    tasks: list[tuple[int, str]] = []
    for lamps in lamp_counts:
        for i in range(per_lamp):
            tasks.append((lamps, SHAPES[i % len(SHAPES)]))
    while len(tasks) < count:
        tasks.append((rng.choice(lamp_counts), rng.choice(SHAPES)))
    rng.shuffle(tasks)
    tasks = tasks[:count]

    name_counter: dict[tuple[int, str], int] = {}
    saved = 0
    fallback_used = 0

    for target_lamps, shape in tasks:
        matrix = generate_room(target_lamps, shape, rng)
        if matrix is None:
            fallback_used += 1
            for fb_shape in rng.sample(SHAPES, len(SHAPES)):
                matrix = generate_room(target_lamps, fb_shape, rng)
                if matrix is not None:
                    shape = fb_shape
                    break
        if matrix is None:
            continue

        key = (target_lamps, shape)
        idx = name_counter.get(key, 0) + 1
        name_counter[key] = idx
        filename = f"shape_{shape}_{target_lamps}lamp_{idx:04d}.json"
        with (output_path / filename).open("w", encoding="utf-8") as f:
            json.dump(_to_json(matrix), f, ensure_ascii=False, indent=2)
        saved += 1

    print(f"[room_gen] saved={saved}  fallback_used={fallback_used}  dir={output_path}")

    lamp_dist: Counter = Counter()
    shape_dist: Counter = Counter()
    for (lamps, shape), cnt in name_counter.items():
        lamp_dist[lamps] += cnt
        shape_dist[shape] += cnt
    print("[room_gen] lamp dist :", dict(sorted(lamp_dist.items())))
    print("[room_gen] shape dist:", dict(sorted(shape_dist.items())))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate irregular rooms for RL training")
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--output", type=str, default="RL/room_gen/json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(count=args.count, output_dir=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
