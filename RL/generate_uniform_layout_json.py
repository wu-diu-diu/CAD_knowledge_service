from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _choose_grid_shape(lamp_count: int, rows: int, cols: int) -> tuple[int, int]:
    """Choose a row/col lamp lattice that best matches the room aspect ratio."""
    aspect = cols / max(rows, 1)
    candidates: list[tuple[float, int, int]] = []
    for r in range(1, lamp_count + 1):
        if lamp_count % r != 0:
            continue
        c = lamp_count // r
        lattice_aspect = c / r
        candidates.append((abs(lattice_aspect - aspect), r, c))

    if not candidates:
        return 1, lamp_count

    candidates.sort(key=lambda item: (item[0], abs((item[2] / item[1]) - 1.0)))
    best_diff = candidates[0][0]
    tied = [(r, c) for diff, r, c in candidates if abs(diff - best_diff) < 1e-9]
    if len(tied) == 1:
        return tied[0]

    if cols >= rows:
        tied.sort(key=lambda rc: (-rc[1], rc[0]))  # prefer more columns when room is wider
    else:
        tied.sort(key=lambda rc: (-rc[0], rc[1]))  # prefer more rows when room is taller
    return tied[0]


def _nearest_legal_cell(matrix: np.ndarray, target_r: float, target_c: float, used: set[tuple[int, int]]) -> tuple[int, int]:
    """Pick the unused placeable cell closest to the target guide point."""
    legal = np.argwhere(matrix == 1)
    if legal.size == 0:
        raise ValueError("No legal lamp positions in room matrix.")

    best_pos: tuple[int, int] | None = None
    best_score: tuple[float, int, int] | None = None
    for r, c in legal.tolist():
        pos = (int(r), int(c))
        if pos in used:
            continue
        dist = (r - target_r) ** 2 + (c - target_c) ** 2
        score = (float(dist), int(r), int(c))
        if best_score is None or score < best_score:
            best_score = score
            best_pos = pos

    if best_pos is None:
        raise ValueError("Unable to find enough unique legal lamp positions.")
    return best_pos


def compute_uniform_positions(matrix: np.ndarray, lamp_count: int) -> list[tuple[int, int]]:
    """Compute uniform lamp positions for a rectangular room matrix."""
    rows, cols = matrix.shape
    grid_r, grid_c = _choose_grid_shape(lamp_count, rows, cols)

    row_guides = np.linspace(0, rows - 1, grid_r + 2)[1:-1]
    col_guides = np.linspace(0, cols - 1, grid_c + 2)[1:-1]

    positions: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for r in row_guides:
        for c in col_guides:
            pos = _nearest_legal_cell(matrix, float(r), float(c), used)
            positions.append(pos)
            used.add(pos)

    if len(positions) != lamp_count:
        raise ValueError(f"Expected {lamp_count} lamp positions, got {len(positions)}.")
    return positions


def apply_uniform_layout(room_payload: dict) -> dict:
    """Return a copy of one room JSON payload with uniform lamp placement encoded as 4."""
    top_key = next(iter(room_payload))
    room = dict(room_payload[top_key])
    matrix = np.array(room["matrix"], dtype=np.int32)
    lamp_count = int(room["lamp_count"])

    positions = compute_uniform_positions(matrix, lamp_count)
    layout = matrix.copy()
    for r, c in positions:
        if layout[r, c] != 1:
            raise ValueError(f"Illegal lamp placement at {(r, c)} with value={layout[r, c]}.")
        layout[r, c] = 4

    room["matrix"] = layout.tolist()
    return {top_key: room}


def generate_uniform_layout_dataset(
    input_dir: str | Path = "RL/room_gen/regular/json",
    output_dir: str | Path = "RL/test_room/layout_room/json",
) -> None:
    """Read regular-room JSON files, place lamps uniformly, and write new JSON files."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No room JSON files found in {input_path}")

    saved = 0
    for src in files:
        payload = json.loads(src.read_text(encoding="utf-8"))
        laid_out = apply_uniform_layout(payload)
        with (output_path / src.name).open("w", encoding="utf-8") as f:
            json.dump(laid_out, f, ensure_ascii=False, indent=2)
        saved += 1

    print(f"[uniform_layout] saved={saved} input={input_path} output={output_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate uniform lamp-layout JSON files for regular rooms")
    parser.add_argument("--input", type=str, default="RL/room_gen/regular/json")
    parser.add_argument("--output", type=str, default="RL/test_room/layout_room/json")
    args = parser.parse_args()

    generate_uniform_layout_dataset(input_dir=args.input, output_dir=args.output)


if __name__ == "__main__":
    main()
