"""
python RL/visualize.py  --json_dir RL/room_gen/regular/json/  --output_dir RL/room_gen/regular/image
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# OpenCV uses BGR instead of RGB.
CELL_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 255),       # red: invalid placement area
    1: (0, 180, 0),       # green: valid placement area
    2: (255, 0, 0),       # blue: door
    3: (0, 255, 255),     # yellow: switch
    4: (0, 0, 0),         # black: lamp placement
}

GRID_LINE_COLOR = (190, 190, 190)
TEXT_COLOR = (30, 30, 30)
LEGEND_TEXT_COLOR = (10, 10, 10)
DEFAULT_CELL_SIZE = 36
LEGEND_HEIGHT = 72
PADDING = 12
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


def render_room_grid(
    matrix: list[list[int]] | np.ndarray,
    *,
    cell_size: int = DEFAULT_CELL_SIZE,
    room_name: str | None = None,
) -> np.ndarray:
    """
    Render a discrete room matrix to a grid image.

    Cell semantics:
        0 -> invalid / wall area (red)
        1 -> placeable area (green)
        2 -> door (blue)
        3 -> switch (yellow)
        4 -> lamp (black)

    Returns:
        OpenCV BGR image as a NumPy array.
    """
    grid = np.asarray(matrix, dtype=np.int32)
    if grid.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={grid.shape}")

    rows, cols = grid.shape
    image_height = rows * cell_size + LEGEND_HEIGHT + PADDING * 2
    image_width = cols * cell_size + PADDING * 2
    image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)

    y0 = PADDING
    x0 = PADDING

    for row in range(rows):
        for col in range(cols):
            value = int(grid[row, col])
            color = CELL_COLORS.get(value, (128, 128, 128))
            left = x0 + col * cell_size
            top = y0 + row * cell_size
            right = left + cell_size
            bottom = top + cell_size
            cv2.rectangle(image, (left, top), (right, bottom), color, thickness=-1)
            cv2.rectangle(image, (left, top), (right, bottom), GRID_LINE_COLOR, thickness=1)

    if room_name:
        image = _draw_text(
            image,
            room_name,
            (x0, y0 + rows * cell_size + 6),
            font_size=28,
            color=TEXT_COLOR,
        )

    image = _draw_legend(image, origin=(x0, y0 + rows * cell_size + 40))
    return image


def save_room_grid_image(
    matrix: list[list[int]] | np.ndarray,
    output_path: str | Path,
    *,
    cell_size: int = DEFAULT_CELL_SIZE,
    room_name: str | None = None,
) -> Path:
    """Render a room matrix and save it to disk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = render_room_grid(matrix, cell_size=cell_size, room_name=room_name)
    ok = cv2.imwrite(str(output), image)
    if not ok:
        raise RuntimeError(f"Failed to write image to {output}")
    return output


def load_room_payloads(json_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load room definitions from a JSON file."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Expected top-level object mapping room names to room payloads.")
    return payload


def export_rooms_from_json(
    json_path: str | Path,
    output_dir: str | Path,
    *,
    cell_size: int | None = None,
) -> list[Path]:
    """
    Export all rooms from the given JSON file to grid images.

    The JSON may contain one or more room payloads. Each payload must contain
    at least `matrix`, and may optionally include `room_name` and `cell_size_px`.
    """
    room_payloads = load_room_payloads(json_path)
    output_root = Path(output_dir)
    written: list[Path] = []

    for room_key, room_data in room_payloads.items():
        matrix = room_data["matrix"]
        room_name = room_data.get("room_name", room_key)
        render_cell_size = int(cell_size or room_data.get("cell_size_px") or DEFAULT_CELL_SIZE)
        safe_name = room_name.replace("/", "_").replace("\\", "_")
        output_path = output_root / f"{safe_name}.png"
        written.append(
            save_room_grid_image(
                matrix,
                output_path,
                cell_size=render_cell_size,
                room_name=room_name,
            )
        )

    return written


def _draw_legend(image: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
    """Draw a simple legend under the room grid."""
    labels = [
        (0, "0 不可布置"),
        (1, "1 可布置"),
        (2, "2 门"),
        (3, "3 开关"),
        (4, "4 灯具"),
    ]
    x, y = origin
    box_size = 18
    gap = 14
    cursor_x = x

    for value, label in labels:
        color = CELL_COLORS[value]
        cv2.rectangle(image, (cursor_x, y), (cursor_x + box_size, y + box_size), color, thickness=-1)
        cv2.rectangle(image, (cursor_x, y), (cursor_x + box_size, y + box_size), GRID_LINE_COLOR, thickness=1)
        image = _draw_text(
            image,
            label,
            (cursor_x + box_size + 6, y - 1),
            font_size=18,
            color=LEGEND_TEXT_COLOR,
        )
        cursor_x += box_size + 6 + len(label) * 10 + gap
    return image


def _draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    *,
    font_size: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    """
    Draw text with a Chinese-capable font via Pillow.

    OpenCV's built-in Hershey fonts do not support Chinese characters, so text
    rendering is delegated to Pillow and then converted back to OpenCV BGR.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)
    font = _load_font(font_size)
    draw.text(position, text, fill=(color[2], color[1], color[0]), font=font)
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load the first available Chinese font from the local system."""
    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, font_size)
            except OSError:
                continue
    return ImageFont.load_default()


def save_padded_room_image(
    original_matrix: np.ndarray,
    padded_size: int,
    output_path: str | Path,
    *,
    cell_size: int = DEFAULT_CELL_SIZE,
    room_name: str | None = None,
) -> Path:
    """
    Save a visualization of the room after padding to padded_size.

    Padding areas are rendered in red (same as invalid/wall areas).

    Args:
        original_matrix: The original room matrix (H x W)
        padded_size: Target padded size (e.g., 32)
        output_path: Where to save the image
        cell_size: Size of each grid cell in pixels
        room_name: Optional room name for the title

    Returns:
        Path to the saved image
    """
    original_matrix = np.asarray(original_matrix, dtype=np.int32)
    grid_rows, grid_cols = original_matrix.shape

    if grid_rows > padded_size or grid_cols > padded_size:
        raise ValueError(
            f"Room shape {original_matrix.shape} exceeds padded size {padded_size}."
        )

    # Calculate offsets (same logic as in env.py)
    row_offset = (padded_size - grid_rows) // 2
    col_offset = (padded_size - grid_cols) // 2

    # Create padded matrix filled with 0 (invalid/wall, will be red)
    padded_matrix = np.zeros((padded_size, padded_size), dtype=np.int32)

    # Copy original room into the center
    padded_matrix[
        row_offset:row_offset + grid_rows,
        col_offset:col_offset + grid_cols
    ] = original_matrix

    # Generate title
    title = f"{room_name or 'Room'} (Padded to {padded_size}x{padded_size})"

    # Render and save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = render_room_grid(padded_matrix, cell_size=cell_size, room_name=title)
    ok = cv2.imwrite(str(output), image)
    if not ok:
        raise RuntimeError(f"Failed to write image to {output}")
    return output

def plot_episode_step_breakdown(
    step_records: list[dict[str, float | int | bool]],
    output_path: Path,
    *,
    episode_idx: int,
) -> Path:
    """Plot per-step potential reward and terminal alignment/wiring diagnostics."""
    if not step_records:
        raise ValueError("Step records are empty. Cannot plot episode breakdown.")

    steps = [int(item["step"]) for item in step_records]

    potential_scores = [float(item["potential_reduction_score"]) for item in step_records]
    alignment_scores = [float(item["alignment_score"]) for item in step_records]
    wiring_scores = [float(item["wiring_score"]) for item in step_records]
    mst_costs = [float(item["mst_cost"]) for item in step_records]
    invalid_penalties = [float(item["invalid_penalty"]) for item in step_records]

    potential_term = [float(item["potential_term"]) for item in step_records]
    alignment_term = [float(item["alignment_term"]) for item in step_records]
    wiring_term = [float(item["wiring_term"]) for item in step_records]
    total_reward = [float(item["step_total"]) for item in step_records]
    terminal_bonus = [float(item["terminal_bonus"]) for item in step_records]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(steps, potential_scores, label="potential_reduction_score", linewidth=2.0)
    axes[0].plot(steps, alignment_scores, label="alignment_score", linewidth=2.0)
    axes[0].plot(steps, wiring_scores, label="wiring_score", linewidth=2.0)
    axes[0].plot(steps, invalid_penalties, label="invalid_penalty", linewidth=2.0, linestyle="--")
    axes[0].set_ylabel("Raw score / penalty")
    axes[0].set_title(f"Episode {episode_idx:04d} Step Scores")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0].legend(loc="best", ncol=2)

    axes[1].plot(steps, potential_term, label="potential_term", linewidth=2.0)
    axes[1].plot(steps, alignment_term, label="terminal_alignment_term", linewidth=2.0)
    axes[1].plot(steps, wiring_term, label="terminal_wiring_term", linewidth=2.0)
    axes[1].plot(steps, invalid_penalties, label="invalid_penalty", linewidth=2.0, linestyle="--")
    axes[1].plot(steps, total_reward, label="step_total", linewidth=2.2, color="black")
    axes[1].plot(steps, terminal_bonus, label="terminal_bonus", linewidth=2.0, linestyle=":")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Weighted contribution")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1].legend(loc="best", ncol=2)

    axes[2].plot(steps, mst_costs, label="mst_cost", linewidth=2.0, color="#6e3fb4")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Cost")
    axes[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[2].legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def main() -> None:
    import argparse

    repo_root = Path(__file__).resolve().parents[1]
    default_json_dir = repo_root / "RL" / "room_gen" / "json"
    default_output_dir = repo_root / "RL" / "room_gen" / "image"

    parser = argparse.ArgumentParser(description="Batch visualize room JSON files to PNG grids.")
    parser.add_argument("--json_dir", type=str, default=str(default_json_dir), help="Input directory containing JSON files.")
    parser.add_argument("--output_dir", type=str, default=str(default_output_dir), help="Output directory for PNG files.")
    parser.add_argument("--cell_size", type=int, default=DEFAULT_CELL_SIZE, help="Grid cell size in pixels.")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.rglob("*.json"))
    if not json_files:
        print(f"[visualize] No JSON files found in: {json_dir}")
        return

    success = 0
    failed: list[tuple[str, str]] = []

    for json_path in json_files:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))

            room_payload: dict[str, Any] | None = None
            if isinstance(payload, dict) and "matrix" in payload:
                room_payload = payload
            elif isinstance(payload, dict):
                for value in payload.values():
                    if isinstance(value, dict) and "matrix" in value:
                        room_payload = value
                        break

            if room_payload is None:
                raise ValueError("No room payload with 'matrix' found.")

            matrix = room_payload.get("matrix")
            if matrix is None:
                raise ValueError("Matrix is missing.")

            output_path = output_dir / f"{json_path.stem}.png"
            save_room_grid_image(
                matrix,
                output_path,
                cell_size=int(args.cell_size),
                room_name=json_path.stem,
            )
            success += 1
        except Exception as exc:
            failed.append((json_path.name, str(exc)))

    print(f"[visualize] json_files={len(json_files)} success={success} failed={len(failed)}")
    if failed:
        print("[visualize] failed samples:")
        for name, message in failed[:20]:
            print(f"  - {name}: {message}")


if __name__ == "__main__":
    main()
