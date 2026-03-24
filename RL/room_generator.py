from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RoomTypeConfig:
    """Configuration for a specific room type."""
    target_lux: float
    lamp_flux_lm: float
    lamp_type: str
    lamp_power_w: float


class RoomGenerator:
    """Multi-shape room generator supporting rectangular, L-shaped, T-shaped, and corridor rooms."""

    def __init__(self):
        self.cell_size_m = 0.4
        self.cell_size_px = 40
        self.uf = 0.6
        self.mf = 0.8

        # Room type configurations
        self.room_type_configs = {
            'office': RoomTypeConfig(
                target_lux=300,
                lamp_flux_lm=3000,
                lamp_type='荧光灯',
                lamp_power_w=35,
            ),
            'lab': RoomTypeConfig(
                target_lux=500,
                lamp_flux_lm=5000,
                lamp_type='LED灯',
                lamp_power_w=50,
            ),
            'corridor': RoomTypeConfig(
                target_lux=200,
                lamp_flux_lm=3000,
                lamp_type='荧光灯',
                lamp_power_w=28,
            ),
            'storage': RoomTypeConfig(
                target_lux=150,
                lamp_flux_lm=2000,
                lamp_type='节能灯',
                lamp_power_w=20,
            ),
        }

    def calculate_lamp_count(self, room_matrix: np.ndarray, function_type: str) -> int:
        """Calculate required lamp count based on illuminance formula."""
        config = self.room_type_configs[function_type]
        room_area_grids = int(np.sum(room_matrix > 0))
        room_area_m2 = room_area_grids * (self.cell_size_m ** 2)

        # E = (N * Φ * UF * MF) / A
        # N = (E * A) / (Φ * UF * MF)
        N = (config.target_lux * room_area_m2) / (
            config.lamp_flux_lm * self.uf * self.mf
        )
        return max(1, int(np.ceil(N)))

    def _random_door_position(self, rows: int, cols: int) -> tuple[int, int]:
        """Generate a random door position on the edge, at least 2 cells from corners."""
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            return (0, random.randint(2, cols - 3))
        elif edge == 'bottom':
            return (rows - 1, random.randint(2, cols - 3))
        elif edge == 'left':
            return (random.randint(2, rows - 3), 0)
        else:  # right
            return (random.randint(2, rows - 3), cols - 1)

    def _place_switch_near_door(self, door_pos: tuple[int, int], rows: int, cols: int) -> tuple[int, int]:
        """Place switch near the door, 1-3 cells away on the same edge."""
        door_r, door_c = door_pos

        # Determine which edge the door is on
        if door_r == 0:  # top edge
            offset = random.randint(1, 3)
            switch_c = max(1, min(cols - 2, door_c + random.choice([-offset, offset])))
            return (0, switch_c)
        elif door_r == rows - 1:  # bottom edge
            offset = random.randint(1, 3)
            switch_c = max(1, min(cols - 2, door_c + random.choice([-offset, offset])))
            return (rows - 1, switch_c)
        elif door_c == 0:  # left edge
            offset = random.randint(1, 3)
            switch_r = max(1, min(rows - 2, door_r + random.choice([-offset, offset])))
            return (switch_r, 0)
        else:  # right edge
            offset = random.randint(1, 3)
            switch_r = max(1, min(rows - 2, door_r + random.choice([-offset, offset])))
            return (switch_r, cols - 1)

    def generate_rectangular_room(self, rows: int, cols: int, function_type: str = 'office') -> np.ndarray:
        """Generate a rectangular room."""
        matrix = np.ones((rows, cols), dtype=np.int32)

        # Add door
        door_pos = self._random_door_position(rows, cols)
        matrix[door_pos] = 2

        # Add switch near door
        switch_pos = self._place_switch_near_door(door_pos, rows, cols)
        matrix[switch_pos] = 3

        # Optionally add internal obstacles (10% probability)
        if random.random() < 0.1 and rows > 12 and cols > 12:
            # Add 1-2 internal walls or pillars
            num_obstacles = random.randint(1, 2)
            for _ in range(num_obstacles):
                obs_r = random.randint(2, rows - 3)
                obs_c = random.randint(2, cols - 3)
                # Make sure not to block door or switch
                if matrix[obs_r, obs_c] == 1:
                    matrix[obs_r, obs_c] = 0

        return matrix

    def generate_l_shaped_room(self, function_type: str = 'office') -> np.ndarray:
        """Generate an L-shaped room by combining two rectangles."""
        # Generate two rectangles
        rect1_rows = random.randint(8, 20)
        rect1_cols = random.randint(8, 20)
        rect2_rows = random.randint(8, 20)
        rect2_cols = random.randint(8, 20)

        # Determine L-shape orientation
        orientation = random.choice(['bottom_right', 'bottom_left', 'top_right', 'top_left'])

        # Create combined matrix
        if orientation == 'bottom_right':
            total_rows = rect1_rows + rect2_rows
            total_cols = max(rect1_cols, rect2_cols)
            matrix = np.zeros((total_rows, total_cols), dtype=np.int32)
            matrix[:rect1_rows, :rect1_cols] = 1
            matrix[rect1_rows:, total_cols - rect2_cols:] = 1
        elif orientation == 'bottom_left':
            total_rows = rect1_rows + rect2_rows
            total_cols = max(rect1_cols, rect2_cols)
            matrix = np.zeros((total_rows, total_cols), dtype=np.int32)
            matrix[:rect1_rows, total_cols - rect1_cols:] = 1
            matrix[rect1_rows:, :rect2_cols] = 1
        elif orientation == 'top_right':
            total_rows = rect1_rows + rect2_rows
            total_cols = max(rect1_cols, rect2_cols)
            matrix = np.zeros((total_rows, total_cols), dtype=np.int32)
            matrix[rect2_rows:, :rect1_cols] = 1
            matrix[:rect2_rows, total_cols - rect2_cols:] = 1
        else:  # top_left
            total_rows = rect1_rows + rect2_rows
            total_cols = max(rect1_cols, rect2_cols)
            matrix = np.zeros((total_rows, total_cols), dtype=np.int32)
            matrix[rect2_rows:, total_cols - rect1_cols:] = 1
            matrix[:rect2_rows, :rect2_cols] = 1

        # Add door on outer edge
        valid_positions = []
        rows, cols = matrix.shape
        for r in range(rows):
            for c in range(cols):
                if matrix[r, c] == 1:
                    # Check if on edge
                    if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                        valid_positions.append((r, c))

        if valid_positions:
            door_pos = random.choice(valid_positions)
            matrix[door_pos] = 2

            # Add switch near door
            switch_candidates = []
            dr, dc = door_pos
            for offset in range(1, 4):
                for nr, nc in [(dr + offset, dc), (dr - offset, dc), (dr, dc + offset), (dr, dc - offset)]:
                    if 0 <= nr < rows and 0 <= nc < cols and matrix[nr, nc] == 1:
                        switch_candidates.append((nr, nc))
            if switch_candidates:
                switch_pos = random.choice(switch_candidates)
                matrix[switch_pos] = 3

        return matrix

    def generate_t_shaped_room(self, function_type: str = 'office') -> np.ndarray:
        """Generate a T-shaped room by combining three rectangles."""
        # Main stem (vertical)
        stem_rows = random.randint(16, 28)
        stem_cols = random.randint(8, 12)

        # Horizontal bar
        bar_rows = random.randint(8, 12)
        bar_cols = random.randint(20, 32)

        # Create combined matrix
        total_rows = stem_rows + bar_rows
        total_cols = bar_cols
        matrix = np.zeros((total_rows, total_cols), dtype=np.int32)

        # Place horizontal bar at top
        matrix[:bar_rows, :] = 1

        # Place vertical stem in the middle
        stem_start_col = (bar_cols - stem_cols) // 2
        matrix[bar_rows:, stem_start_col:stem_start_col + stem_cols] = 1

        # Add door at one of the endpoints
        door_positions = [
            (0, random.randint(2, bar_cols - 3)),  # top
            (total_rows - 1, stem_start_col + random.randint(1, stem_cols - 2)),  # bottom
        ]
        door_pos = random.choice(door_positions)
        matrix[door_pos] = 2

        # Add switch near door
        dr, dc = door_pos
        switch_candidates = []
        for offset in range(1, 4):
            for nr, nc in [(dr + offset, dc), (dr - offset, dc), (dr, dc + offset), (dr, dc - offset)]:
                if 0 <= nr < total_rows and 0 <= nc < total_cols and matrix[nr, nc] == 1:
                    switch_candidates.append((nr, nc))
        if switch_candidates:
            switch_pos = random.choice(switch_candidates)
            matrix[switch_pos] = 3

        return matrix

    def generate_corridor(self, function_type: str = 'corridor') -> np.ndarray:
        """Generate a corridor (long narrow room)."""
        length = random.randint(40, 100)
        width = random.randint(6, 12)

        # Randomly choose horizontal or vertical orientation
        if random.random() < 0.5:
            rows, cols = width, length
        else:
            rows, cols = length, width

        matrix = np.ones((rows, cols), dtype=np.int32)

        # Add doors at both ends
        if rows < cols:  # horizontal corridor
            door1_pos = (random.randint(1, rows - 2), 0)
            door2_pos = (random.randint(1, rows - 2), cols - 1)
        else:  # vertical corridor
            door1_pos = (0, random.randint(1, cols - 2))
            door2_pos = (rows - 1, random.randint(1, cols - 2))

        matrix[door1_pos] = 2
        matrix[door2_pos] = 2

        # Add switch near one door
        dr, dc = door1_pos
        switch_candidates = []
        for offset in range(1, 4):
            for nr, nc in [(dr + offset, dc), (dr - offset, dc), (dr, dc + offset), (dr, dc - offset)]:
                if 0 <= nr < rows and 0 <= nc < cols and matrix[nr, nc] == 1:
                    switch_candidates.append((nr, nc))
        if switch_candidates:
            switch_pos = random.choice(switch_candidates)
            matrix[switch_pos] = 3

        return matrix

    def generate_room_sample(self, shape_type: str | None = None, function_type: str | None = None) -> dict[str, Any]:
        """Generate a complete room sample with calculated lamp count."""
        # Determine shape type
        if shape_type is None:
            shape_type = random.choices(
                ['rectangular', 'l_shaped', 't_shaped', 'corridor'],
                weights=[0.6, 0.2, 0.1, 0.1],
            )[0]

        # Determine function type based on shape
        if function_type is None:
            if shape_type == 'rectangular':
                function_type = random.choices(['office', 'lab', 'storage'], weights=[0.6, 0.3, 0.1])[0]
            elif shape_type in ['l_shaped', 't_shaped']:
                function_type = random.choices(['office', 'lab'], weights=[0.8, 0.2])[0]
            else:  # corridor
                function_type = 'corridor'

        # Generate room matrix
        if shape_type == 'rectangular':
            rows = random.randint(10, 32)
            cols = random.randint(10, 32)
            matrix = self.generate_rectangular_room(rows, cols, function_type)
        elif shape_type == 'l_shaped':
            matrix = self.generate_l_shaped_room(function_type)
        elif shape_type == 't_shaped':
            matrix = self.generate_t_shaped_room(function_type)
        else:  # corridor
            matrix = self.generate_corridor(function_type)

        # Calculate lamp count
        lamp_count = self.calculate_lamp_count(matrix, function_type)

        # Get room type config
        config = self.room_type_configs[function_type]

        # Calculate room area
        room_area_grids = int(np.sum(matrix > 0))
        room_area_m2 = room_area_grids * (self.cell_size_m ** 2)

        # Generate room name
        room_name = f"generated_{shape_type}_{function_type}_{uuid.uuid4().hex[:8]}"

        return {
            'room_name': room_name,
            'grid_rows': int(matrix.shape[0]),
            'grid_cols': int(matrix.shape[1]),
            'cell_size_px': self.cell_size_px,
            'illuminance': config.target_lux,
            'lamp_type': config.lamp_type,
            'lamp': {
                'lamp_power_w': config.lamp_power_w,
                'lamp_luminous_flux_lm': config.lamp_flux_lm,
                'uf': self.uf,
                'mf': self.mf,
                'lamp_count': lamp_count,
            },
            'matrix': matrix.tolist(),
            'room_area_m2': room_area_m2,
            'room_type': shape_type,
            'function_type': function_type,
        }

    def augment_room(self, room_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Augment a room with rotations and flips."""
        augmented = []
        matrix = np.array(room_data['matrix'], dtype=np.int32)
        base_name = room_data['room_name']

        # Original
        augmented.append(room_data)

        # Rotation 90°
        rot90_matrix = np.rot90(matrix, k=1)
        augmented.append({
            **room_data,
            'matrix': rot90_matrix.tolist(),
            'grid_rows': rot90_matrix.shape[0],
            'grid_cols': rot90_matrix.shape[1],
            'room_name': f"{base_name}_rot90",
        })

        # Rotation 180°
        rot180_matrix = np.rot90(matrix, k=2)
        augmented.append({
            **room_data,
            'matrix': rot180_matrix.tolist(),
            'grid_rows': rot180_matrix.shape[0],
            'grid_cols': rot180_matrix.shape[1],
            'room_name': f"{base_name}_rot180",
        })

        # Rotation 270°
        rot270_matrix = np.rot90(matrix, k=3)
        augmented.append({
            **room_data,
            'matrix': rot270_matrix.tolist(),
            'grid_rows': rot270_matrix.shape[0],
            'grid_cols': rot270_matrix.shape[1],
            'room_name': f"{base_name}_rot270",
        })

        # Horizontal flip
        fliph_matrix = np.fliplr(matrix)
        augmented.append({
            **room_data,
            'matrix': fliph_matrix.tolist(),
            'room_name': f"{base_name}_fliph",
        })

        # Vertical flip
        flipv_matrix = np.flipud(matrix)
        augmented.append({
            **room_data,
            'matrix': flipv_matrix.tolist(),
            'room_name': f"{base_name}_flipv",
        })

        return augmented

    def generate_batch(
        self,
        count: int,
        output_dir: str | Path,
        augment: bool = False,
        visualize: bool = False,
    ) -> list[Path]:
        """Generate a batch of room samples and save to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for i in range(count):
            room_data = self.generate_room_sample()

            if augment:
                augmented_rooms = self.augment_room(room_data)
            else:
                augmented_rooms = [room_data]

            for room in augmented_rooms:
                filename = f"{room['room_name']}.json"
                filepath = output_path / filename

                # Save as single-room JSON (compatible with test_room.json format)
                with filepath.open('w', encoding='utf-8') as f:
                    json.dump({room['room_name']: room}, f, ensure_ascii=False, indent=2)

                saved_files.append(filepath)

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{count} base rooms...")

        print(f"Total files saved: {len(saved_files)}")
        return saved_files


def main():
    """Command-line interface for room generator."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic room samples for RL training')
    parser.add_argument('--count', type=int, default=100, help='Number of base rooms to generate')
    parser.add_argument('--output', type=str, default='RL/generated_rooms', help='Output directory')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation (6x)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization images')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Generate rooms
    generator = RoomGenerator()
    saved_files = generator.generate_batch(
        count=args.count,
        output_dir=args.output,
        augment=args.augment,
        visualize=args.visualize,
    )

    print(f"\n=== Generation Summary ===")
    print(f"Base rooms: {args.count}")
    print(f"Total files: {len(saved_files)}")
    print(f"Output directory: {args.output}")
    print(f"Augmentation: {'enabled (6x)' if args.augment else 'disabled'}")

    # Show sample statistics
    print(f"\n=== Sample Statistics ===")
    sample_files = saved_files[:min(5, len(saved_files))]
    for filepath in sample_files:
        with filepath.open('r') as f:
            data = json.load(f)
            room = list(data.values())[0]
            print(f"{room['room_name']}: {room['grid_rows']}x{room['grid_cols']}, "
                  f"type={room['room_type']}, function={room['function_type']}, "
                  f"lamps={room['lamp']['lamp_count']}")


if __name__ == '__main__':
    main()
