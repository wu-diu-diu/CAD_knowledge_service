"""
预分割布线数据集，保存 train/val/test 索引文件。

按灯具数量（_n_lamps）分层，64:16:20 分割。

用法：
  cd /home/chen/punchy/CAD_knowledge_service/RL
  ../.venv/bin/python split_wiring_dataset.py \
    --room_dir test_room/layout_room/json \
    --output_dir test_room/layout_room/split
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def _extract_n_lamps(filename: str) -> int:
    m = re.search(r'_(\d+)lamp_', filename)
    return int(m.group(1)) if m else 0


def split_and_save(
    room_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.64,
    val_ratio: float = 0.16,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)

    by_lamps: dict[int, list[str]] = defaultdict(list)
    for f in sorted(room_dir.glob("*.json")):
        n = _extract_n_lamps(f.name)
        by_lamps[n].append(f.name)

    train_idx, val_idx, test_idx = [], [], []

    print("[split] Stratified split by lamp count (64:16:20):")
    for n_lamps in sorted(by_lamps):
        files = by_lamps[n_lamps]
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        for fname in files[:n_train]:
            train_idx.append({"filename": fname, "n_lamps": n_lamps})
        for fname in files[n_train:n_train + n_val]:
            val_idx.append({"filename": fname, "n_lamps": n_lamps})
        for fname in files[n_train + n_val:]:
            test_idx.append({"filename": fname, "n_lamps": n_lamps})

        print(f"  {n_lamps}lamp: {n:3d} total -> train={n_train:3d}, val={n_val:3d}, test={n_test:3d}")

    print(f"[split] Total: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        path = output_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[split] saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_and_save(Path(args.room_dir), Path(args.output_dir), seed=args.seed)


if __name__ == "__main__":
    main()
