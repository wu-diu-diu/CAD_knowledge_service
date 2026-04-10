"""
预分割数据集，保存 train/val/test 索引文件。

分层策略（双层）：
  - 第一层：regular vs irregular
  - 第二层：irregular 内按 shape 类型分组
  每组内独立按 64:16:20 比例分割，保证各组比例一致。

用法：
  cd /home/chen/punchy/CAD_knowledge_service
  ../.venv/bin/python RL/split_dataset.py \
    --room_dir RL/room_gen/all/json \
    --output_dir RL/room_gen/all/split
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path


def _extract_shape_type(filename: str) -> str:
    match = re.search(r'^shape_(.+)_\d+lamp_\d+', filename)
    return match.group(1) if match else "unknown"


def split_and_save(
    room_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.64,
    val_ratio: float = 0.16,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)

    # 收集所有 JSON 文件，提取 shape_type 和 kind
    by_group: dict[str, list[str]] = defaultdict(list)
    for f in sorted(room_dir.glob("*.json")):
        shape_type = _extract_shape_type(f.name)
        by_group[shape_type].append(f.name)

    train_idx, val_idx, test_idx = [], [], []

    print("[split] Stratified split (regular/irregular × shape):")
    for shape_type in sorted(by_group.keys()):
        files = by_group[shape_type]
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        kind = "regular" if shape_type == "regular" else "irregular"

        for fname in files[:n_train]:
            train_idx.append({"filename": fname, "shape_type": shape_type, "kind": kind})
        for fname in files[n_train:n_train + n_val]:
            val_idx.append({"filename": fname, "shape_type": shape_type, "kind": kind})
        for fname in files[n_train + n_val:]:
            test_idx.append({"filename": fname, "shape_type": shape_type, "kind": kind})

        print(f"  {shape_type:12s}: {n:3d} total -> "
              f"train={n_train:3d}, val={n_val:3d}, test={n - n_train - n_val:3d}")

    print(f"[split] Total: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # 统计 regular/irregular 分布
    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        reg = sum(1 for r in idx if r["kind"] == "regular")
        irr = sum(1 for r in idx if r["kind"] == "irregular")
        print(f"  {split_name}: regular={reg}, irregular={irr}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        path = output_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[split] saved {path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Pre-split room dataset into train/val/test")
    parser.add_argument("--room_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_and_save(Path(args.room_dir), Path(args.output_dir), seed=args.seed)


if __name__ == "__main__":
    main()
