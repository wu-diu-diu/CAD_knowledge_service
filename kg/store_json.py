from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .models import GraphDocument


def ensure_kg_dirs(store_dir: Path) -> None:
    (store_dir / "doc_graphs").mkdir(parents=True, exist_ok=True)
    (store_dir / "artifacts").mkdir(parents=True, exist_ok=True)


def graph_doc_path(store_dir: Path, source_id: str) -> Path:
    return store_dir / "doc_graphs" / f"{source_id}.json"


def graph_exists(store_dir: Path, source_id: str) -> bool:
    return graph_doc_path(store_dir, source_id).exists()


def save_graph_document(store_dir: Path, graph_doc: GraphDocument) -> Path:
    ensure_kg_dirs(store_dir)
    out_path = graph_doc_path(store_dir, graph_doc.source_id)
    out_path.write_text(json.dumps(graph_doc.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def save_blocks_snapshot(store_dir: Path, source_id: str, blocks: list[dict]) -> Path:
    ensure_kg_dirs(store_dir)
    out_path = store_dir / "artifacts" / f"{source_id}_blocks.json"
    out_path.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def append_manifest_record(store_dir: Path, record: Dict) -> None:
    ensure_kg_dirs(store_dir)
    manifest_path = store_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
