from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

from .extractors.clause_norms import extract_clause_requirements
from .extractors.structure import extract_structure_graph
from .extractors.table_norms import extract_table_requirements
from .md_parser import parse_markdown_blocks
from .models import GraphBuilder, GraphDocument
from .normalize import add_entity_normalization
from .store_neo4j import neo4j_enabled, upsert_graph_document_neo4j
from .store_json import append_manifest_record, graph_exists, save_blocks_snapshot, save_graph_document


def build_graph_document(md_text: str, source_id: str, source_path: str, source_type: str) -> tuple[GraphDocument, list]:
    blocks = parse_markdown_blocks(md_text)
    builder = GraphBuilder()
    builder.extend(extract_structure_graph(blocks, source_id=source_id, source_path=source_path, source_type=source_type))
    builder.extend(extract_table_requirements(blocks, source_id=source_id, source_path=source_path))
    builder.extend(extract_clause_requirements(blocks, source_id=source_id, source_path=source_path))
    norm_stats = add_entity_normalization(builder)

    requirement_count = sum(1 for node in builder.nodes if node.label == "Requirement")
    table_count = sum(1 for node in builder.nodes if node.label == "Table")
    figure_count = sum(1 for node in builder.nodes if node.label == "Figure")
    clause_count = sum(1 for node in builder.nodes if node.label == "Clause")
    graph_doc = GraphDocument(
        source_id=source_id,
        source_path=source_path,
        source_type=source_type,
        nodes=builder.nodes,
        edges=builder.edges,
        stats={
            "blocks": len(blocks),
            "nodes": len(builder.nodes),
            "edges": len(builder.edges),
            "requirements": requirement_count,
            "tables": table_count,
            "figures": figure_count,
            "clauses": clause_count,
            **norm_stats,
        },
    )
    return graph_doc, blocks


def ingest_markdown_to_kg(
    md_text: str,
    store_dir: Path,
    source_id: str,
    source_path: str,
    source_type: str,
    save_blocks: bool = True,
) -> Dict[str, int]:
    graph_doc, blocks = build_graph_document(
        md_text=md_text,
        source_id=source_id,
        source_path=source_path,
        source_type=source_type,
    )
    graph_path = save_graph_document(store_dir, graph_doc)
    if neo4j_enabled():
        upsert_graph_document_neo4j(graph_doc)
    if save_blocks:
        save_blocks_snapshot(store_dir, source_id, [b.to_dict() for b in blocks])
    append_manifest_record(
        store_dir,
        {
            "ts": time.time(),
            "source_id": source_id,
            "source_path": source_path,
            "source_type": source_type,
            "graph_path": str(graph_path),
            **graph_doc.stats,
        },
    )
    return {
        "nodes": int(graph_doc.stats.get("nodes", 0)),
        "edges": int(graph_doc.stats.get("edges", 0)),
        "requirements": int(graph_doc.stats.get("requirements", 0)),
    }


def kg_already_ingested(store_dir: Path, source_id: str) -> bool:
    return graph_exists(store_dir, source_id)
