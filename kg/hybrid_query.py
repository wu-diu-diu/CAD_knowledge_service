from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .vector_store import search_requirements


def hybrid_query(
    query: str,
    kg_store_dir: Path,
    embed_fn: Callable[[List[str]], np.ndarray],
    top_k: int = 10,
    entity: Optional[str] = None,
    metric: Optional[str] = None,
) -> Dict[str, Any]:
    """
    向量召回 top_k 个 Requirement 节点，可选按 entity/metric 过滤。

    返回格式与 kg/query.py 的 query_graph_requirements() 保持一致：
    {
        "results": [
            {
                "score": float,
                "requirement_id": str,
                "source_id": str,
                "source_path": str,
                "page_no": int | None,
                "requirement_type": str,
                "modality": str | None,
                "text": str,
                "clause_text": None,
                "entities": List[str],
                "metrics": List[str],
                "values": List[str],
            }
        ],
        "total": int,
    }
    """
    # 向量召回，多取一些候选以便过滤后仍有足够结果
    fetch_k = top_k * 3 if (entity or metric) else top_k
    raw = search_requirements(query, kg_store_dir, embed_fn, top_k=fetch_k)

    entity_kw = (entity or "").strip().lower()
    metric_kw = (metric or "").strip().lower()

    results: List[Dict[str, Any]] = []
    for rec in raw:
        if entity_kw:
            if not any(entity_kw in e.lower() for e in rec.get("entities", [])):
                continue
        if metric_kw:
            if not any(metric_kw in m.lower() for m in rec.get("metrics", [])):
                continue
        results.append({
            "score": rec["score"],
            "requirement_id": rec["requirement_id"],
            "source_id": rec.get("source_id", ""),
            "source_path": rec.get("source_path", ""),
            "page_no": rec.get("page_no"),
            "requirement_type": rec.get("requirement_type", ""),
            "modality": rec.get("modality"),
            "text": rec.get("text", ""),
            "clause_text": None,
            "entities": rec.get("entities", []),
            "metrics": rec.get("metrics", []),
            "values": rec.get("values", []),
            "conditions": rec.get("conditions", []),
        })
        if len(results) >= top_k:
            break

    return {"results": results, "total": len(results)}
