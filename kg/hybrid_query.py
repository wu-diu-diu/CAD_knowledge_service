from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .vector_store import search_requirements


# 模块级 BM25 缓存，避免每次查询重新构建索引
_bm25_cache: Dict[Path, Any] = {}


def _get_bm25(kg_store_dir: Path):
    """懒加载并缓存 BM25 索引。"""
    key = kg_store_dir.resolve()
    if key in _bm25_cache:
        return _bm25_cache[key]

    try:
        from rank_bm25 import BM25Okapi
        import jieba
    except ImportError as e:
        raise RuntimeError("请安装依赖：pip install rank-bm25 jieba") from e

    meta_path = kg_store_dir / "req_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"req_metadata.json 不存在，请先调用 build_requirement_index()。路径: {meta_path}"
        )

    metadata: List[dict] = json.loads(meta_path.read_text(encoding="utf-8"))
    tokenized = [list(jieba.cut(r["search_text"])) for r in metadata]
    bm25 = BM25Okapi(tokenized)

    _bm25_cache[key] = (bm25, metadata, jieba)
    return _bm25_cache[key]


def _bm25_search(query: str, kg_store_dir: Path, top_k: int) -> List[Dict[str, Any]]:
    bm25, metadata, jieba = _get_bm25(kg_store_dir)
    tokens = list(jieba.cut(query))
    scores = bm25.get_scores(tokens)
    top_indices = scores.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        rec = dict(metadata[idx])
        rec["score"] = float(scores[idx])
        results.append(rec)
    return results


def _rrf_merge(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion：合并 BM25 和向量检索结果。
    score(d) = 1/(k + rank_bm25 + 1) + 1/(k + rank_vector + 1)
    """
    rrf_scores: Dict[str, float] = {}
    # requirement_id -> 元数据记录（取第一次出现的）
    rec_map: Dict[str, Dict[str, Any]] = {}

    for rank, rec in enumerate(bm25_results):
        rid = rec["requirement_id"]
        rrf_scores[rid] = rrf_scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
        if rid not in rec_map:
            rec_map[rid] = rec

    for rank, rec in enumerate(vector_results):
        rid = rec["requirement_id"]
        rrf_scores[rid] = rrf_scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
        if rid not in rec_map:
            rec_map[rid] = rec

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for rid, score in merged:
        rec = dict(rec_map[rid])
        rec["score"] = score
        results.append(rec)
    return results


def hybrid_query(
    query: str,
    kg_store_dir: Path,
    embed_fn: Callable[[List[str]], np.ndarray],
    top_k: int = 10,
    entity: Optional[str] = None,
    metric: Optional[str] = None,
    rrf_k: int = 60,
) -> Dict[str, Any]:
    """
    BM25 + 向量 RRF 混合检索，可选按 entity/metric 图结构精确过滤。

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
                "conditions": List[str],
            }
        ],
        "total": int,
    }
    """
    # 多取候选以便过滤后仍有足够结果
    fetch_k = top_k * 3 if (entity or metric) else top_k * 2

    bm25_results = _bm25_search(query, kg_store_dir, top_k=fetch_k)
    vector_results = search_requirements(query, kg_store_dir, embed_fn, top_k=fetch_k)

    merged = _rrf_merge(bm25_results, vector_results, k=rrf_k)

    entity_kw = (entity or "").strip().lower()
    metric_kw = (metric or "").strip().lower()

    results: List[Dict[str, Any]] = []
    for rec in merged:
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
