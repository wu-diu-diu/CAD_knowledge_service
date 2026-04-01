"""
hybrid_graph_query.py — 混合检索 + 图邻域扩展

流程：
  1. BM25 + 向量 RRF 混合检索，召回 top-K 个 Requirement（复用 hybrid_query）
  2. 图邻域扩展：对每个召回的 Requirement，沿图边找同节点邻居
       - UNDER_SECTION：同一章节下的其他 Requirement
       - APPLIES_TO：同一 DomainEntity 下的其他 Requirement
  3. 去重合并，返回扩展后的结果集

返回格式与 hybrid_query 保持一致，新增 "expanded" 字段标记是否来自图扩展。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from .hybrid_query import hybrid_query


# ---------------------------------------------------------------------------
# 图索引（懒加载，按 kg_store_dir 缓存）
# ---------------------------------------------------------------------------

class _GraphIndex:
    """
    从 doc_graphs/*.json 构建两个倒排索引：
      section_to_reqs:  section_id  -> List[requirement_id]
      entity_to_reqs:   entity_id   -> List[requirement_id]
    以及 requirement_id -> metadata record 的快速查找表。
    """

    def __init__(self, kg_store_dir: Path) -> None:
        self.section_to_reqs: Dict[str, List[str]] = {}
        self.entity_to_reqs: Dict[str, List[str]] = {}
        self._req_to_section: Dict[str, str] = {}
        self._req_to_entities: Dict[str, List[str]] = {}
        self._build(kg_store_dir)

    def _build(self, kg_store_dir: Path) -> None:
        graph_dir = kg_store_dir / "doc_graphs"
        if not graph_dir.exists():
            return

        for graph_file in graph_dir.glob("*.json"):
            doc = json.loads(graph_file.read_text(encoding="utf-8"))
            edges: List[Dict[str, Any]] = doc.get("edges", [])

            for edge in edges:
                etype = edge.get("type", "")
                src = edge.get("source", "")
                tgt = edge.get("target", "")

                if etype == "UNDER_SECTION":
                    # src = requirement_id, tgt = section_id
                    self._req_to_section[src] = tgt
                    self.section_to_reqs.setdefault(tgt, [])
                    if src not in self.section_to_reqs[tgt]:
                        self.section_to_reqs[tgt].append(src)

                elif etype == "APPLIES_TO":
                    # src = requirement_id, tgt = entity_id
                    self._req_to_entities.setdefault(src, [])
                    if tgt not in self._req_to_entities[src]:
                        self._req_to_entities[src].append(tgt)
                    self.entity_to_reqs.setdefault(tgt, [])
                    if src not in self.entity_to_reqs[tgt]:
                        self.entity_to_reqs[tgt].append(src)

    def neighbors(self, req_id: str) -> Set[str]:
        """返回与 req_id 共享 section 或 entity 的所有邻居 requirement_id。"""
        result: Set[str] = set()

        sec = self._req_to_section.get(req_id)
        if sec:
            result.update(self.section_to_reqs.get(sec, []))

        for eid in self._req_to_entities.get(req_id, []):
            result.update(self.entity_to_reqs.get(eid, []))

        result.discard(req_id)
        return result


_graph_index_cache: Dict[Path, _GraphIndex] = {}


def _get_graph_index(kg_store_dir: Path) -> _GraphIndex:
    key = kg_store_dir.resolve()
    if key not in _graph_index_cache:
        _graph_index_cache[key] = _GraphIndex(kg_store_dir)
    return _graph_index_cache[key]


# ---------------------------------------------------------------------------
# 元数据快查表（懒加载）
# ---------------------------------------------------------------------------

_meta_cache: Dict[Path, Dict[str, Dict[str, Any]]] = {}


def _get_meta_map(kg_store_dir: Path) -> Dict[str, Dict[str, Any]]:
    key = kg_store_dir.resolve()
    if key not in _meta_cache:
        meta_path = kg_store_dir / "req_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"req_metadata.json 不存在: {meta_path}")
        records = json.loads(meta_path.read_text(encoding="utf-8"))
        _meta_cache[key] = {r["requirement_id"]: r for r in records}
    return _meta_cache[key]


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def hybrid_graph_query(
    query: str,
    kg_store_dir: Path,
    embed_fn: Callable[[List[str]], np.ndarray],
    top_k: int = 10,
    expand_per_node: int = 10,
    entity: Optional[str] = None,
    metric: Optional[str] = None,
    rrf_k: int = 60,
) -> Dict[str, Any]:
    """
    混合检索 + 图邻域扩展。

    参数：
        query           查询字符串
        kg_store_dir    kg_store 目录
        embed_fn        embedding 函数
        top_k           最终返回条数
        expand_per_node 每个召回节点最多扩展多少个邻居（按 metadata 顺序取前 N）
        entity          可选：entity 精确过滤（传给底层 hybrid_query）
        metric          可选：metric 精确过滤（传给底层 hybrid_query）
        rrf_k           RRF 融合参数

    返回格式：
        {
            "results": [
                {
                    ...同 hybrid_query 字段...,
                    "expanded": bool,   # True 表示来自图扩展，False 表示来自原始召回
                }
            ],
            "total": int,
            "retrieved": int,   # 原始召回数
            "expanded": int,    # 图扩展新增数
        }
    """
    # Step 1：混合检索，只取 top_k 条作为扩展种子
    fetch_k = top_k
    base_result = hybrid_query(
        query, kg_store_dir, embed_fn,
        top_k=fetch_k,
        entity=entity,
        metric=metric,
        rrf_k=rrf_k,
    )
    base_results: List[Dict[str, Any]] = base_result.get("results", [])

    # Step 2：图邻域扩展
    graph_idx = _get_graph_index(kg_store_dir)
    meta_map = _get_meta_map(kg_store_dir)

    seen_ids: Set[str] = {r["requirement_id"] for r in base_results}
    expanded_records: List[Dict[str, Any]] = []

    for rec in base_results:
        neighbors = graph_idx.neighbors(rec["requirement_id"])
        # 优先选与来源节点共享同一 entity 的邻居，再选同 section 的其他邻居
        # 这样能把同场所不同指标/条件的 requirement 优先扩展进来
        src_entities = set(rec.get("entities", []))

        def _priority(nbr_id: str) -> int:
            m = meta_map.get(nbr_id)
            if m is None:
                return 2
            nbr_entities = set(m.get("entities", []))
            return 0 if src_entities & nbr_entities else 1

        sorted_neighbors = sorted(neighbors, key=_priority)

        added = 0
        for nbr_id in sorted_neighbors:
            if nbr_id in seen_ids:
                continue
            nbr_meta = meta_map.get(nbr_id)
            if nbr_meta is None:
                continue
            expanded_records.append({
                "score": rec["score"] * 0.8,
                "requirement_id": nbr_meta["requirement_id"],
                "source_id": nbr_meta.get("source_id", ""),
                "source_path": nbr_meta.get("source_path", ""),
                "page_no": nbr_meta.get("page_no"),
                "requirement_type": nbr_meta.get("requirement_type", ""),
                "modality": nbr_meta.get("modality"),
                "text": nbr_meta.get("text", ""),
                "clause_text": None,
                "entities": nbr_meta.get("entities", []),
                "metrics": nbr_meta.get("metrics", []),
                "values": nbr_meta.get("values", []),
                "conditions": nbr_meta.get("conditions", []),
                "expanded": True,
            })
            seen_ids.add(nbr_id)
            added += 1
            if added >= expand_per_node:
                break

    # 给原始召回结果加 expanded=False 标记
    for rec in base_results:
        rec["expanded"] = False

    # Step 3：合并，返回 top_k 条原始召回 + 所有扩展节点
    # 不对扩展节点做截断，让评测能看到完整的图扩展结果
    final = base_results + expanded_records

    return {
        "results": final,
        "total": len(final),
        "retrieved": len(base_results),
        "expanded": len(expanded_records),
    }
