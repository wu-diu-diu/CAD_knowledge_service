from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

REQ_INDEX_NAME = "req_index.faiss"
REQ_META_NAME = "req_metadata.json"


def _iter_graph_docs(store_dir: Path):
    docs_dir = store_dir / "doc_graphs"
    if not docs_dir.exists():
        return
    for path in sorted(docs_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            yield path, data


def _extract_requirements(doc: dict) -> List[Dict[str, Any]]:
    """
    从单个 GraphDocument JSON 中提取所有 Requirement 节点，
    并通过遍历出边补充 entity / metric / value / condition 信息。
    """
    node_map: Dict[str, dict] = {n["id"]: n for n in doc.get("nodes", []) if isinstance(n, dict)}
    edges: List[dict] = [e for e in doc.get("edges", []) if isinstance(e, dict)]

    out_map: Dict[str, List[dict]] = {}
    for e in edges:
        out_map.setdefault(e.get("source", ""), []).append(e)

    records: List[Dict[str, Any]] = []
    for node in node_map.values():
        if node.get("label") != "Requirement":
            continue

        props = node.get("props") or {}
        req_id = node["id"]
        req_text = str(props.get("text") or props.get("raw_cell") or "")
        req_type = str(props.get("requirement_type") or "")
        page_no = props.get("page_no")
        source_path = props.get("source_path") or doc.get("source_path", "")
        source_id = props.get("source_id") or doc.get("source_id", "")
        modality = props.get("modality")
        clause_no = props.get("clause_no")
        table_caption = props.get("table_caption")

        entities: List[str] = []
        metrics: List[str] = []
        values: List[str] = []
        conditions: List[str] = []

        for e in out_map.get(req_id, []):
            etype = e.get("type")
            target = node_map.get(e.get("target", ""))
            if not target:
                continue
            tlabel = target.get("label")
            tprops = target.get("props") or {}
            if etype == "APPLIES_TO" and tlabel == "DomainEntity":
                name = str(tprops.get("canonical_name") or tprops.get("name") or "")
                if name:
                    entities.append(name)
            elif etype == "CONSTRAINS_METRIC" and tlabel == "Metric":
                name = str(tprops.get("canonical_name") or tprops.get("name") or "")
                if name:
                    metrics.append(name)
            elif etype == "HAS_VALUE_SPEC" and tlabel == "ValueSpec":
                raw = str(tprops.get("raw_text") or tprops.get("value") or "")
                if raw:
                    values.append(raw)
            elif etype == "UNDER_CONDITION" and tlabel == "Condition":
                ctext = str(tprops.get("text") or "")
                if ctext:
                    conditions.append(ctext)

        # 构造检索文本：entity + metric + condition + text/raw_cell + value
        parts = entities + metrics + conditions
        if req_text:
            parts.append(req_text)
        parts += values
        ## 只保留非空部分，并用空格连接
        search_text = " ".join(p for p in parts if p).strip()
        if not search_text:
            continue

        records.append({
            "requirement_id": req_id,
            "requirement_type": req_type,
            "search_text": search_text,
            "text": req_text,
            "entities": entities,
            "metrics": metrics,
            "values": values,
            "conditions": conditions,
            "modality": modality,
            "clause_no": clause_no,
            "table_caption": table_caption,
            "page_no": page_no,
            "source_path": source_path,
            "source_id": source_id,
        })

    return records


def build_requirement_index(
    kg_store_dir: Path,
    embed_fn: Callable[[List[str]], np.ndarray],
) -> int:
    """
    遍历 kg_store_dir/doc_graphs/*.json，提取所有 Requirement 节点，
    构建 FAISS IndexFlatIP 向量索引，保存到 kg_store_dir。

    返回索引中的向量数量。
    """
    try:
        import faiss
    except ImportError as e:
        raise RuntimeError("faiss 未安装，请安装 faiss-cpu 或 faiss-gpu。") from e

    all_records: List[Dict[str, Any]] = []
    for _, doc in _iter_graph_docs(kg_store_dir):
        all_records.extend(_extract_requirements(doc))

    if not all_records:
        return 0

    texts = [r["search_text"] for r in all_records]
    vectors = embed_fn(texts)  # shape: [N, D], float32, L2-normalized

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_path = kg_store_dir / REQ_INDEX_NAME
    meta_path = kg_store_dir / REQ_META_NAME

    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")

    return len(all_records)


def search_requirements(
    query: str,
    kg_store_dir: Path,
    embed_fn: Callable[[List[str]], np.ndarray],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    向量检索 Requirement 节点。

    返回 top_k 条结果，每条包含：
      requirement_id, text, entities, metrics, values, conditions,
      modality, clause_no, table_caption, page_no, source_path, score
    """
    try:
        import faiss
    except ImportError as e:
        raise RuntimeError("faiss 未安装。") from e

    index_path = kg_store_dir / REQ_INDEX_NAME
    meta_path = kg_store_dir / REQ_META_NAME

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"向量索引不存在，请先调用 build_requirement_index()。"
            f"期望路径: {index_path}"
        )

    index = faiss.read_index(str(index_path))
    metadata: List[dict] = json.loads(meta_path.read_text(encoding="utf-8"))

    q_vec = embed_fn([query])  # [1, D]
    k = min(top_k, index.ntotal)
    scores, indices = index.search(q_vec, k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        rec = dict(metadata[idx])
        rec["score"] = float(score)
        results.append(rec)

    return results


def requirement_index_exists(kg_store_dir: Path) -> bool:
    return (kg_store_dir / REQ_INDEX_NAME).exists() and (kg_store_dir / REQ_META_NAME).exists()


def requirement_index_size(kg_store_dir: Path) -> int:
    meta_path = kg_store_dir / REQ_META_NAME
    if not meta_path.exists():
        return 0
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0
