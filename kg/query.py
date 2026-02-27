from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


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


def _contains(text: Optional[str], keyword: str) -> bool:
    if not text:
        return False
    return keyword.lower() in str(text).lower()


def _tokenize_query(query: str) -> List[str]:
    return [t for t in re.split(r"\s+", (query or "").strip()) if t]


def _adjacency(edges: List[dict]) -> tuple[Dict[str, List[dict]], Dict[str, List[dict]]]:
    out_map: Dict[str, List[dict]] = {}
    in_map: Dict[str, List[dict]] = {}
    for e in edges:
        s = e.get("source")
        t = e.get("target")
        if s:
            out_map.setdefault(s, []).append(e)
        if t:
            in_map.setdefault(t, []).append(e)
    return out_map, in_map


def query_graph_requirements(
    store_dir: Path,
    query: str = "",
    entity: Optional[str] = None,
    metric: Optional[str] = None,
    source_id: Optional[str] = None,
    top_k: int = 10,
) -> Dict:
    results: List[dict] = []
    q_tokens = _tokenize_query(query)
    entity_kw = (entity or "").strip()
    metric_kw = (metric or "").strip()

    for _, doc in _iter_graph_docs(store_dir) or []:
        if source_id and doc.get("source_id") != source_id:
            continue
        node_map = {n.get("id"): n for n in doc.get("nodes", []) if isinstance(n, dict)}
        out_map, in_map = _adjacency(doc.get("edges", []))

        for node in node_map.values():
            if node.get("label") != "Requirement":
                continue

            req_id = node.get("id")
            props = node.get("props") or {}
            req_text = str(props.get("text") or props.get("raw_cell") or "")
            page_no = props.get("page_no")
            source_path = props.get("source_path") or doc.get("source_path")

            entity_names: List[str] = []
            metric_names: List[str] = []
            values: List[str] = []
            clause_text = None

            for e in out_map.get(req_id, []):
                etype = e.get("type")
                target = node_map.get(e.get("target"))
                if not target:
                    continue
                tlabel = target.get("label")
                tprops = target.get("props") or {}
                if etype == "APPLIES_TO" and tlabel == "DomainEntity":
                    entity_names.append(str(tprops.get("canonical_name") or tprops.get("name") or ""))
                elif etype == "CONSTRAINS_METRIC" and tlabel == "Metric":
                    metric_names.append(str(tprops.get("canonical_name") or tprops.get("name") or ""))
                elif etype == "HAS_VALUE_SPEC" and tlabel == "ValueSpec":
                    values.append(str(tprops.get("raw_text") or tprops.get("value") or ""))

            for e in in_map.get(req_id, []):
                if e.get("type") == "CLAUSE_EXPRESSES_REQUIREMENT":
                    clause = node_map.get(e.get("source"))
                    if clause and clause.get("label") == "Clause":
                        clause_text = str((clause.get("props") or {}).get("text") or "")

            score = 0
            if entity_kw:
                if any(entity_kw in name for name in entity_names):
                    score += 5
                else:
                    continue
            if metric_kw:
                if any(metric_kw.lower() in name.lower() for name in metric_names):
                    score += 5
                else:
                    continue

            searchable_parts = [req_text, clause_text or "", " ".join(entity_names), " ".join(metric_names), " ".join(values)]
            joined = " ".join(part for part in searchable_parts if part)
            if q_tokens:
                matched = 0
                for token in q_tokens:
                    if token.lower() in joined.lower():
                        matched += 1
                if matched == 0:
                    continue
                score += matched
            elif not entity_kw and not metric_kw:
                # no filters, no query => skip broad scan
                continue

            results.append(
                {
                    "score": score,
                    "requirement_id": req_id,
                    "source_id": doc.get("source_id"),
                    "source_path": source_path,
                    "page_no": page_no,
                    "requirement_type": props.get("requirement_type"),
                    "modality": props.get("modality"),
                    "text": req_text or clause_text,
                    "clause_text": clause_text,
                    "entities": sorted({x for x in entity_names if x}),
                    "metrics": sorted({x for x in metric_names if x}),
                    "values": sorted({x for x in values if x}),
                }
            )

    results.sort(key=lambda x: (-x["score"], str(x.get("source_path") or ""), str(x.get("requirement_id") or "")))
    return {"results": results[: max(1, min(int(top_k), 100))], "total": len(results)}


def kg_status(store_dir: Path) -> Dict[str, int]:
    docs = 0
    nodes = 0
    edges = 0
    requirements = 0
    for _, doc in _iter_graph_docs(store_dir) or []:
        docs += 1
        node_list = doc.get("nodes", []) if isinstance(doc.get("nodes"), list) else []
        edge_list = doc.get("edges", []) if isinstance(doc.get("edges"), list) else []
        nodes += len(node_list)
        edges += len(edge_list)
        requirements += sum(1 for n in node_list if isinstance(n, dict) and n.get("label") == "Requirement")
    return {"docs": docs, "nodes": nodes, "edges": edges, "requirements": requirements}
