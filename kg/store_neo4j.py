from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

from .models import GraphDocument

try:
    from neo4j import GraphDatabase, READ_ACCESS
except ImportError:  # pragma: no cover - optional dependency
    GraphDatabase = None
    READ_ACCESS = None


LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_driver = None


def neo4j_enabled() -> bool:
    return os.getenv("KG_NEO4J_ENABLED", "0").lower() in {"1", "true", "yes"}


def _cfg(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


def _safe_label(name: str, fallback: str = "KGNodeType") -> str:
    value = (name or "").strip()
    return value if LABEL_RE.match(value) else fallback


def _safe_rel(name: str, fallback: str = "KG_REL") -> str:
    value = (name or "").strip()
    return value if REL_RE.match(value) else fallback


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _edge_key(edge: dict) -> str:
    payload = {
        "type": edge["type"],
        "source": edge["source"],
        "target": edge["target"],
        "props": edge.get("props", {}),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def get_driver():
    global _driver
    if _driver is not None:
        return _driver
    if not neo4j_enabled():
        return None
    if GraphDatabase is None:
        raise RuntimeError("neo4j package is not installed. Install dependency `neo4j` first.")

    uri = _cfg("KG_NEO4J_URI")
    user = _cfg("KG_NEO4J_USER")
    password = _cfg("KG_NEO4J_PASSWORD")
    if not uri or not user or not password:
        raise RuntimeError("KG_NEO4J_ENABLED=1 but KG_NEO4J_URI/KG_NEO4J_USER/KG_NEO4J_PASSWORD are not fully configured.")

    _driver = GraphDatabase.driver(uri, auth=(user, password))
    _ensure_schema(_driver, _cfg("KG_NEO4J_DATABASE"))
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def _run_write(driver, cypher: str, params: Optional[dict] = None, database: Optional[str] = None) -> None:
    with driver.session(database=database) as session:
        session.execute_write(lambda tx: tx.run(cypher, params or {}).consume())


def _ensure_schema(driver, database: Optional[str]) -> None:
    # Neo4j 5 supports IF NOT EXISTS.
    statements = [
        "CREATE CONSTRAINT kg_node_id IF NOT EXISTS FOR (n:KGNode) REQUIRE n.id IS UNIQUE",
        "CREATE INDEX kg_node_kind IF NOT EXISTS FOR (n:KGNode) ON (n.kind)",
        "CREATE INDEX kg_node_source_id IF NOT EXISTS FOR (n:KGNode) ON (n.source_id)",
    ]
    for stmt in statements:
        _run_write(driver, stmt, database=database)


def _group_nodes(nodes: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for node in nodes:
        label = _safe_label(node.get("label") or "KGNodeType")
        props = {"id": node["id"], "kind": node.get("label"), **{k: _jsonable(v) for k, v in (node.get("props") or {}).items()}}
        groups.setdefault(label, []).append({"id": node["id"], "props": props})
    return groups


def _group_edges(edges: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for edge in edges:
        rel_type = _safe_rel(edge.get("type") or "KG_REL")
        props = {k: _jsonable(v) for k, v in (edge.get("props") or {}).items()}
        props["edge_type"] = edge.get("type")
        groups.setdefault(rel_type, []).append(
            {
                "source": edge["source"],
                "target": edge["target"],
                "props": props,
                "key": _edge_key(edge),
            }
        )
    return groups


def _batched(rows: List[dict], size: int = 500) -> Iterable[List[dict]]:
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def upsert_graph_document_neo4j(graph_doc: GraphDocument) -> None:
    driver = get_driver()
    if driver is None:
        return

    database = _cfg("KG_NEO4J_DATABASE")
    node_groups = _group_nodes([n.to_dict() for n in graph_doc.nodes])
    edge_groups = _group_edges([e.to_dict() for e in graph_doc.edges])

    # Stamp graph metadata on document root node as well.
    for rows in node_groups.values():
        for row in rows:
            row["props"].setdefault("source_id", graph_doc.source_id)
            row["props"].setdefault("source_path", graph_doc.source_path)
            row["props"].setdefault("source_type", graph_doc.source_type)

    for label, rows in node_groups.items():
        cypher = f"""
        UNWIND $rows AS row
        MERGE (n:KGNode:{label} {{id: row.id}})
        SET n += row.props
        """
        for batch in _batched(rows):
            _run_write(driver, cypher, {"rows": batch}, database=database)

    for rel_type, rows in edge_groups.items():
        cypher = f"""
        UNWIND $rows AS row
        MATCH (s:KGNode {{id: row.source}})
        MATCH (t:KGNode {{id: row.target}})
        MERGE (s)-[r:{rel_type} {{key: row.key}}]->(t)
        SET r += row.props
        """
        for batch in _batched(rows):
            _run_write(driver, cypher, {"rows": batch}, database=database)


def run_cypher_query(cypher: str, params: Optional[dict] = None, top_k: int = 1000) -> Dict[str, Any]:
    driver = get_driver()
    if driver is None:
        raise RuntimeError("Neo4j is not enabled. Set KG_NEO4J_ENABLED=1 and configure connection env vars.")

    database = _cfg("KG_NEO4J_DATABASE")
    query = (cypher or "").strip()
    if not query:
        raise ValueError("Empty cypher query.")

    params = dict(params or {})
    params.setdefault("limit", int(max(1, min(top_k, 5000))))

    # Execute in read mode for safety (GraphRAG query use-case).
    session_kwargs = {"database": database}
    if READ_ACCESS is not None:
        session_kwargs["default_access_mode"] = READ_ACCESS
    with driver.session(**session_kwargs) as session:
        records = session.execute_read(lambda tx: list(tx.run(query, params)))

    result_rows: List[Dict[str, Any]] = []
    keys: List[str] = []
    for rec in records:
        if not keys:
            keys = list(rec.keys())
        row: Dict[str, Any] = {}
        for k in rec.keys():
            v = rec.get(k)
            if hasattr(v, "items") and hasattr(v, "get"):  # neo4j Node/Relationship behaves mapping-like
                row[k] = dict(v)
            else:
                row[k] = _jsonable(v)
        result_rows.append(row)

    return {"keys": keys, "rows": result_rows, "count": len(result_rows)}
