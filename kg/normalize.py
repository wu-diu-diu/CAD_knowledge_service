from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .models import GraphBuilder

PUNCT_RE = re.compile(r"[\s\u3000（）()【】\[\]、,，:：;；]")

ENTITY_ALIAS_MAP = {
    "走廊楼梯间": "走廊、楼梯间",
    "楼梯间走廊": "走廊、楼梯间",
    "电梯前厅": "电梯前厅",
}

METRIC_ALIAS_MAP = {
    "照度标准值lx": "照度标准值",
    "照度标准值1x": "照度标准值",
    "ra": "Ra",
    "gr": "GR",
}


def _norm_key(text: str) -> str:
    x = (text or "").strip().lower()
    x = x.replace("１", "1").replace("２", "2").replace("３", "3")
    x = PUNCT_RE.sub("", x)
    return x


def _canonicalize_entity(name: str) -> str:
    key = _norm_key(name)
    if key in ENTITY_ALIAS_MAP:
        return ENTITY_ALIAS_MAP[key]
    return (name or "").strip()


def _canonicalize_metric(name: str) -> str:
    key = _norm_key(name)
    if key in METRIC_ALIAS_MAP:
        return METRIC_ALIAS_MAP[key]
    return (name or "").strip()


def add_entity_normalization(builder: GraphBuilder) -> Dict[str, int]:
    stats = {
        "entity_alias_edges": 0,
        "canonical_entity_nodes": 0,
        "metric_alias_edges": 0,
        "canonical_metric_nodes": 0,
    }

    groups: Dict[str, List[Tuple[str, str]]] = {}
    metric_groups: Dict[str, List[Tuple[str, str]]] = {}

    for node in builder.nodes:
        if node.label == "DomainEntity":
            raw_name = str(node.props.get("name") or node.props.get("canonical_name") or "")
            canonical = _canonicalize_entity(raw_name)
            node.props["canonical_name"] = canonical
            key = _norm_key(canonical) or _norm_key(raw_name)
            if key:
                groups.setdefault(key, []).append((node.node_id, canonical))
        elif node.label == "Metric":
            raw_name = str(node.props.get("name") or "")
            canonical = _canonicalize_metric(raw_name)
            node.props["canonical_name"] = canonical
            key = _norm_key(canonical) or _norm_key(raw_name)
            if key:
                metric_groups.setdefault(key, []).append((node.node_id, canonical))

    for key, items in groups.items():
        canonical_name = sorted({name for _, name in items if name}, key=len)[0] if items else key
        canonical_id = f"entity_canonical:{key}"
        builder.add_node(canonical_id, "CanonicalEntity", name=canonical_name, key=key)
        stats["canonical_entity_nodes"] += 1
        for node_id, _ in items:
            builder.add_edge("ALIAS_OF", node_id, canonical_id)
            stats["entity_alias_edges"] += 1

    for key, items in metric_groups.items():
        canonical_name = sorted({name for _, name in items if name}, key=len)[0] if items else key
        canonical_id = f"metric_canonical:{key}"
        builder.add_node(canonical_id, "CanonicalMetric", name=canonical_name, key=key)
        stats["canonical_metric_nodes"] += 1
        for node_id, _ in items:
            builder.add_edge("ALIAS_OF_METRIC", node_id, canonical_id)
            stats["metric_alias_edges"] += 1

    return stats
