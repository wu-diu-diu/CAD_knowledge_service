from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MdBlock:
    index: int
    block_type: str
    text: str
    page_no: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)  ## 用于存储额外的元信息，例如表格的标题、JSON块的解析结果等

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "block_type": self.block_type,
            "text": self.text,
            "page_no": self.page_no,
            "meta": self.meta,
        }


@dataclass
class NodeRecord:
    node_id: str
    label: str
    props: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.node_id, "label": self.label, "props": self.props}


@dataclass
class EdgeRecord:
    edge_type: str
    source: str
    target: str
    props: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple:
        props_key = tuple(sorted((k, str(v)) for k, v in self.props.items()))
        return (self.edge_type, self.source, self.target, props_key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.edge_type,
            "source": self.source,
            "target": self.target,
            "props": self.props,
        }


@dataclass
class GraphDocument:
    source_id: str
    source_path: str
    source_type: str
    nodes: List[NodeRecord] = field(default_factory=list)
    edges: List[EdgeRecord] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "stats": self.stats,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }


class GraphBuilder:
    def __init__(self) -> None:
        self._nodes: Dict[str, NodeRecord] = {}
        self._edges: Dict[tuple, EdgeRecord] = {}
    ## 添加节点时，如果节点ID已存在但标签不同，则保留原标签并将新标签添加到_alt_labels属性中以供调试使用；如果节点ID已存在且标签相同，则更新节点属性
    def add_node(self, node_id: str, label: str, **props: Any) -> str:
        clean_props = {k: v for k, v in props.items() if v is not None}
        existing = self._nodes.get(node_id)
        if existing is None:
            self._nodes[node_id] = NodeRecord(node_id=node_id, label=label, props=clean_props)
            return node_id
        if existing.label != label:
            # Keep first label stable; attach alternate labels for debugging.
            labels = set(existing.props.get("_alt_labels", []))
            labels.add(label)
            existing.props["_alt_labels"] = sorted(labels)
        existing.props.update(clean_props)
        return node_id
    ## 添加边时，如果边已存在（即边类型、源节点、目标节点和属性相同），则不重复添加；否则添加新边
    def add_edge(self, edge_type: str, source: str, target: str, **props: Any) -> None:
        edge = EdgeRecord(edge_type=edge_type, source=source, target=target, props={k: v for k, v in props.items() if v is not None})
        self._edges[edge.key()] = edge
    ## 将另一个GraphBuilder中的节点和边合并到当前GraphBuilder中，如果节点ID相同但标签不同，则保留原标签并将新标签添加到_alt_labels属性中以供调试使用；如果边类型、源节点、目标节点和属性相同，则不重复添加
    def extend(self, other: "GraphBuilder") -> None:
        for node in other.nodes:
            self.add_node(node.node_id, node.label, **node.props)
        for edge in other.edges:
            self.add_edge(edge.edge_type, edge.source, edge.target, **edge.props)

    @property
    def nodes(self) -> List[NodeRecord]:
        return list(self._nodes.values())

    @property
    def edges(self) -> List[EdgeRecord]:
        return list(self._edges.values())
