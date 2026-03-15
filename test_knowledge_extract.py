from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from kg.extractors.clause_norms import (
    CLAUSE_RE as CLAUSE_TEXT_RE,
    _detect_modality,
    _extract_conditions,
    _extract_control_methods,
    _extract_entities,
    _extract_metrics,
    _extract_value_tokens,
    extract_clause_requirements,
)
from kg.extractors.structure import extract_structure_graph
from kg.extractors.table_norms import (
    _looks_value_cell,
    _normalize_headers,
    _parse_numeric_prefix,
    _pick_entity_name,
    extract_table_requirements,
)
from kg.md_parser import parse_markdown_blocks
from kg.models import EdgeRecord, GraphBuilder, MdBlock, NodeRecord
from kg.normalize import add_entity_normalization


DEFAULT_MD_PATH = Path("/home/chen/punchy/CAD_knowledge_service/test_files/test_clause.md")
SOURCE_ID = "test1_processed_demo"
SOURCE_TYPE = "local_file"


def rule(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def preview(text: str, limit: int = 160) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def block_brief(block: MdBlock) -> str:
    extra = []
    if block.page_no is not None:
        extra.append(f"page={block.page_no}")
    if block.block_type == "heading":
        extra.append(f"level={int((block.meta or {}).get('level', 1))}")
    if block.block_type == "table":
        caption = (block.meta or {}).get("caption")
        if caption:
            extra.append(f"caption={caption}")
    extra_text = ", ".join(extra)
    return f"[block {block.index}] type={block.block_type}" + (f" ({extra_text})" if extra_text else "")


def evidence_id(source_id: str, block: MdBlock) -> str:
    digest = hashlib.sha1(block.text.encode("utf-8")).hexdigest()[:12]
    return f"evidence:{source_id}:{block.index}:{digest}"


def find_node(builder: GraphBuilder, node_id: str) -> Optional[NodeRecord]:
    for node in builder.nodes:
        if node.node_id == node_id:
            return node
    return None


def find_edges(
    builder: GraphBuilder,
    *,
    source: Optional[str] = None,
    target: Optional[str] = None,
    edge_type: Optional[str] = None,
) -> list[EdgeRecord]:
    edges: list[EdgeRecord] = []
    for edge in builder.edges:
        if source is not None and edge.source != source:
            continue
        if target is not None and edge.target != target:
            continue
        if edge_type is not None and edge.edge_type != edge_type:
            continue
        edges.append(edge)
    return edges


def print_node(node: NodeRecord) -> None:
    print(f"  节点: {node.label} | id={node.node_id}")
    for key, value in sorted(node.props.items()):
        print(f"    - {key}: {value}")


def print_edge(edge: EdgeRecord) -> None:
    print(f"  边: {edge.edge_type} | {edge.source} -> {edge.target}")
    if edge.props:
        for key, value in sorted(edge.props.items()):
            print(f"    - {key}: {value}")


def print_connected_edges(builder: GraphBuilder, node_id: str) -> None:
    connected = [edge for edge in builder.edges if edge.source == node_id or edge.target == node_id]
    if not connected:
        print("  相连边: 无")
        return
    print("  相连边:")
    for edge in connected:
        print_edge(edge)


def explain_blocks(blocks: list[MdBlock]) -> None:
    rule("1. Markdown 分块结果")
    counts = Counter(block.block_type for block in blocks)
    print(f"总块数: {len(blocks)}")
    for block_type, count in sorted(counts.items()):
        print(f"- {block_type}: {count}")

    for block in blocks:
        print()
        print(block_brief(block))
        print(f"  文本预览: {preview(block.text, 240)}")


def explain_structure(blocks: list[MdBlock], source_id: str, source_path: str) -> GraphBuilder:
    builder = extract_structure_graph(
        blocks,
        source_id=source_id,
        source_path=source_path,
        source_type=SOURCE_TYPE,
    )

    rule("2. 文档结构抽取（extract_structure_graph）")
    print("该阶段负责从 page / heading / paragraph / table 中抽取文档骨架节点。")

    doc_id = f"doc:{source_id}"
    doc_node = find_node(builder, doc_id)
    if doc_node is not None:
        print()
        print("文档根节点:")
        print_node(doc_node)

    for block in blocks:
        print()
        print(block_brief(block))
        print(f"  文本预览: {preview(block.text, 160)}")

        if block.block_type == "page_marker":
            page_no = int((block.meta or {}).get("page_no") or block.page_no or 0)
            page_id = f"page:{source_id}:{page_no}"
            page_node = find_node(builder, page_id)
            if page_node is None:
                print("  结构抽取结果: 无")
                continue
            print("  结构抽取结果:")
            print_node(page_node)
            print_connected_edges(builder, page_id)
            continue

        ev_id = evidence_id(source_id, block)
        ev_node = find_node(builder, ev_id)
        if ev_node is not None:
            print("  EvidenceSpan:")
            print_node(ev_node)
            print_connected_edges(builder, ev_id)

        if block.block_type == "heading":
            section_id = f"section:{source_id}:{block.index}"
            section_node = find_node(builder, section_id)
            if section_node is None:
                print("  结构抽取结果: 无 Section 节点")
                continue
            print("  Section 节点:")
            print_node(section_node)
            print_connected_edges(builder, section_id)
            continue

        if block.block_type == "paragraph":
            clause_match = CLAUSE_TEXT_RE.match(block.text)
            if not clause_match:
                print("  结构抽取结果: 不是条款号开头的段落，因此不会生成 Clause 节点。")
                continue
            clause_id = f"clause:{source_id}:{block.index}"
            clause_node = find_node(builder, clause_id)
            if clause_node is None:
                print("  结构抽取结果: 未生成 Clause 节点。")
                continue
            print("  Clause 节点:")
            print_node(clause_node)
            print_connected_edges(builder, clause_id)
            continue

        if block.block_type == "table":
            table_id = f"table:{source_id}:{block.index}"
            table_node = find_node(builder, table_id)
            if table_node is None:
                print("  结构抽取结果: 未生成 Table 节点。")
                continue
            print("  Table 节点:")
            print_node(table_node)
            print_connected_edges(builder, table_id)
            row_edges = find_edges(builder, source=table_id, edge_type="HAS_ROW")
            print(f"  该表生成了 {len(row_edges)} 个 TableRow 节点。")
            if row_edges:
                print("  前 3 个 TableRow 关系示例:")
                for edge in row_edges[:3]:
                    print_edge(edge)
            continue

    return builder


def explain_clause_extraction(blocks: list[MdBlock], source_id: str, source_path: str) -> GraphBuilder:
    builder = extract_clause_requirements(blocks, source_id=source_id, source_path=source_path)

    rule("3. 条文知识抽取（extract_clause_requirements）")
    print("该阶段会识别主条文，并把其后的编号分项拆成多条 Requirement；同时抽取对象、条件、控制方式、指标和值。")

    for block in blocks:
        if block.block_type != "paragraph":
            continue

        print()
        print(block_brief(block))
        print(f"  原文: {block.text}")

        match = CLAUSE_TEXT_RE.match(block.text)
        if not match:
            print("  结果: 跳过。原因: 该段落不符合“条款号 + 正文”的正则格式。")
            continue

        clause_no, clause_text = match.groups()
        token, modality = _detect_modality(clause_text)
        metrics = _extract_metrics(clause_text)
        entities = _extract_entities(clause_text, token)
        conditions = _extract_conditions(clause_text)
        control_methods = _extract_control_methods(clause_text)
        values = _extract_value_tokens(clause_text)

        print(f"  条款号: {clause_no}")
        print(f"  规范性触发词: {token}")
        print(f"  modality: {modality}")
        print(f"  指标抽取: {metrics}")
        print(f"  对象抽取: {entities}")
        print(f"  条件抽取: {conditions}")
        print(f"  控制方式抽取: {control_methods}")
        print(f"  数值抽取: {values}")

        clause_id = f"clause:{source_id}:{block.index}"
        req_edges = find_edges(builder, source=clause_id, edge_type="CLAUSE_EXPRESSES_REQUIREMENT")
        if not req_edges:
            print("  图谱结果: 未生成 Requirement。原因: 没有识别到“应/宜/不得/必须”等规范性模态。")
            continue

        print(f"  图谱结果: 共生成 {len(req_edges)} 条 Requirement")
        for edge in req_edges:
            req_node = find_node(builder, edge.target)
            if req_node is None:
                continue
            print("  Requirement 节点:")
            print_node(req_node)
            related_edges = [
                rel
                for rel in builder.edges
                if rel.source == req_node.node_id or rel.target == req_node.node_id
            ]
            print("  与该 Requirement 相关的边:")
            for rel in related_edges:
                print_edge(rel)

    return builder


def explain_table_extraction(blocks: list[MdBlock], source_id: str, source_path: str) -> GraphBuilder:
    builder = extract_table_requirements(blocks, source_id=source_id, source_path=source_path)

    rule("4. 表格知识抽取（extract_table_requirements）")
    print("该阶段只处理 table 块。它会按“第一列主对象 + 可选第二列子对象/场景 + 后续属性列”生成 Requirement。")

    for block in blocks:
        if block.block_type != "table":
            continue

        table = (block.meta or {}).get("table") or {}
        headers = _normalize_headers(table.get("headers") or [])
        rows = table.get("rows") or []

        print()
        print(block_brief(block))
        print(f"  表题: {(block.meta or {}).get('caption')}")
        print(f"  归一化表头: {headers}")
        print(f"  数据行数: {len(rows)}")

        table_req_ids = set()
        for row_idx, row in enumerate(rows, start=1):
            padded_row = list(row) + [""] * max(0, len(headers) - len(row))
            entity_name = (padded_row[0] or "").strip() if padded_row else None
            print(f"  第 {row_idx} 行: {padded_row}")
            print(f"    - 识别出的主体对象(entity): {entity_name}")

            extracted_any = False
            for col_idx, (header, cell) in enumerate(zip(headers, padded_row), start=1):
                cell_text = (cell or "").strip()
                if not cell_text:
                    continue
                if len(headers) >= 2 and col_idx <= 2 and _normalize_headers(headers[:2])[0].split("#")[0] == _normalize_headers(headers[:2])[1].split("#")[0]:
                    continue
                if len(headers) < 2 and col_idx == 1:
                    continue
                if len(headers) >= 2 and not (_normalize_headers(headers[:2])[0].split("#")[0] == _normalize_headers(headers[:2])[1].split("#")[0]) and col_idx == 1:
                    continue
                extracted_any = True
                numeric_value = _parse_numeric_prefix(cell_text)
                print(
                    f"    - 单元格约束: col={col_idx}, metric={header}, raw='{cell_text}', parsed_value={numeric_value}"
                )
                req_id = f"req:{source_id}:{block.index}:{row_idx}:{col_idx}"
                table_req_ids.add(req_id)
                req_node = find_node(builder, req_id)
                if req_node is not None:
                    print("      Requirement 节点:")
                    print_node(req_node)
                    for edge in find_edges(builder, source=req_id):
                        print_edge(edge)
                    for edge in find_edges(builder, target=req_id):
                        print_edge(edge)

            reference_plane = None
            for header, cell in zip(headers, padded_row):
                if "参考平面" in header and str(cell).strip():
                    reference_plane = str(cell).strip()
                    break
            if reference_plane:
                scope_id = f"table_row_scope:{source_id}:{block.index}:{row_idx}"
                scope_node = find_node(builder, scope_id)
                if scope_node is not None:
                    print(f"    - 行级上下文条件: {reference_plane}")
                    print_node(scope_node)
                    for edge in find_edges(builder, source=scope_id):
                        print_edge(edge)
                    for edge in find_edges(builder, target=scope_id):
                        print_edge(edge)

            if not extracted_any:
                print("    - 本行没有识别到可抽取的数值单元格。")

        entity_ids = sorted(
            {
                edge.target
                for edge in builder.edges
                if edge.edge_type == "APPLIES_TO"
                and edge.source in table_req_ids
            }
        )
        if entity_ids:
            print("  本表抽取出的 DomainEntity 及其相连节点:")
            for entity_id in entity_ids:
                entity_node = find_node(builder, entity_id)
                if entity_node is None:
                    continue
                print("  DomainEntity 节点:")
                print_node(entity_node)
                print_connected_edges(builder, entity_id)
                neighbor_ids = []
                for edge in builder.edges:
                    if edge.source == entity_id:
                        neighbor_ids.append(edge.target)
                    elif edge.target == entity_id:
                        neighbor_ids.append(edge.source)
                seen = set()
                ordered_neighbor_ids = []
                for nid in neighbor_ids:
                    if nid in seen:
                        continue
                    seen.add(nid)
                    ordered_neighbor_ids.append(nid)
                if ordered_neighbor_ids:
                    print("  相邻节点:")
                    for nid in ordered_neighbor_ids:
                        node = find_node(builder, nid)
                        if node is not None:
                            print_node(node)

    return builder


def explain_final_graph(
    structure_builder: GraphBuilder,
    table_builder: GraphBuilder,
    clause_builder: GraphBuilder,
) -> None:
    merged = GraphBuilder()
    merged.extend(structure_builder)
    merged.extend(table_builder)
    merged.extend(clause_builder)
    norm_stats = add_entity_normalization(merged)

    rule("5. 合并后的最终图谱统计")
    label_counts = Counter(node.label for node in merged.nodes)
    edge_counts = Counter(edge.edge_type for edge in merged.edges)

    print(f"节点总数: {len(merged.nodes)}")
    print(f"边总数: {len(merged.edges)}")
    print("节点类型分布:")
    for label, count in sorted(label_counts.items()):
        print(f"- {label}: {count}")

    print("关系类型分布:")
    for edge_type, count in sorted(edge_counts.items()):
        print(f"- {edge_type}: {count}")

    print("归一化统计:")
    for key, value in sorted(norm_stats.items()):
        print(f"- {key}: {value}")

    canonical_entities = [node for node in merged.nodes if node.label == "CanonicalEntity"]
    canonical_metrics = [node for node in merged.nodes if node.label == "CanonicalMetric"]

    if canonical_entities:
        print("CanonicalEntity 示例:")
        for node in canonical_entities[:10]:
            print_node(node)

    if canonical_metrics:
        print("CanonicalMetric 示例:")
        for node in canonical_metrics[:10]:
            print_node(node)


def test_knowledge_extract(md_path: Path = DEFAULT_MD_PATH) -> None:
    md_path = md_path.expanduser().resolve()
    if not md_path.exists():
        raise FileNotFoundError(f"测试文档不存在: {md_path}")

    text = md_path.read_text(encoding="utf-8", errors="ignore")
    blocks = parse_markdown_blocks(text)

    rule("0. 测试文档")
    print(f"文件: {md_path}")
    print(f"source_id: {SOURCE_ID}")
    print(f"总字符数: {len(text)}")

    explain_blocks(blocks)
    structure_builder = explain_structure(blocks, SOURCE_ID, str(md_path))
    clause_builder = explain_clause_extraction(blocks, SOURCE_ID, str(md_path))
    table_builder = explain_table_extraction(blocks, SOURCE_ID, str(md_path))
    explain_final_graph(structure_builder, table_builder, clause_builder)


if __name__ == "__main__":
    test_knowledge_extract()
