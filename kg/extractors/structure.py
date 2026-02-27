from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

from ..models import GraphBuilder, MdBlock

CLAUSE_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+(.*)")
TABLE_NO_RE = re.compile(r"表\s*([\d.\-]+)")
FIGURE_NO_RE = re.compile(r"(图|Figure|Fig\.)\s*([\d.\-]+)", re.I)


def _evidence_id(source_id: str, block: MdBlock) -> str:
    digest = hashlib.sha1(block.text.encode("utf-8")).hexdigest()[:12]  ## 为每一块文本创建一个唯一的哈希值，作为该块相关信息的载体节点ID的一部分，以便在图中追踪该块的信息来源和内容
    return f"evidence:{source_id}:{block.index}:{digest}"


def extract_structure_graph(blocks: List[MdBlock], source_id: str, source_path: str, source_type: str) -> GraphBuilder:
    """
    从Markdown块列表中提取文档结构信息，构建一个GraphBuilder对象，包含文档、页面、章节、条款、表格和图表等节点，以及它们之间的关系边
    输入：
    - blocks: Markdown块列表，每个块包含文本、类型、页码等信息
    - source_id: 文档唯一标识符（哈希值）
    - source_path: 文档来源路径
    - source_type: 文档来源类型，例如"local_file"、"url"等
    """
    g = GraphBuilder()
    doc_id = f"doc:{source_id}"
    g.add_node(doc_id, "StandardDocument", source_id=source_id, source_path=source_path, source_type=source_type)  ## 添加文档节点

    current_page_id: Optional[str] = None
    section_stack: Dict[int, str] = {}
    current_section_id: Optional[str] = None
    last_clause_id: Optional[str] = None
    table_counter = 0
    figure_counter = 0
    clause_counter = 0

    for block in blocks:
        if block.block_type == "page_marker":
            ## 如果块是页码标记类型，直接使用块的页码信息创建页面节点，并与文档节点建立HAS_PAGE关系；如果当前已有页面节点，则更新current_page_id以便后续块关联到正确的页面
            page_no = block.meta.get("page_no") or block.page_no
            if page_no is None:
                continue
            current_page_id = f"page:{source_id}:{page_no}"
            g.add_node(current_page_id, "Page", page_no=int(page_no), source_id=source_id)
            g.add_edge("HAS_PAGE", doc_id, current_page_id)
            continue

        if block.page_no is not None:  ## 如果块不是page_maker，但是包含页码信息，比如段落，表格等，则也创建页面节点并与文档节点建立HAS_PAGE关系，以确保这些块能够关联到正确的页面
            page_id = f"page:{source_id}:{block.page_no}"
            g.add_node(page_id, "Page", page_no=int(block.page_no), source_id=source_id)
            g.add_edge("HAS_PAGE", doc_id, page_id)
            current_page_id = page_id

        evidence_id = _evidence_id(source_id, block)  ## 用每个块的text内容创建一个唯一的evidence_id，作为该块相关信息的载体节点
        g.add_node(
            evidence_id,
            "EvidenceSpan",
            block_index=block.index,
            block_type=block.block_type,
            page_no=block.page_no,
            text=block.text,
            source_id=source_id,
            source_path=source_path,
        )
        g.add_edge("HAS_EVIDENCE", doc_id, evidence_id)
        if current_page_id:
            g.add_edge("LOCATED_ON_PAGE", evidence_id, current_page_id)

        if block.block_type == "heading":
            level = int(block.meta.get("level", 1))
            title = block.text.strip()
            section_no = block.meta.get("section_no")
            section_id = f"section:{source_id}:{block.index}"
            g.add_node(
                section_id,
                "Section",
                title=title,
                level=level,
                section_no=section_no,
                page_no=block.page_no,
                source_id=source_id,
            )
            g.add_edge("HAS_EVIDENCE", section_id, evidence_id)
            if current_page_id:
                g.add_edge("LOCATED_ON_PAGE", section_id, current_page_id)

            parent_level = max(1, level - 1)
            parent_id = None
            for lv in range(parent_level, 0, -1):
                if lv in section_stack:
                    parent_id = section_stack[lv]
                    break
            if parent_id:
                g.add_edge("HAS_SECTION", parent_id, section_id)
            else:
                g.add_edge("HAS_SECTION", doc_id, section_id)

            for lv in list(section_stack.keys()):
                if lv >= level:
                    section_stack.pop(lv, None)
            section_stack[level] = section_id
            current_section_id = section_id
            continue

        if block.block_type == "paragraph":
            m = CLAUSE_RE.match(block.text)
            if not m:
                continue
            clause_no, clause_text = m.groups()
            clause_id = f"clause:{source_id}:{block.index}"
            clause_counter += 1
            g.add_node(
                clause_id,
                "Clause",
                clause_no=clause_no,
                text=clause_text.strip(),
                page_no=block.page_no,
                source_id=source_id,
            )
            if current_section_id:
                g.add_edge("HAS_CLAUSE", current_section_id, clause_id)
            else:
                g.add_edge("HAS_CLAUSE", doc_id, clause_id)
            g.add_edge("HAS_EVIDENCE", clause_id, evidence_id)
            if current_page_id:
                g.add_edge("LOCATED_ON_PAGE", clause_id, current_page_id)
            if last_clause_id:
                g.add_edge("NEXT_CLAUSE", last_clause_id, clause_id)
            last_clause_id = clause_id
            continue

        if block.block_type == "table":
            table_counter += 1
            caption = (block.meta.get("caption") or "").strip() or None
            table_no = None
            if caption:
                tm = TABLE_NO_RE.search(caption)
                if tm:
                    table_no = tm.group(1)
            table_id = f"table:{source_id}:{block.index}"
            g.add_node(
                table_id,
                "Table",
                table_no=table_no,
                title=caption,
                page_no=block.page_no,
                row_count=len(block.meta.get("table", {}).get("rows", [])),
                source_id=source_id,
            )
            if current_section_id:
                g.add_edge("HAS_TABLE", current_section_id, table_id)
            else:
                g.add_edge("HAS_TABLE", doc_id, table_id)
            g.add_edge("HAS_EVIDENCE", table_id, evidence_id)
            if current_page_id:
                g.add_edge("LOCATED_ON_PAGE", table_id, current_page_id)

            table_data = block.meta.get("table", {})
            headers = table_data.get("headers", [])
            for row_idx, row in enumerate(table_data.get("rows", []), start=1):
                row_id = f"table_row:{source_id}:{block.index}:{row_idx}"
                g.add_node(
                    row_id,
                    "TableRow",
                    row_index=row_idx,
                    values=row,
                    headers=headers,
                    page_no=block.page_no,
                    source_id=source_id,
                )
                g.add_edge("HAS_ROW", table_id, row_id)
                if current_page_id:
                    g.add_edge("LOCATED_ON_PAGE", row_id, current_page_id)
            continue

        if block.block_type == "figure_json":
            figure_counter += 1
            payload = block.meta.get("json") or {}
            caption = payload.get("figure_title")
            figure_no = None
            if isinstance(caption, str):
                fm = FIGURE_NO_RE.search(caption)
                if fm:
                    figure_no = fm.group(2)
            figure_id = f"figure:{source_id}:{block.index}"
            g.add_node(
                figure_id,
                "Figure",
                figure_no=figure_no,
                caption=caption,
                page_no=block.page_no,
                source_id=source_id,
            )
            if current_section_id:
                g.add_edge("HAS_FIGURE", current_section_id, figure_id)
            else:
                g.add_edge("HAS_FIGURE", doc_id, figure_id)
            g.add_edge("HAS_EVIDENCE", figure_id, evidence_id)
            if current_page_id:
                g.add_edge("LOCATED_ON_PAGE", figure_id, current_page_id)

            insight_id = f"figure_insight:{source_id}:{block.index}"
            g.add_node(
                insight_id,
                "FigureInsight",
                purpose=payload.get("purpose"),
                role_in_paper=payload.get("role_in_paper"),
                visual_elements_summary=payload.get("visual_elements_summary"),
                page_no=block.page_no,
            )
            g.add_edge("HAS_INSIGHT", figure_id, insight_id)

            for idx, insight in enumerate(payload.get("key_insights", []) or [], start=1):
                key_id = f"key_insight:{source_id}:{block.index}:{idx}"
                g.add_node(key_id, "KeyInsight", text=str(insight), rank=idx, page_no=block.page_no)
                g.add_edge("HAS_KEY_INSIGHT", insight_id, key_id)

    g.add_node(
        doc_id,
        "StandardDocument",
        source_id=source_id,
        source_path=source_path,
        source_type=source_type,
        block_count=len(blocks),
        clause_count=clause_counter,
        table_count=table_counter,
        figure_count=figure_counter,
    )
    return g
