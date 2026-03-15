from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..models import GraphBuilder, MdBlock

NUMERIC_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*$")
VALUE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)")


def _normalize_headers(headers: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    normalized: List[str] = []
    for idx, header in enumerate(headers):
        name = (header or "").strip()
        if not name:
            name = f"列{idx + 1}"
        count = seen.get(name, 0) + 1
        seen[name] = count
        normalized.append(name if count == 1 else f"{name}#{count}")
    return normalized


def _looks_value_cell(text: str) -> bool:
    t = text.strip()
    if not t or t in {"-", "—", "/"}:
        return False
    if any(ch.isdigit() for ch in t):
        return True
    return False


def _parse_numeric_prefix(text: str) -> Optional[float]:
    m = VALUE_RE.match(text.strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _pick_entity_name(row: List[str]) -> Optional[str]:
    for cell in row:
        t = cell.strip()
        if not t:
            continue
        if _looks_value_cell(t):
            continue
        return t
    return None


def _base_header_name(header: str) -> str:
    return re.sub(r"#\d+$", "", (header or "").strip())


def _build_section_context(blocks: List[MdBlock], source_id: str) -> Dict[int, Optional[str]]:
    current_section_id: Optional[str] = None
    mapping: Dict[int, Optional[str]] = {}
    for block in blocks:
        if block.block_type == "heading":
            current_section_id = f"section:{source_id}:{block.index}"
        mapping[block.index] = current_section_id
    return mapping


def _attach_context_edges(
    builder: GraphBuilder,
    node_id: str,
    *,
    source_id: str,
    page_no: Optional[int],
    section_id: Optional[str],
) -> None:
    if page_no is not None:
        builder.add_edge("LOCATED_ON_PAGE", node_id, f"page:{source_id}:{page_no}")
    if section_id:
        builder.add_edge("UNDER_SECTION", node_id, section_id)


def extract_table_requirements(blocks: List[MdBlock], source_id: str, source_path: str) -> GraphBuilder:
    """
    输入：
    - blocks: Markdown 解析后的块列表，只会对其中 block_type == "table" 的块做处理。
    - source_id: 当前源文档的唯一标识，用于拼接节点 ID。
    - source_path: 当前源文档路径，写入 Requirement 节点属性，便于追溯来源。

    输出：
    - GraphBuilder: 一个只包含“表格规范要求”相关节点和边的图构建器。
      当前采用的表格语义是：
      1. 第一列无论表头是什么，都视为主对象 DomainEntity
      2. 第二列若表头与第一列表头相同，则视为主对象下的子对象/子场景列；
         但只有当第二列内容与第一列主对象内容不相同时，才实际生成 Condition
      3. 后续各属性列各自生成一条 Requirement，每条 Requirement 对应一个 Metric 和一个 ValueSpec
    """
    g = GraphBuilder()  ## 初始化一个空图，用于承接所有从表格中抽出的节点和关系
    section_context = _build_section_context(blocks, source_id)  ## 记录每个块所在的最近标题，用于给语义节点补 section 上下文
    for block in blocks:  ## 遍历所有 Markdown 块
        if block.block_type != "table":  ## 只处理表格块，其他块直接跳过
            continue
        table = block.meta.get("table") or {}  ## 取出表格的结构化内容，通常包含 headers 和 rows
        headers = _normalize_headers(table.get("headers") or [])  ## 归一化表头，处理空表头和重名表头
        rows = table.get("rows") or []  ## 取出表格的数据行
        if not headers or not rows:  ## 如果没有表头或没有数据行，就无法抽取“对象-指标-值”结构，直接跳过
            continue

        table_id = f"table:{source_id}:{block.index}"  ## 构造当前表格节点 ID，与结构图中的 Table 节点 ID 保持一致
        caption = block.meta.get("caption")  ## 表题，后面会写入 Requirement 属性，帮助理解该要求来自哪个表
        page_no = block.page_no  ## 当前表格所在页码
        section_id = section_context.get(block.index)  ## 当前表格所在的最近标题节点
        base_headers = [_base_header_name(header) for header in headers]  ## 去掉重复表头后缀，便于判断列语义
        has_subentity_column = len(base_headers) >= 2 and base_headers[0] == base_headers[1]  ## 第二列表头若与第一列相同，则视作子对象/子场景列

        for row_idx, row in enumerate(rows, start=1):  ## 逐行处理表格数据，行索引从 1 开始
            if len(row) < len(headers):  ## 若当前行列数少于表头列数，则补空字符串，保证后续 zip 不丢列
                row = row + [""] * (len(headers) - len(row))
            row_id = f"table_row:{source_id}:{block.index}:{row_idx}"  ## 构造当前行的节点 ID
            g.add_node(row_id, "TableRow", row_index=row_idx, values=row, headers=headers, page_no=page_no, source_id=source_id)  ## 显式创建 TableRow 节点，便于后续把该行所有 requirement 都挂到一行上
            _attach_context_edges(g, row_id, source_id=source_id, page_no=page_no, section_id=section_id)

            entity_name = (row[0] or "").strip() if row else ""  ## 第一列固定视为主对象文本，不再要求表头必须是“房间或场所”
            entity_id = None  ## 默认当前行未识别出主体对象
            if entity_name:  ## 如果第一列非空，则创建或复用主对象节点
                entity_id = f"entity:{entity_name}"
                g.add_node(entity_id, "DomainEntity", name=entity_name, canonical_name=entity_name, source_id=source_id)
                _attach_context_edges(g, entity_id, source_id=source_id, page_no=page_no, section_id=section_id)

            subentity_text = ""  ## 第二列若存在同名“房间或场所”列，则视作子对象/子场景文本
            condition_id = None
            if has_subentity_column and len(row) >= 2:
                subentity_text = (row[1] or "").strip()
                if subentity_text and subentity_text != entity_name:  ## 只有子对象文本与主对象不相同，才认为存在有效子场景
                    condition_id = f"condition:{source_id}:{block.index}:{row_idx}:subentity"
                    g.add_node(
                        condition_id,
                        "Condition",
                        condition_type="subspace_or_activity",
                        text=subentity_text,
                        source_id=source_id,
                        page_no=page_no,
                    )
                    _attach_context_edges(g, condition_id, source_id=source_id, page_no=page_no, section_id=section_id)

            attr_start_idx = 2 if has_subentity_column else 1  ## 属性列的起始位置：若第二列与第一列表头同名，则从第三列开始；否则从第二列开始
            for col_idx, (header, cell) in enumerate(zip(headers, row), start=1):  ## 逐列遍历“表头-单元格”对
                if col_idx <= attr_start_idx:  ## 跳过主对象列，以及可选的子对象列
                    continue
                cell_text = (cell or "").strip()  ## 清洗单元格文本，去除前后空白
                if not cell_text:  ## 属性值为空则跳过
                    continue
                metric_name = header  ## 把当前列表头视为被约束的指标名称
                metric_id = f"metric:{metric_name}"  ## 构造指标节点 ID
                g.add_node(metric_id, "Metric", name=metric_name, source_id=source_id)  ## 创建或复用指标节点
                _attach_context_edges(g, metric_id, source_id=source_id, page_no=page_no, section_id=section_id)

                req_id = f"req:{source_id}:{block.index}:{row_idx}:{col_idx}"  ## 为当前“行-列”单元格构造 Requirement 节点 ID
                val_id = f"value:{source_id}:{block.index}:{row_idx}:{col_idx}"  ## 为当前单元格值构造 ValueSpec 节点 ID
                g.add_node(  ## 创建 Requirement 节点，表示“这一格表达了一条要求”
                    req_id,
                    "Requirement",
                    source_id=source_id,
                    source_path=source_path,
                    page_no=page_no,
                    table_caption=caption,
                    row_index=row_idx,
                    col_index=col_idx,
                    requirement_type="table_cell_constraint",
                    raw_cell=cell_text,
                )
                _attach_context_edges(g, req_id, source_id=source_id, page_no=page_no, section_id=section_id)
                g.add_node(  ## 创建 ValueSpec 节点，保存当前单元格的原始值、解析出的数值前缀及单位
                    val_id,
                    "ValueSpec",
                    raw_text=cell_text,
                    value=_parse_numeric_prefix(cell_text),
                    unit="lx" if ("照度" in metric_name or "照度" in (caption or "")) else None,
                    page_no=page_no,
                    source_id=source_id,
                )
                _attach_context_edges(g, val_id, source_id=source_id, page_no=page_no, section_id=section_id)

                g.add_edge("ROW_EXPRESSES_REQUIREMENT", row_id, req_id)  ## 表示当前表格行表达了这条 Requirement
                g.add_edge("CONSTRAINS_METRIC", req_id, metric_id)  ## 表示这条 Requirement 约束的是哪个指标
                g.add_edge("HAS_VALUE_SPEC", req_id, val_id)  ## 表示这条 Requirement 对应的取值描述是什么
                g.add_edge("SOURCE_OF", req_id, table_id)  ## 表示这条 Requirement 来源于哪一个表
                if entity_id:  ## 如果当前行识别出了主体对象
                    g.add_edge("APPLIES_TO", req_id, entity_id)  ## 则把这条 Requirement 连到适用对象上
                if condition_id:  ## 如果当前行识别出了子对象/子场景，则把每条属性 requirement 都挂到该条件下
                    g.add_edge("UNDER_CONDITION", req_id, condition_id)

    return g  ## 返回从所有表格块中抽取出的规范要求图
