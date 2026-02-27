from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .models import MdBlock

PAGE_RE = re.compile(r"^\[PAGE:(\d+)\]\s*$")  ## 匹配[PAGE:5]这种格式的文本
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
TABLE_CAPTION_RE = re.compile(r"^\s*(表\s*\d[\d.\-]*\s+.*|Table\s+\d[\d.\-]*\s+.*)\s*$", re.I)
CLAUSE_NO_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s*")


def _split_pipe_row(line: str) -> List[str]:
    raw = line.strip()
    if raw.startswith("|"):
        raw = raw[1:]
    if raw.endswith("|"):
        raw = raw[:-1]
    cells = [cell.strip() for cell in raw.split("|")]
    return cells


def _is_delimiter_row(cells: List[str]) -> bool:
    if not cells:
        return False
    for cell in cells:
        compact = cell.replace(":", "").replace("-", "").replace(" ", "")
        if compact:
            if compact != "":
                return False
    return True


def parse_markdown_table(table_text: str) -> Dict[str, Any]:
    lines = [ln.rstrip() for ln in table_text.splitlines() if ln.strip()]
    pipe_lines = [ln for ln in lines if ln.lstrip().startswith("|")]
    if len(pipe_lines) < 2:
        return {"headers": [], "rows": []}

    rows: List[List[str]] = [_split_pipe_row(ln) for ln in pipe_lines]
    rows = [row for row in rows if row]
    if not rows:
        return {"headers": [], "rows": []}

    headers = rows[0]
    body: List[List[str]] = []
    for row in rows[1:]:
        if _is_delimiter_row(row):
            continue
        body.append(row)

    width = max([len(headers)] + [len(r) for r in body]) if headers else 0
    if width:
        headers = headers + [""] * (width - len(headers))
        body = [r + [""] * (width - len(r)) for r in body]

    deduped_body: List[List[str]] = []
    for row in body:
        if row == headers:
            continue
        deduped_body.append(row)

    return {"headers": headers, "rows": deduped_body}


def _try_parse_json_block(text: str) -> Optional[dict]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_markdown_blocks(md_text: str) -> List[MdBlock]:
    """

    函数功能：将Markdown文本解析为一个个结构化的块（MdBlock），每个块包含类型、文本内容、页码和额外的元信息。
    解析逻辑：
    1. 统一换行符，避免不同平台的换行差异导致的问题。
    2. 根据换行符分割文本为行。
    3. 逐行处理文本，根据不同的模式识别不同类型的块：
    - page_marker：匹配[PAGE:5]这类文本，提取页码并创建一个page_marker类型的块。
    - figure_json：匹配以```json开头的代码块，尝试解析其中的内容为JSON，如果成功且包含特定字段则标记为figure_json类型，否则标记为json_block类型。
    - heading：匹配Markdown中的标题行，提取标题级别和文本内容，并尝试从标题文本中提取章节编号，如果存在的话。
    - table：匹配以|开头的表格行，连续的以|开头的行被视为同一个表格，提取表格文本并尝试从前面的段落中提取表格标题，如果存在的话。   
    - paragraph：处理普通段落，连续的非空行被视为同一个段落，提取段落文本。
    4. 对于每个识别出的块，创建一个MdBlock对象，包含块的类型、文本内容、页码和额外的元信息，并将其添加到块列表中。
    5. 返回解析得到的块列表。  

    """
    ## 统一换行符，避免不同平台的换行差异导致的问题
    text = md_text.replace("\r\n", "\n").replace("\r", "\n")
    ## 根据换行符分割文本为行
    lines = text.split("\n")

    blocks: List[MdBlock] = []
    page_no: Optional[int] = None
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        ## 处理[PAGE:5]这类文本，提取页码并创建一个page_marker类型的块
        page_match = PAGE_RE.match(stripped)
        if page_match:
            page_no = int(page_match.group(1))
            blocks.append(MdBlock(index=len(blocks), block_type="page_marker", text=stripped, page_no=page_no, meta={"page_no": page_no}))
            i += 1
            continue

        if not stripped:
            i += 1
            continue
        ## 处理以```json开头的代码块，尝试解析其中的内容为JSON，如果成功且包含特定字段则标记为figure_json类型，否则标记为json_block类型
        if stripped.startswith("```json"):
            j = i + 1
            payload_lines: List[str] = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                payload_lines.append(lines[j])
                j += 1
            payload_text = "\n".join(payload_lines).strip()
            payload = _try_parse_json_block(payload_text)
            if payload and any(k in payload for k in ("purpose", "visual_elements_summary", "figure_title", "role_in_paper")):
                blocks.append(
                    MdBlock(
                        index=len(blocks),
                        block_type="figure_json",
                        text=payload_text,
                        page_no=page_no,
                        meta={"json": payload},
                    )
                )
            else:
                blocks.append(MdBlock(index=len(blocks), block_type="json_block", text=payload_text, page_no=page_no, meta={"json": payload}))
            i = j + 1 if j < len(lines) else len(lines)
            continue
        ## 处理Markdown中的标题行，提取标题级别和文本内容，并尝试从标题文本中提取章节编号，如果存在的话
        heading_match = HEADING_RE.match(line)
        if heading_match:
            marks, title = heading_match.groups()
            level = len(marks)  ## 标题层级由#的数量决定，1-6级
            section_no = None ## 尝试从标题文本中提取章节编号，匹配类似"1.2.3 "开头的格式
            clause_match = CLAUSE_NO_RE.match(title)  ## 匹配类似"1.2.3 "开头的格式，提取章节编号
            if clause_match:
                section_no = clause_match.group(1)  ## 提取章节编号，例如"1.2.3"
            blocks.append(
                MdBlock(
                    index=len(blocks),
                    block_type="heading",
                    text=title.strip(),
                    page_no=page_no,
                    meta={"level": level, "section_no": section_no},
                )
            )
            i += 1
            continue
        ## 处理以|开头的表格行，连续的以|开头的行被视为同一个表格，提取表格文本并尝试从前面的段落中提取表格标题，如果存在的话
        if line.lstrip().startswith("|"):
            table_lines: List[str] = []
            j = i
            while j < len(lines) and lines[j].strip():
                if not lines[j].lstrip().startswith("|"):
                    break
                table_lines.append(lines[j])
                j += 1
            table_text = "\n".join(table_lines)
            parsed = parse_markdown_table(table_text)
            caption = None  ## 表格标题，尝试从前一个块（如果存在且是段落）中提取符合表格标题格式的文本
            if blocks and blocks[-1].block_type == "paragraph":
                candidate = blocks[-1].text.strip()
                if TABLE_CAPTION_RE.match(candidate):
                    caption = candidate
                    blocks.pop()
            blocks.append(
                MdBlock(
                    index=len(blocks),
                    block_type="table",
                    text=(caption + "\n" if caption else "") + table_text,
                    page_no=page_no,
                    meta={"caption": caption, "table": parsed},
                )
            )
            i = j
            continue
        ## 处理普通段落，连续的非空行被视为同一个段落，提取段落文本
        para_lines = [line]
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            nxt_stripped = nxt.strip()
            if not nxt_stripped:
                break
            if PAGE_RE.match(nxt_stripped):
                break
            if nxt_stripped.startswith("```json") or HEADING_RE.match(nxt) or nxt.lstrip().startswith("|"):
                break
            para_lines.append(nxt)
            j += 1
        para_text = "\n".join(para_lines).strip()
        blocks.append(MdBlock(index=len(blocks), block_type="paragraph", text=para_text, page_no=page_no, meta={}))
        i = j

    return blocks
