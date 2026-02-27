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


def extract_table_requirements(blocks: List[MdBlock], source_id: str, source_path: str) -> GraphBuilder:
    g = GraphBuilder()
    for block in blocks:
        if block.block_type != "table":
            continue
        table = block.meta.get("table") or {}
        headers = _normalize_headers(table.get("headers") or [])
        rows = table.get("rows") or []
        if not headers or not rows:
            continue

        table_id = f"table:{source_id}:{block.index}"
        caption = block.meta.get("caption")
        page_no = block.page_no

        for row_idx, row in enumerate(rows, start=1):
            if len(row) < len(headers):
                row = row + [""] * (len(headers) - len(row))
            row_id = f"table_row:{source_id}:{block.index}:{row_idx}"
            entity_name = _pick_entity_name(row)
            entity_id = None
            if entity_name:
                entity_id = f"entity:{entity_name}"
                g.add_node(entity_id, "DomainEntity", name=entity_name, canonical_name=entity_name, source_id=source_id)

            for col_idx, (header, cell) in enumerate(zip(headers, row), start=1):
                cell_text = (cell or "").strip()
                if not _looks_value_cell(cell_text):
                    continue
                metric_name = header
                metric_id = f"metric:{metric_name}"
                g.add_node(metric_id, "Metric", name=metric_name, source_id=source_id)

                req_id = f"req:{source_id}:{block.index}:{row_idx}:{col_idx}"
                val_id = f"value:{source_id}:{block.index}:{row_idx}:{col_idx}"
                g.add_node(
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
                g.add_node(
                    val_id,
                    "ValueSpec",
                    raw_text=cell_text,
                    value=_parse_numeric_prefix(cell_text),
                    unit="lx" if ("照度" in metric_name or "照度" in (caption or "")) else None,
                    page_no=page_no,
                    source_id=source_id,
                )

                g.add_edge("ROW_EXPRESSES_REQUIREMENT", row_id, req_id)
                g.add_edge("CONSTRAINS_METRIC", req_id, metric_id)
                g.add_edge("HAS_VALUE_SPEC", req_id, val_id)
                g.add_edge("SOURCE_OF", req_id, table_id)
                if entity_id:
                    g.add_edge("APPLIES_TO", req_id, entity_id)

            if entity_name and len(row) >= 2:
                # Create lightweight condition node for the common "参考平面及其高度" column when present.
                for header, cell in zip(headers, row):
                    if "参考平面" not in header:
                        continue
                    cond_text = (cell or "").strip()
                    if not cond_text:
                        continue
                    cond_id = f"condition:{source_id}:{block.index}:{row_idx}:reference_plane"
                    req_scope_id = f"table_row_scope:{source_id}:{block.index}:{row_idx}"
                    g.add_node(req_scope_id, "Requirement", requirement_type="row_context", source_id=source_id, page_no=page_no)
                    g.add_node(cond_id, "Condition", condition_type="reference_plane", text=cond_text, source_id=source_id, page_no=page_no)
                    g.add_edge("ROW_EXPRESSES_REQUIREMENT", row_id, req_scope_id)
                    g.add_edge("UNDER_CONDITION", req_scope_id, cond_id)
                    if entity_id:
                        g.add_edge("APPLIES_TO", req_scope_id, entity_id)
                    g.add_edge("SOURCE_OF", req_scope_id, table_id)
                    break

    return g
