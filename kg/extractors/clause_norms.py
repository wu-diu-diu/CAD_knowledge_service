from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..models import GraphBuilder, MdBlock

CLAUSE_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+(.*)")
VALUE_TOKEN_RE = re.compile(
    r"(?P<raw>\d+(?:\.\d+)?(?:\s*[~\-]\s*\d+(?:\.\d+)?)?(?:/\d+(?:\.\d+)?)?\*?)\s*(?P<unit>lx|m|mm|cm|kW|W/m2|W/㎡|W/m²|%|K|Pa|dB|Ra|GR)?",
    re.I,
)

MODALITY_RULES = [
    ("严禁", "prohibit"),
    ("不得", "prohibit"),
    ("不应", "discourage"),
    ("必须", "must"),
    ("应", "must"),
    ("宜", "recommended"),
    ("可", "optional"),
]

METRIC_KEYWORDS = [
    ("照度标准值", "照度标准值"),
    ("照度", "照度"),
    ("显色指数", "显色指数"),
    ("Ra", "Ra"),
    ("眩光值", "眩光值"),
    ("GR", "GR"),
    ("功率密度", "照明功率密度"),
    ("色温", "色温"),
]

ENTITY_HINT_RE = re.compile(r"([^\s，。；：,:]{1,20}(?:室|厅|间|廊|道|场所|场地|区域|区|建筑|房间|空间))")
CONDITION_RE = re.compile(r"(当[^，。；;]*?时|在[^，。；;]*?(?:内|下|时)|对于[^，。；;]+)")


def _detect_modality(text: str) -> Tuple[Optional[str], Optional[str]]:
    for token, modality in MODALITY_RULES:
        if token in text:
            return token, modality
    return None, None


def _extract_metrics(text: str) -> List[str]:
    names: List[str] = []
    seen = set()
    for keyword, canonical in METRIC_KEYWORDS:
        if keyword in text and canonical not in seen:
            seen.add(canonical)
            names.append(canonical)
    return names


def _extract_entities(text: str, token: Optional[str]) -> List[str]:
    scope = text
    if token and token in text:
        scope = text.split(token, 1)[0]
    candidates: List[str] = []
    for m in ENTITY_HINT_RE.finditer(scope):
        name = m.group(1).strip("（）()、，,。；; ")
        if name and name not in candidates:
            candidates.append(name)
    return candidates


def _extract_conditions(text: str) -> List[str]:
    vals: List[str] = []
    for m in CONDITION_RE.finditer(text):
        val = m.group(1).strip()
        if val and val not in vals:
            vals.append(val)
    return vals


def _extract_value_tokens(text: str) -> List[Tuple[str, Optional[float], Optional[str]]]:
    vals: List[Tuple[str, Optional[float], Optional[str]]] = []
    seen = set()
    for m in VALUE_TOKEN_RE.finditer(text):
        raw = (m.group("raw") or "").strip()
        unit = (m.group("unit") or "").strip() or None
        if not raw:
            continue
        key = (raw, unit or "")
        if key in seen:
            continue
        seen.add(key)
        num = None
        first_num = re.match(r"^\d+(?:\.\d+)?", raw)
        if first_num:
            try:
                num = float(first_num.group(0))
            except ValueError:
                num = None
        vals.append((raw, num, unit))
    return vals


def extract_clause_requirements(blocks: List[MdBlock], source_id: str, source_path: str) -> GraphBuilder:
    g = GraphBuilder()
    for block in blocks:
        if block.block_type != "paragraph":
            continue
        match = CLAUSE_RE.match(block.text)
        if not match:
            continue
        clause_no, clause_text = match.groups()
        token, modality = _detect_modality(clause_text)
        if modality is None:
            # Only extract normative clauses in MVP to reduce noise.
            continue

        clause_id = f"clause:{source_id}:{block.index}"
        req_id = f"clause_req:{source_id}:{block.index}"
        g.add_node(
            req_id,
            "Requirement",
            requirement_type="clause_rule",
            clause_no=clause_no,
            text=clause_text.strip(),
            modality=modality,
            modality_token=token,
            source_id=source_id,
            source_path=source_path,
            page_no=block.page_no,
        )
        g.add_edge("CLAUSE_EXPRESSES_REQUIREMENT", clause_id, req_id)
        g.add_edge("SOURCE_OF", req_id, clause_id)

        for metric_name in _extract_metrics(clause_text):
            metric_id = f"metric:{metric_name}"
            g.add_node(metric_id, "Metric", name=metric_name, source_id=source_id)
            g.add_edge("CONSTRAINS_METRIC", req_id, metric_id)

        for idx, (raw, num, unit) in enumerate(_extract_value_tokens(clause_text), start=1):
            val_id = f"clause_value:{source_id}:{block.index}:{idx}"
            g.add_node(
                val_id,
                "ValueSpec",
                raw_text=raw,
                value=num,
                unit=unit,
                source_id=source_id,
                page_no=block.page_no,
            )
            g.add_edge("HAS_VALUE_SPEC", req_id, val_id)

        for idx, entity_name in enumerate(_extract_entities(clause_text, token), start=1):
            entity_id = f"entity:{entity_name}"
            g.add_node(entity_id, "DomainEntity", name=entity_name, canonical_name=entity_name, source_id=source_id)
            g.add_edge("APPLIES_TO", req_id, entity_id, role="subject", rank=idx)

        for idx, cond_text in enumerate(_extract_conditions(clause_text), start=1):
            cond_id = f"clause_condition:{source_id}:{block.index}:{idx}"
            g.add_node(cond_id, "Condition", text=cond_text, condition_type="clause_context", source_id=source_id, page_no=block.page_no)
            g.add_edge("UNDER_CONDITION", req_id, cond_id)

    return g
