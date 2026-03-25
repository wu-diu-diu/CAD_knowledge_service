from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from ..models import GraphBuilder, MdBlock

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM fallback client (lazy-initialised, shared across calls)
# ---------------------------------------------------------------------------

_llm_client = None


def _get_llm_client():
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("缺少 openai 依赖，请安装 `openai`。") from e
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip() or "https://api.deepseek.com/v1"
    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY 环境变量，无法使用 LLM fallback。")
    _llm_client = OpenAI(api_key=api_key, base_url=base_url)
    return _llm_client


_LLM_SYSTEM = (
    "你是建筑照明规范专家。从给定的规范条款文本中抽取结构化信息，"
    "严格按照 JSON 格式输出，不要输出任何其他内容。"
)

_LLM_PROMPT_TMPL = """\
请从以下规范条款中抽取所有"实体-指标-值"三元组。

条款文本：
{text}

输出格式（JSON 数组，每个元素对应一条要求）：
[
  {{
    "entity": "适用对象，如办公室、走廊，没有则为null",
    "metric": "被约束的指标名称，如照度标准值、维护系数、色容差",
    "value": "数值或范围字符串，如300、0.7、300~500，没有则为null",
    "unit": "单位，如lx、%、SDCM，没有则为null",
    "modality": "must/recommended/optional/prohibit/discourage 之一，没有则为null",
    "condition": "适用条件，如当天然采光不足时，没有则为null"
  }}
]

注意：
- entity 只写空间/场所名称，不写动词或修饰语
- metric 写指标的规范名称，不要包含数值
- 如果一句话包含多个指标，拆成多条
- 只输出 JSON，不要解释"""


def _llm_extract(text: str) -> List[Dict]:
    """调用 LLM 从条款文本中抽取实体-指标-值三元组，失败时返回空列表。"""
    try:
        client = _get_llm_client()
    except RuntimeError as e:
        logger.debug("LLM client unavailable: %s", e)
        return []
    try:
        resp = client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": _LLM_PROMPT_TMPL.format(text=text)},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        raw = resp.choices[0].message.content or ""
        # 去掉可能的 markdown 代码块包裹
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        logger.warning("LLM fallback 抽取失败: %s", e)
        return []

CLAUSE_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+(.*)")
LIST_ITEM_RE = re.compile(r"^\s*[-•]?\s*(\d+)[、.\s]+\s*(.*\S)\s*$")
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

CONTROL_METHOD_KEYWORDS = [
    ("感应调光控制", "感应调光控制"),
    ("自动感应控制", "自动感应控制"),
    ("时钟控制", "时钟控制"),
    ("场景控制", "场景控制"),
    ("分区或群组控制", "分区/群组控制"),
    ("分区控制", "分区控制"),
    ("群组控制", "群组控制"),
    ("单灯或分组控制", "单灯/分组控制"),
    ("单灯控制", "单灯控制"),
    ("分组控制", "分组控制"),
    ("顺序控制", "顺序控制"),
    ("语音控制", "语音控制"),
    ("协同控制", "协同控制"),
    ("自动调节照度", "自动调节照度"),
    ("自动开关", "自动开关"),
    ("集中控制", "集中控制"),
    ("调光", "调光"),
]

ENTITY_HINT_RE = re.compile(
    r"([^\s，。；：,:]{1,30}(?:室|厅|间|廊|道|场所|场地|区域|区|建筑|房间|空间|楼|馆|库|车库|厕所|厨房|病房|门厅|走廊|楼梯间|电梯厅|餐厅|报告厅|教室|阅览室|办公室))"
)
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


def _extract_control_methods(text: str) -> List[str]:
    methods: List[str] = []
    seen = set()
    for keyword, canonical in CONTROL_METHOD_KEYWORDS:
        if keyword in text and canonical not in seen:
            seen.add(canonical)
            methods.append(canonical)
    return methods


def _split_numbered_items(text: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    current_no: Optional[str] = None
    current_parts: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = LIST_ITEM_RE.match(stripped)
        if match:
            if current_no is not None:
                item_text = " ".join(part for part in current_parts if part).strip()
                if item_text:
                    items.append((current_no, item_text))
            current_no = match.group(1)
            current_parts = [match.group(2).strip()]
            continue
        if current_no is not None:
            current_parts.append(stripped)

    if current_no is not None:
        item_text = " ".join(part for part in current_parts if part).strip()
        if item_text:
            items.append((current_no, item_text))

    return items


def _has_open_continuation(text: str) -> bool:
    compact = (text or "").rstrip()
    if not compact:
        return False
    return compact[-1] not in "。；;：:!！?？"


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


def _dedupe_texts(values: List[str]) -> List[str]:
    deduped: List[str] = []
    for value in values:
        compact = (value or "").strip()
        if compact and compact not in deduped:
            deduped.append(compact)
    return deduped


def _create_requirement(
    builder: GraphBuilder,
    *,
    source_id: str,
    source_path: str,
    clause_id: str,
    clause_block_index: int,
    clause_no: str,
    text: str,
    modality_token: Optional[str],
    modality: Optional[str],
    page_no: Optional[int],
    section_id: Optional[str],
    requirement_type: str,
    item_no: Optional[str] = None,
    parent_context: Optional[str] = None,
) -> None:
    """
    将一条条款文本或一个条款分项落成图谱中的 Requirement 及其语义节点。

    输入：
    - builder: 当前图构建器，负责新增节点和关系。
    - source_id/source_path: 文档来源信息，用于节点 ID 拼接与结果追溯。
    - clause_id: 当前 Requirement 所归属的 Clause 节点 ID。
    - clause_block_index: 原始 Markdown block 序号，用于生成稳定的 Requirement / Value / Condition 节点 ID。
    - clause_no: 条款号，例如 `7.3.6`。
    - text: 当前要抽取的条款正文；如果是分项，则为单个分项文本。
    - modality_token/modality: 规范模态词及其归类结果，例如 `宜 -> recommended`。
    - page_no/section_id: 上下文定位信息，用于把语义节点关联到页码和章节。
    - requirement_type: Requirement 类型，例如 `clause_rule` 或 `clause_item_rule`。
    - item_no: 分项编号；非分项条文为空。
    - parent_context: 主条文上下文。对于编号分项，会把父句语义一起并入抽取。

    输出：
    - 无返回值。函数会直接向图中写入：
      1. 一条 Requirement 节点；
      2. 与 Requirement 相连的 DomainEntity / Metric / ValueSpec / Condition / ControlMethod 节点；
      3. Requirement 与 Clause、页码、章节之间的结构关系。
    """
    if modality is None:
        # 没有规范模态词时，不认为它构成可执行/可约束的 requirement，直接跳过。
        return

    # 为当前 requirement 构造稳定 ID：
    # - 普通条文使用 `base`
    # - 编号分项使用其 item_no
    req_suffix = item_no if item_no is not None else "base"
    req_id = f"clause_req:{source_id}:{clause_block_index}:{req_suffix}"

    # 第一步：先把当前条款或分项本身落成 Requirement 节点。
    # 这里保留原文、条款号、模态词、父条文上下文等信息，
    # 便于后续查询和答案追溯。
    builder.add_node(
        req_id,
        "Requirement",
        requirement_type=requirement_type,
        clause_no=clause_no,
        item_no=item_no,
        text=text.strip(),
        parent_context=parent_context.strip() if parent_context else None,
        modality=modality,
        modality_token=modality_token,
        source_id=source_id,
        source_path=source_path,
        page_no=page_no,
    )

    # 第二步：建立 Clause <-> Requirement 的主干关系。
    # - CLAUSE_EXPRESSES_REQUIREMENT: 这个条款表达了哪条 requirement
    # - SOURCE_OF: 反向保留"该 requirement 来自哪个 clause"的可追溯关系
    builder.add_edge("CLAUSE_EXPRESSES_REQUIREMENT", clause_id, req_id)
    builder.add_edge("SOURCE_OF", req_id, clause_id)

    # 第三步：把 Requirement 挂到页码和章节上下文下，
    # 便于后续按页码、章节范围做检索与回答。
    _attach_context_edges(builder, req_id, source_id=source_id, page_no=page_no, section_id=section_id)

    # 对于分项条文，很多对象/条件/控制方式写在主条文里，
    # 而不完整写在分项文本里，所以这里把父句和子句拼起来，
    # 作为补充抽取的上下文。
    combined_text = f"{parent_context or ''} {text}".strip()

    # 第四步：抽取适用对象（DomainEntity）。
    # 优先从当前文本里找；如果当前分项里没写明对象，
    # 再退回到"父条文 + 当前分项"的合并上下文里找。
    entity_names = _extract_entities(text, modality_token)
    if not entity_names:
        entity_names = _extract_entities(combined_text, modality_token)

    # 第五步：抽取该 requirement 约束的指标（Metric），
    # 例如照度、Ra、功率密度等。
    # 同样采用"当前文本优先，不足时用父句补充"的策略。
    metric_names = _extract_metrics(text)
    if not metric_names:
        metric_names = _extract_metrics(combined_text)

    # LLM fallback：规则未能抽到实体或指标时，调用 LLM 补全。
    # LLM 结果只用于补充缺失部分，不覆盖规则已抽到的内容。
    llm_triples: List[Dict] = []
    if not entity_names or not metric_names:
        llm_triples = _llm_extract(combined_text)
        if llm_triples:
            logger.debug("LLM fallback triggered, clause: %s, triples: %d", clause_no, len(llm_triples))

    # 若规则实体为空，从 LLM 结果里补充
    if not entity_names and llm_triples:
        seen: set = set()
        for triple in llm_triples:
            name = (triple.get("entity") or "").strip()
            if name and name not in seen:
                seen.add(name)
                entity_names.append(name)

    for idx, entity_name in enumerate(entity_names, start=1):
        entity_id = f"entity:{entity_name}"
        builder.add_node(entity_id, "DomainEntity", name=entity_name, canonical_name=entity_name, source_id=source_id)
        builder.add_edge("APPLIES_TO", req_id, entity_id, role="subject", rank=idx)
        _attach_context_edges(builder, entity_id, source_id=source_id, page_no=page_no, section_id=section_id)

    # 若规则指标为空，从 LLM 结果里补充；同时收集 LLM 抽到的值用于后续 ValueSpec
    llm_value_triples: List[Dict] = []  # 记录 LLM 补充的指标对应的值信息
    if not metric_names and llm_triples:
        seen_metrics: set = set()
        for triple in llm_triples:
            mname = (triple.get("metric") or "").strip()
            if mname and mname not in seen_metrics:
                seen_metrics.add(mname)
                metric_names.append(mname)
                llm_value_triples.append(triple)

    for metric_name in metric_names:
        metric_id = f"metric:{metric_name}"
        builder.add_node(metric_id, "Metric", name=metric_name, source_id=source_id)
        builder.add_edge("CONSTRAINS_METRIC", req_id, metric_id)
        _attach_context_edges(builder, metric_id, source_id=source_id, page_no=page_no, section_id=section_id)

    # 第六步：抽取值（ValueSpec）。
    # 优先用规则正则抽取；若规则未抽到值且 LLM 补充了指标，则用 LLM 的值信息。
    rule_values = _extract_value_tokens(text)
    if rule_values:
        for idx, (raw, num, unit) in enumerate(rule_values, start=1):
            val_id = f"clause_value:{source_id}:{clause_block_index}:{req_suffix}:{idx}"
            builder.add_node(
                val_id,
                "ValueSpec",
                raw_text=raw,
                value=num,
                unit=unit,
                source_id=source_id,
                page_no=page_no,
            )
            builder.add_edge("HAS_VALUE_SPEC", req_id, val_id)
            _attach_context_edges(builder, val_id, source_id=source_id, page_no=page_no, section_id=section_id)
    elif llm_value_triples:
        for idx, triple in enumerate(llm_value_triples, start=1):
            raw = (triple.get("value") or "").strip()
            if not raw:
                continue
            unit = (triple.get("unit") or "").strip() or None
            try:
                num: Optional[float] = float(re.match(r"[\d.]+", raw).group())  # type: ignore[union-attr]
            except (AttributeError, ValueError):
                num = None
            val_id = f"clause_value:{source_id}:{clause_block_index}:{req_suffix}:llm{idx}"
            builder.add_node(
                val_id,
                "ValueSpec",
                raw_text=raw,
                value=num,
                unit=unit,
                source_id=source_id,
                page_no=page_no,
            )
            builder.add_edge("HAS_VALUE_SPEC", req_id, val_id)
            _attach_context_edges(builder, val_id, source_id=source_id, page_no=page_no, section_id=section_id)

    # 第七步：抽取条件（Condition）。
    # 条件既可能在父条文里，也可能在当前分项里，所以两边都抽，
    # 再去重后逐条挂到 Requirement 下。
    condition_texts = _dedupe_texts(_extract_conditions(parent_context or "") + _extract_conditions(text))
    for idx, cond_text in enumerate(condition_texts, start=1):
        cond_id = f"clause_condition:{source_id}:{clause_block_index}:{req_suffix}:{idx}"
        builder.add_node(
            cond_id,
            "Condition",
            text=cond_text,
            condition_type="clause_context",
            source_id=source_id,
            page_no=page_no,
        )
        builder.add_edge("UNDER_CONDITION", req_id, cond_id)
        _attach_context_edges(builder, cond_id, source_id=source_id, page_no=page_no, section_id=section_id)

    # 第八步：抽取控制方式（ControlMethod）。
    # 这一步主要服务于"时钟控制 / 场景控制 / 分组控制 / 语音控制"等
    # 非数值型文本知识，使其不只是停留在 Requirement 原文里，
    # 而是成为可查询的独立语义节点。
    for method_name in _extract_control_methods(combined_text):
        method_id = f"control_method:{method_name}"
        builder.add_node(method_id, "ControlMethod", name=method_name, source_id=source_id)
        builder.add_edge("USES_CONTROL_METHOD", req_id, method_id)
        _attach_context_edges(builder, method_id, source_id=source_id, page_no=page_no, section_id=section_id)


def extract_clause_requirements(blocks: List[MdBlock], source_id: str, source_path: str) -> GraphBuilder:
    """
    输入：
    - blocks: Markdown 解析后的块列表，只处理其中 block_type == "paragraph" 的块。
    - source_id: 当前文档的唯一标识，用于拼接 Clause / Requirement / Condition 等节点 ID。
    - source_path: 当前文档来源路径，写入 Requirement 属性以支持结果追溯。

    输出：
    - GraphBuilder: 一个只包含"条文语义抽取结果"的图构建器。

    当前抽取逻辑分两类：
    1. 普通规范条文：
       - 识别"条款号 + 正文"
       - 提取一条 Requirement
       - 再抽取其中的 DomainEntity / Condition / Metric / ValueSpec / ControlMethod
    2. 主条文 + 编号分项条文：
       - 例如"7.3.7 ……宜采用下列措施："
       - 先识别主条文
       - 再向后扫描后续 paragraph 中的 `- 1 ... / - 2 ...` 分项
       - 每个分项单独生成一条 Requirement，并继承主条文上下文
    """
    g = GraphBuilder()  ## 初始化条文抽取结果图
    section_context = _build_section_context(blocks, source_id)  ## 预先计算每个块所在的最近标题节点，后续给语义节点补 UNDER_SECTION
    i = 0  ## 使用 while 手工控制游标，便于在识别到"主条文 + 分项"时一次性消费多个 block

    while i < len(blocks):  ## 顺序扫描所有 Markdown 块
        block = blocks[i]  ## 当前正在处理的块
        if block.block_type != "paragraph":  ## 只处理 paragraph，标题/表格/图片等跳过
            i += 1
            continue

        match = CLAUSE_RE.match(block.text)  ## 判断该段是否是"7.3.6 正文"这种条款格式
        if not match:  ## 不是条款号开头的段落，不做条文语义抽取
            i += 1
            continue

        clause_no, clause_text = match.groups()  ## 拆出条款号和正文文本
        token, modality = _detect_modality(clause_text)  ## 从正文里识别模态词，例如"应/宜/可/不得"
        if modality is None:  ## 没有模态词则认为不是需要结构化的规范要求，直接跳过
            i += 1
            continue

        clause_id = f"clause:{source_id}:{block.index}"  ## 当前条文在结构层已经对应的 Clause 节点 ID
        section_id = section_context.get(block.index)  ## 当前条文所属的最近标题节点，用于补 section 上下文

        item_blocks_text: List[str] = []  ## 用来暂存主条文后续若干个"编号分项块"的原始文本
        if clause_text.rstrip().endswith(("：", ":")) or "下列" in clause_text:  ## 只有明显像"引出后续分项"的主条文才继续向后扫描
            j = i + 1  ## 从当前条文的下一个块开始看
            saw_items = False  ## 记录是否已经看到过至少一个有效分项
            while j < len(blocks):  ## 继续向后扫描连续块，收集属于当前主条文的分项内容
                next_block = blocks[j]  ## 候选后续块
                if next_block.block_type == "page_marker":  ## 允许分项跨页，因此页码块本身不终止扫描
                    if saw_items:
                        j += 1
                        continue
                    break
                if next_block.block_type != "paragraph":  ## 非段落块说明分项链结束
                    break
                if CLAUSE_RE.match(next_block.text):  ## 遇到新的条款号开头段落，说明进入下一条条文，停止收集
                    break

                item_matches = _split_numbered_items(next_block.text)  ## 看当前段是否能拆出 `1 / 2 / 3` 这种编号分项
                if item_matches:  ## 当前段包含一个或多个编号分项
                    saw_items = True
                    item_blocks_text.append(next_block.text)  ## 保存该段原文，后面统一拆成分项列表
                    j += 1
                    continue

                if saw_items and item_blocks_text and _has_open_continuation(item_blocks_text[-1]):  ## 若上一段最后一项未结束，当前段可视为跨页/续行补全
                    item_blocks_text.append(next_block.text)
                    j += 1
                    continue

                break  ## 其他情况说明已经离开该主条文的分项区域

            clause_items = _split_numbered_items("\n".join(item_blocks_text)) if item_blocks_text else []  ## 把收集到的所有分项文本重新统一拆分为 (item_no, item_text)
            if clause_items:  ## 如果确实抽到了编号分项
                parent_context = clause_text.rstrip("：:").strip()  ## 主条文文本去掉结尾冒号，作为每个子 requirement 的继承上下文
                for item_no, item_text in clause_items:  ## 为每个编号分项分别建一条 Requirement
                    item_token, item_modality = _detect_modality(item_text)  ## 先看子项自己有没有更具体的模态词
                    _create_requirement(
                        g,
                        source_id=source_id,
                        source_path=source_path,
                        clause_id=clause_id,
                        clause_block_index=block.index,
                        clause_no=clause_no,
                        text=item_text,
                        modality_token=item_token or token,  ## 子项没有模态词时继承主条文模态词
                        modality=item_modality or modality,  ## 子项没有模态类别时继承主条文模态类别
                        page_no=block.page_no,
                        section_id=section_id,
                        requirement_type="clause_item_rule",  ## 标记这是"条文分项 requirement"
                        item_no=item_no,
                        parent_context=parent_context,  ## 主条文上下文会被用于补充对象、条件、控制方式等抽取
                    )
                i = j  ## 当前主条文及其后续分项已经整体消费完，游标直接跳到分项之后
                continue

        _create_requirement(  ## 如果不是"主条文 + 分项"结构，就按普通条文生成单条 Requirement
            g,
            source_id=source_id,
            source_path=source_path,
            clause_id=clause_id,
            clause_block_index=block.index,
            clause_no=clause_no,
            text=clause_text,
            modality_token=token,
            modality=modality,
            page_no=block.page_no,
            section_id=section_id,
            requirement_type="clause_rule",
        )
        i += 1  ## 普通条文只消耗当前 block，继续处理下一个块

    return g  ## 返回条文语义抽取后的图
