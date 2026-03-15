from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from .store_neo4j import run_cypher_query

FORBIDDEN_CYPHER_RE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL|LOAD\s+CSV|FOREACH)\b",
    re.I,
)

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.I)
ENTITY_METRIC_QUESTION_RE = re.compile(r"^\s*(?P<entity>.+?)的(?P<metric>.+?)(?:是多少|是什么|为多少|有哪些)\s*[？?]?\s*$")

SCHEMA_SUMMARY = """
图谱标签（常用）:
- StandardDocument(source_id, source_path, source_type)
- Page(page_no)
- Section(title, level, section_no, page_no)
- Clause(clause_no, text, page_no)
- Table(table_no, title, page_no)
- TableRow(row_index, values, headers, page_no)
- Requirement(requirement_type, text/raw_cell, modality, page_no, clause_no, table_caption)
- DomainEntity(name, canonical_name)
- ControlMethod(name)
- Metric(name, canonical_name)
- ValueSpec(raw_text, value, unit)
- Condition(text, condition_type)
- EvidenceSpan(text, page_no, block_type)
- Figure(name, caption, figure_no, purpose, role_in_paper, key_insights, visual_elements_summary)

关系（常用）:
- HAS_SECTION, HAS_CLAUSE, HAS_TABLE, HAS_ROW, HAS_FIGURE
- LOCATED_ON_PAGE, HAS_EVIDENCE, UNDER_SECTION
- CLAUSE_EXPRESSES_REQUIREMENT, ROW_EXPRESSES_REQUIREMENT
- APPLIES_TO, CONSTRAINS_METRIC, HAS_VALUE_SPEC, UNDER_CONDITION, USES_CONTROL_METHOD, SOURCE_OF
- ALIAS_OF, ALIAS_OF_METRIC

Neo4j 存储规则:
- 所有节点都有基类标签 :KGNode，并且有唯一属性 id
- 业务标签同时存在，如 (:KGNode:Requirement)、(:KGNode:DomainEntity)
- 节点原始类型在属性 kind 中
"""

QUERY_PATTERNS = """
常见查询模式:
1. 查询某对象的某项标准值（最常见，例如“起居室的照度标准是多少”）
   推荐模式:
   MATCH (e:KGNode:DomainEntity)
   WHERE e.name = $entity OR e.canonical_name = $entity
   MATCH (r:KGNode:Requirement)-[:APPLIES_TO]->(e)
   MATCH (r)-[:CONSTRAINS_METRIC]->(m:KGNode:Metric)
   MATCH (r)-[:HAS_VALUE_SPEC]->(v:KGNode:ValueSpec)
   OPTIONAL MATCH (r)-[:UNDER_CONDITION]->(c:KGNode:Condition)
   WHERE r.requirement_type = "table_cell_constraint"
     AND (m.name CONTAINS $metric OR m.canonical_name = $metric OR m.name = "照度")
   RETURN e.name AS entity, c.text AS condition, m.name AS metric, v.raw_text AS raw_value, v.value AS value, v.unit AS unit, r.table_caption AS table_caption, r.page_no AS page_no, r.source_path AS source_path
   ORDER BY c.text, r.page_no
   LIMIT $limit

2. 查询某对象的所有属性值
   从 DomainEntity 出发，连到 Requirement，再连 Metric/ValueSpec/Condition。

3. 查询某种控制方式适用于哪些场所
   从 ControlMethod 出发，反查 Requirement，再查 DomainEntity/Condition。

重要约束:
- 一律使用 (:KGNode:Label) 形式写节点标签。
- 优先用参数: $entity, $metric, $source_id, $limit，不要把用户值硬编码进 Cypher。
- 如果问题是“某对象的标准值是多少”，优先限定 r.requirement_type = "table_cell_constraint"。
- 如果要返回条件，使用 OPTIONAL MATCH (r)-[:UNDER_CONDITION]->(c:KGNode:Condition)。
- 如果问题明显在问“照度标准”，优先匹配 Metric 中的“照度标准值”，不要只查泛化的 Clause 文本。
"""

GEN_CYPHER_SYSTEM_PROMPT = f"""
你是一个 Neo4j Cypher 查询生成器，服务于建筑设计规范知识图谱问答。
你的任务是把用户问题转成只读 Cypher 查询。

要求:
1. 只生成只读查询（MATCH/OPTIONAL MATCH/WITH/WHERE/RETURN/ORDER BY/LIMIT）。
2. 不要使用 CREATE/MERGE/DELETE/SET/CALL 等写入或管理语句。
3. 返回结果时优先包含: requirement、entity、metric、value、page_no、source_path、clause_no/table_caption。
4. 使用参数化查询；限制条数使用 $limit。
5. 如果用户给了 entity/metric/source_id 提示，请尽量利用；如果没给但问题里能明确识别，例如“起居室的照度标准是多少”，你应主动在 params 中填入 entity="起居室"、metric="照度标准值" 等。
6. 输出严格 JSON，不要输出多余解释。
7. 生成 Cypher 时，优先复用下方“常见查询模式”，不要自行发明与 schema 不一致的关系。
8. 不要写 `MATCH (m:Metric {{name: $metric}})` 这类对 Metric.name 的精确匹配；指标名可能带单位后缀，例如 `照度标准值(lx)`、`照度标准值(1x)`，应优先使用 `m.canonical_name = $metric` 或 `m.name CONTAINS $metric`。
9. 优先使用 `(:KGNode:DomainEntity)`、`(:KGNode:Requirement)`、`(:KGNode:Metric)` 等完整标签形式。

{SCHEMA_SUMMARY}

{QUERY_PATTERNS}

输出 JSON 格式:
{{
  "cypher": "MATCH ... RETURN ... LIMIT $limit",
  "params": {{"entity": "...", "metric": "..."}},
  "reasoning": "简短说明查询意图"
}}
"""

ANSWER_SYSTEM_PROMPT = """
你是建筑规范知识图谱问答助手。请基于查询结果回答用户问题。
要求:
1. 仅根据提供的数据回答，不要编造。
2. 优先给出明确数值、单位、适用对象和条件。
3. 尽量引用页码(page_no)和来源(source_path)。
4. 若结果不足以回答，明确说明缺少什么信息。
5. 使用中文，简洁清晰。
6. 只回答与用户问题直接相关的结果，不要补充“其他结果无关、已忽略”等说明。
7. 如果同一对象在不同条件下有多条结果，直接按条件列出即可。
8. 不要扩展解释星号、注释或规范含义，除非查询结果中明确包含这些信息。
9. 如果提供了 candidate_entities，请优先参考得分更高的候选实体；除非用户明确要求模糊匹配结果，否则不要把低相关候选混入最终答案。
"""


def _json_load_loose(text: str) -> Optional[dict]:
    payload = text.strip()
    m = JSON_BLOCK_RE.search(payload)
    if m:
        payload = m.group(1).strip()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("缺少 openai 依赖，请安装 `openai`。") from e

    api_key = os.getenv("KG_LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 KG_LLM_API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY。")

    base_url = os.getenv("KG_LLM_BASE_URL")
    if not base_url and os.getenv("DASHSCOPE_API_KEY"):
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _chat_json(prompt: str, system_prompt: str, model: str) -> dict:
    client = _get_openai_client()
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    content = completion.choices[0].message.content or ""
    parsed = _json_load_loose(content)
    if parsed is None:
        raise RuntimeError(f"LLM 未返回有效 JSON。原始输出: {content[:500]}")
    return parsed


def _chat_text(prompt: str, system_prompt: str, model: str) -> str:
    client = _get_openai_client()
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return (completion.choices[0].message.content or "").strip()


def validate_read_only_cypher(cypher: str) -> str:
    query = (cypher or "").strip()
    if not query:
        raise ValueError("LLM 生成的 Cypher 为空。")
    if ";" in query.rstrip(";"):
        raise ValueError("不允许多条 Cypher 语句。")
    query = query.rstrip(";").strip()

    if FORBIDDEN_CYPHER_RE.search(query):
        raise ValueError("Cypher 包含禁止的写入/管理语句。")
    if "return" not in query.lower():
        raise ValueError("Cypher 必须包含 RETURN。")
    if "match" not in query.lower():
        raise ValueError("Cypher 必须包含 MATCH。")
    return query


def ensure_limit_param(cypher: str) -> str:
    if re.search(r"\blimit\b", cypher, re.I):
        return cypher
    return f"{cypher}\nLIMIT $limit"


def _infer_metric_hint(question: str, metric: Optional[str]) -> Optional[str]:
    if metric and metric.strip():
        raw = metric.strip()
    else:
        q = (question or "").strip()
        match = ENTITY_METRIC_QUESTION_RE.match(q)
        raw = (match.group("metric") if match else q).strip()

    if not raw:
        return None
    if "照度标准" in raw:
        return "照度标准值"
    if raw == "照度":
        return "照度"
    if "显色指数" in raw or raw.lower() == "ra":
        return "Ra"
    if "眩光" in raw or raw.upper() == "GR":
        return "GR"
    return raw


def _infer_entity_hint(question: str, entity: Optional[str]) -> Optional[str]:
    if entity and entity.strip():
        return entity.strip()
    q = (question or "").strip()
    match = ENTITY_METRIC_QUESTION_RE.match(q)
    if match:
        return match.group("entity").strip()
    return None


def _score_entity_candidate(query: str, candidate: Dict[str, Any]) -> float:
    q = (query or "").strip()
    name = str(candidate.get("name") or "").strip()
    canonical = str(candidate.get("canonical_name") or "").strip()
    score = 0.0

    if not q:
        return score

    if name == q:
        score += 120.0
    if canonical == q:
        score += 120.0

    if q and q in name:
        score += 80.0
        score += min(len(q) / max(len(name), 1), 1.0) * 10.0
        if name.endswith(q) and name != q:
            score += 8.0
        if name.startswith(q) and name != q:
            score -= 10.0
    if q and q in canonical:
        score += 80.0
        score += min(len(q) / max(len(canonical), 1), 1.0) * 10.0
        if canonical.endswith(q) and canonical != q:
            score += 8.0
        if canonical.startswith(q) and canonical != q:
            score -= 10.0

    if name and name in q and name != q:
        score += 55.0
    if canonical and canonical in q and canonical != q:
        score += 55.0

    if name:
        score -= max(len(name) - len(q), 0) * 0.05
    if canonical:
        score -= max(len(canonical) - len(q), 0) * 0.05

    return score


def _recall_entity_candidates(entity_hint: str, top_k: int = 3) -> List[Dict[str, Any]]:
    hint = (entity_hint or "").strip()
    if not hint:
        return []

    cypher = """
MATCH (e:KGNode:DomainEntity)
WHERE e.name CONTAINS $entity
   OR e.canonical_name CONTAINS $entity
   OR $entity CONTAINS e.name
   OR $entity CONTAINS e.canonical_name
RETURN
  e.id AS entity_id,
  e.name AS name,
  e.canonical_name AS canonical_name
LIMIT $limit
""".strip()
    result = run_cypher_query(
        cypher=cypher,
        params={"entity": hint},
        top_k=max(20, top_k * 5),
    )

    deduped: Dict[str, Dict[str, Any]] = {}
    for row in result.get("rows", []):
        entity_id = str(row.get("entity_id") or "").strip()
        if not entity_id:
            continue
        row = dict(row)
        row["score"] = _score_entity_candidate(hint, row)
        if entity_id not in deduped or row["score"] > float(deduped[entity_id].get("score", 0.0)):
            deduped[entity_id] = row

    ranked = sorted(
        deduped.values(),
        key=lambda item: (
            -float(item.get("score", 0.0)),
            len(str(item.get("name") or "")),
            str(item.get("name") or ""),
        ),
    )
    return ranked[:top_k]


def _build_entity_metric_query(
    *,
    entity: str,
    metric: str,
    source_id: Optional[str],
    top_k: int,
    prefer_table: bool,
    entity_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    where_lines = [
        '(m.name CONTAINS $metric OR m.canonical_name = $metric OR ($metric = "照度标准值" AND m.name = "照度"))',
    ]
    if prefer_table:
        where_lines.append('r.requirement_type = "table_cell_constraint"')
    if source_id:
        where_lines.append('r.source_id = $source_id')

    candidates = [candidate for candidate in (entity_candidates or []) if candidate.get("entity_id")]
    use_candidate_ids = bool(candidates)
    entity_match = "WHERE e.id IN $entity_ids" if use_candidate_ids else "WHERE e.name = $entity OR e.canonical_name = $entity"

    cypher = f"""
MATCH (e:KGNode:DomainEntity)
{entity_match}
MATCH (r:KGNode:Requirement)-[:APPLIES_TO]->(e)
MATCH (r)-[:CONSTRAINS_METRIC]->(m:KGNode:Metric)
MATCH (r)-[:HAS_VALUE_SPEC]->(v:KGNode:ValueSpec)
WHERE {' AND '.join(where_lines)}
WITH e, r, m, v
OPTIONAL MATCH (r)-[:UNDER_CONDITION]->(c:KGNode:Condition)
RETURN
  e.name AS entity,
  c.text AS condition,
  m.name AS metric,
  v.raw_text AS raw_value,
  v.value AS value,
  v.unit AS unit,
  r.table_caption AS table_caption,
  r.clause_no AS clause_no,
  r.page_no AS page_no,
  r.source_path AS source_path
ORDER BY c.text, r.page_no
LIMIT $limit
""".strip()

    params: Dict[str, Any] = {
        "entity": entity,
        "metric": metric,
        "limit": int(max(1, min(top_k, 200))),
    }
    if use_candidate_ids:
        params["entity_ids"] = [str(candidate["entity_id"]) for candidate in candidates]
    if source_id:
        params["source_id"] = source_id

    return {
        "cypher": cypher,
        "params": params,
        "reasoning": "deterministic entity-metric query template with recalled entity candidates" if use_candidate_ids else "deterministic entity-metric query template",
        "model": "deterministic_template",
        "candidate_entities": candidates,
    }


def generate_cypher_plan(
    question: str,
    entity: Optional[str] = None,
    metric: Optional[str] = None,
    source_id: Optional[str] = None,
    top_k: int = 20,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    inferred_entity = _infer_entity_hint(question, entity)
    inferred_metric = _infer_metric_hint(question, metric)
    prefer_table = "标准" in (question or "") or (inferred_metric == "照度标准值")
    if inferred_entity and inferred_metric:
        entity_candidates = _recall_entity_candidates(inferred_entity, top_k=3)
        return _build_entity_metric_query(
            entity=inferred_entity,
            metric=inferred_metric,
            source_id=source_id,
            top_k=top_k,
            prefer_table=prefer_table,
            entity_candidates=entity_candidates,
        )

    llm_model = model or os.getenv("KG_LLM_MODEL", "qwen-plus")
    user_prompt = {
        "question": question,
        "hints": {
            "entity": inferred_entity or entity,
            "metric": inferred_metric or metric,
            "source_id": source_id,
        },
        "limit": int(max(1, min(top_k, 200))),
    }
    plan = _chat_json(
        prompt=json.dumps(user_prompt, ensure_ascii=False, indent=2),
        system_prompt=GEN_CYPHER_SYSTEM_PROMPT,
        model=llm_model,
    )
    raw_cypher = str(plan.get("cypher") or "").strip()
    safe_cypher = ensure_limit_param(validate_read_only_cypher(raw_cypher))
    params = plan.get("params") if isinstance(plan.get("params"), dict) else {}
    params = dict(params)
    params["limit"] = int(max(1, min(top_k, 200)))
    if source_id and "source_id" not in params:
        params["source_id"] = source_id
    return {
        "cypher": safe_cypher,
        "params": params,
        "reasoning": plan.get("reasoning"),
        "model": llm_model,
    }


def answer_with_llm(
    question: str,
    cypher: str,
    query_result: Dict[str, Any],
    candidate_entities: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
) -> str:
    llm_model = model or os.getenv("KG_LLM_ANSWER_MODEL") or os.getenv("KG_LLM_MODEL", "qwen-plus")
    payload = {
        "question": question,
        "cypher": cypher,
        "candidate_entities": candidate_entities or [],
        "result_keys": query_result.get("keys", []),
        "result_rows": query_result.get("rows", []),
        "result_count": query_result.get("count", 0),
    }
    return _chat_text(
        prompt=json.dumps(payload, ensure_ascii=False, indent=2),
        system_prompt=ANSWER_SYSTEM_PROMPT,
        model=llm_model,
    )


def llm_query_graph(
    question: str,
    entity: Optional[str] = None,
    metric: Optional[str] = None,
    source_id: Optional[str] = None,
    top_k: int = 20,
    model: Optional[str] = None,
    synthesize_answer: bool = True,
) -> Dict[str, Any]:
    plan = generate_cypher_plan(
        question=question,
        entity=entity,
        metric=metric,
        source_id=source_id,
        top_k=top_k,
        model=model,
    )
    query_result = run_cypher_query(
        cypher=plan["cypher"],
        params=plan["params"],
        top_k=top_k,
    )
    response: Dict[str, Any] = {
        "question": question,
        "cypher": plan["cypher"],
        "params": plan["params"],
        "reasoning": plan.get("reasoning"),
        "candidate_entities": plan.get("candidate_entities", []),
        "query_result": query_result,
    }
    if synthesize_answer:
        try:
            response["answer"] = answer_with_llm(
                question=question,
                cypher=plan["cypher"],
                query_result=query_result,
                candidate_entities=plan.get("candidate_entities", []),
                model=model,
            )
        except Exception as e:
            response["answer_error"] = str(e)
    return response
