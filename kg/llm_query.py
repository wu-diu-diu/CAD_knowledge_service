from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from .store_neo4j import run_cypher_query

FORBIDDEN_CYPHER_RE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL|LOAD\s+CSV|FOREACH)\b",
    re.I,
)

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.I)

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
- Metric(name, canonical_name)
- ValueSpec(raw_text, value, unit)
- Condition(text, condition_type)
- EvidenceSpan(text, page_no, block_type)
- Figure / FigureInsight / KeyInsight

关系（常用）:
- HAS_SECTION, HAS_CLAUSE, HAS_TABLE, HAS_ROW, HAS_FIGURE
- LOCATED_ON_PAGE, HAS_EVIDENCE
- CLAUSE_EXPRESSES_REQUIREMENT, ROW_EXPRESSES_REQUIREMENT
- APPLIES_TO, CONSTRAINS_METRIC, HAS_VALUE_SPEC, UNDER_CONDITION, SOURCE_OF
- ALIAS_OF, ALIAS_OF_METRIC

Neo4j 存储规则:
- 所有节点都有基类标签 :KGNode，并且有唯一属性 id
- 业务标签同时存在，如 (:KGNode:Requirement)、(:KGNode:DomainEntity)
- 节点原始类型在属性 kind 中
"""

GEN_CYPHER_SYSTEM_PROMPT = f"""
你是一个 Neo4j Cypher 查询生成器，服务于建筑设计规范知识图谱问答。
你的任务是把用户问题转成只读 Cypher 查询。

要求:
1. 只生成只读查询（MATCH/OPTIONAL MATCH/WITH/WHERE/RETURN/ORDER BY/LIMIT）。
2. 不要使用 CREATE/MERGE/DELETE/SET/CALL 等写入或管理语句。
3. 返回结果时优先包含: requirement、entity、metric、value、page_no、source_path、clause_no/table_caption。
4. 使用参数化查询；限制条数使用 $limit。
5. 如果用户给了 entity/metric/source_id 提示，请尽量利用。
6. 输出严格 JSON，不要输出多余解释。

{SCHEMA_SUMMARY}

输出 JSON 格式:
{{
  "cypher": "MATCH ... RETURN ... LIMIT $limit",
  "params": {{}},
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


def generate_cypher_plan(
    question: str,
    entity: Optional[str] = None,
    metric: Optional[str] = None,
    source_id: Optional[str] = None,
    top_k: int = 20,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    llm_model = model or os.getenv("KG_LLM_MODEL", "qwen-plus")
    user_prompt = {
        "question": question,
        "hints": {
            "entity": entity,
            "metric": metric,
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
    model: Optional[str] = None,
) -> str:
    llm_model = model or os.getenv("KG_LLM_ANSWER_MODEL") or os.getenv("KG_LLM_MODEL", "qwen-plus")
    payload = {
        "question": question,
        "cypher": cypher,
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
        "query_result": query_result,
    }
    if synthesize_answer:
        try:
            response["answer"] = answer_with_llm(
                question=question,
                cypher=plan["cypher"],
                query_result=query_result,
                model=model,
            )
        except Exception as e:
            response["answer_error"] = str(e)
    return response

