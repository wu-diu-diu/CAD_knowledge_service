"""
多跳 QA 数据集生成脚本。

从 kg_store/req_metadata.json 中挖掘需要跨多条 Requirement 才能回答的问题，
生成 eval/qa_multihop_dataset.json。

支持三种多跳类型：
  Type 1 - 条件聚合 (condition_agg)
      同一场所在不同活动/条件下的照度标准值分别是多少？
      需要跨 >=2 条带不同 condition 的 Requirement。

  Type 2 - 跨指标综合 (multi_metric)
      某场所需要同时满足哪些照明指标？（照度/UGR/Ra/U0）
      需要跨 >=3 条不同 metric 的 Requirement。

  Type 3 - 跨场所对比 (cross_entity)
      两个相似场所的某项指标有什么区别？
      需要跨 2 个不同 entity 的同类 Requirement。

用法：
    python eval/build_multihop_dataset.py
    python eval/build_multihop_dataset.py --limit 200
    python eval/build_multihop_dataset.py --output eval/qa_multihop_dataset.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 核心照明指标（用于跨指标多跳）
CORE_METRICS = {"照度标准值", "UGR", "Ra", "U0", "参考平面及其高度"}

# 过滤掉非场所类 entity（纯数字、数学表达式等）
_NON_PLACE_RE = re.compile(r"^[\d\s<>=≤≥.+\-*/P%\\()\[\]]+$")

SYSTEM_PROMPT = """
你是建筑规范 QA 数据集构造助手。

你的任务是：根据提供的多条建筑规范 Requirement，生成一道需要综合多条规范才能回答的多跳问题，以及对应的标准答案。

硬性要求：
1. 问题必须是"问数值/问标准"的问法，不能是解释型问题。
2. 问题必须需要用到所有提供的 Requirement 才能完整回答。
3. 答案必须严格来自提供的 Requirement，不能编造数据。
4. 必须返回严格 JSON，格式如下：
{
  "question": "...",
  "answer": "...",
  "answer_raw": "..."
}
其中 answer 是自然语言完整答案，answer_raw 是结构化的简短答案（如"一般活动:100lx; 书写阅读:300lx"）。
5. 不能输出解释，不能输出额外字段。
""".strip()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _is_place_entity(e: str) -> bool:
    if not e or len(e) <= 1:
        return False
    if _NON_PLACE_RE.match(e):
        return False
    return True


def _normalize_metric(m: str) -> str:
    """去掉 metric 名中的 #N 后缀（表格重复列标记）。"""
    return re.sub(r"#\d+$", "", m).strip()


def _load_metadata(kg_store_dir: Path) -> List[Dict[str, Any]]:
    path = kg_store_dir / "req_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"req_metadata.json 不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _filter_lighting(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """只保留建筑照明标准的记录。"""
    return [r for r in records if "建筑照明" in (r.get("source_path") or "")]


# ---------------------------------------------------------------------------
# Type 1：条件聚合多跳
# ---------------------------------------------------------------------------

def _build_condition_agg_candidates(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    找同一场所在不同条件下有不同照度标准值的 entity，
    生成"X 在不同条件下的照度标准值分别是多少"类问题的原材料。
    """
    entity_cond_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if not r.get("conditions"):
            continue
        metrics = [_normalize_metric(m) for m in (r.get("metrics") or [])]
        if "照度标准值" not in metrics:
            continue
        for e in r.get("entities", []):
            if _is_place_entity(e):
                entity_cond_map[e].append(r)

    candidates = []
    for entity, reqs in entity_cond_map.items():
        # 去重：同一 condition 只保留一条
        seen_cond: Dict[str, Dict[str, Any]] = {}
        for r in reqs:
            cond = r["conditions"][0]
            if cond not in seen_cond:
                seen_cond[cond] = r
        if len(seen_cond) < 2:
            continue
        candidates.append({
            "type": "condition_agg",
            "entity": entity,
            "requirements": list(seen_cond.values()),
            "source_path": reqs[0]["source_path"],
        })
    return candidates


# ---------------------------------------------------------------------------
# Type 2：跨指标综合多跳
# ---------------------------------------------------------------------------

def _build_multi_metric_candidates(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    找同一场所在同一表格下有 >=3 个核心照明指标的 entity，
    生成"X 需要满足哪些照明指标要求"类问题的原材料。
    """
    # key: (entity, table_caption)
    key_map: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in records:
        table_cap = (r.get("table_caption") or "").strip()
        if not table_cap:
            continue
        metrics = [_normalize_metric(m) for m in (r.get("metrics") or [])]
        # 只关心核心指标
        core = [m for m in metrics if m in CORE_METRICS]
        if not core:
            continue
        for e in r.get("entities", []):
            if _is_place_entity(e):
                for m in core:
                    key_map[(e, table_cap)][m] = r

    candidates = []
    for (entity, table_cap), metric_req_map in key_map.items():
        # 需要至少有照度标准值 + 另外2个指标
        if "照度标准值" not in metric_req_map:
            continue
        if len(metric_req_map) < 3:
            continue
        candidates.append({
            "type": "multi_metric",
            "entity": entity,
            "table_caption": table_cap,
            "metric_req_map": metric_req_map,
            "requirements": list(metric_req_map.values()),
            "source_path": list(metric_req_map.values())[0]["source_path"],
        })
    return candidates


# ---------------------------------------------------------------------------
# Type 3：跨场所对比多跳
# ---------------------------------------------------------------------------

def _build_cross_entity_candidates(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    找同一表格下两个不同场所的照度标准值，
    生成"X 和 Y 的照度标准值有什么区别"类问题的原材料。
    """
    # 按 table_caption 分组，找有照度标准值的 entity
    table_entity_map: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in records:
        table_cap = (r.get("table_caption") or "").strip()
        if not table_cap:
            continue
        metrics = [_normalize_metric(m) for m in (r.get("metrics") or [])]
        if "照度标准值" not in metrics:
            continue
        for e in r.get("entities", []):
            if _is_place_entity(e):
                if e not in table_entity_map[table_cap]:
                    table_entity_map[table_cap][e] = r

    candidates = []
    for table_cap, entity_req_map in table_entity_map.items():
        entities = list(entity_req_map.keys())
        if len(entities) < 2:
            continue
        # 只取前 N 对，避免组合爆炸
        pairs: List[Tuple[str, str]] = []
        for i in range(min(len(entities), 8)):
            for j in range(i + 1, min(len(entities), 8)):
                pairs.append((entities[i], entities[j]))
        for e1, e2 in pairs[:6]:  # 每张表最多6对
            r1 = entity_req_map[e1]
            r2 = entity_req_map[e2]
            v1 = (r1.get("values") or ["?"])[0]
            v2 = (r2.get("values") or ["?"])[0]
            if v1 == v2:
                continue  # 值相同的对比没意义
            candidates.append({
                "type": "cross_entity",
                "entity_1": e1,
                "entity_2": e2,
                "table_caption": table_cap,
                "requirements": [r1, r2],
                "source_path": r1["source_path"],
            })
    return candidates


# ---------------------------------------------------------------------------
# LLM 问题生成
# ---------------------------------------------------------------------------

def _get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少 openai 依赖，请安装 `openai`。") from exc
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip() or "https://api.deepseek.com/v1"
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY。")
    return OpenAI(api_key=api_key, base_url=base_url)


def _build_user_prompt(candidate: Dict[str, Any]) -> str:
    hop_type = candidate["type"]
    reqs = candidate["requirements"]

    req_lines = []
    for i, r in enumerate(reqs, 1):
        metrics = [_normalize_metric(m) for m in (r.get("metrics") or [])]
        req_lines.append(
            f"Requirement {i}: 场所={r.get('entities')}, "
            f"指标={metrics}, 值={r.get('values')}, 条件={r.get('conditions')}"
        )
    req_text = "\n".join(req_lines)

    if hop_type == "condition_agg":
        entity = candidate["entity"]
        hint = f"场所\"{entity}\"在不同条件下的照度标准值不同，请生成一道需要列举所有条件及对应照度值的问题。"
    elif hop_type == "multi_metric":
        entity = candidate["entity"]
        metrics_list = list(candidate["metric_req_map"].keys())
        hint = f"场所\"{entity}\"有多个照明指标要求（{metrics_list}），请生成一道需要综合列举所有指标及其标准值的问题。"
    else:  # cross_entity
        e1 = candidate["entity_1"]
        e2 = candidate["entity_2"]
        hint = f"场所\"{e1}\"和\"{e2}\"的照度标准值不同，请生成一道对比两者照度标准值的问题。"

    return (
        f"以下是来自建筑规范的多条 Requirement：\n\n{req_text}\n\n"
        f"提示：{hint}\n\n"
        "请生成一道多跳问题及答案，严格只输出 JSON。"
    )


def _json_load_loose(text: str) -> Optional[Dict[str, Any]]:
    payload = (text or "").strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", payload, re.I)
    if match:
        payload = match.group(1).strip()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _generate_qa(
    client: Any,
    candidate: Dict[str, Any],
    model: str,
    max_retries: int,
    retry_sleep: float,
) -> Dict[str, str]:
    prompt = _build_user_prompt(candidate)
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = completion.choices[0].message.content or ""
            payload = _json_load_loose(content)
            if payload is None:
                raise ValueError(f"LLM 未返回合法 JSON。原始输出: {content[:300]}")
            question = str(payload.get("question") or "").strip()
            answer = str(payload.get("answer") or "").strip()
            answer_raw = str(payload.get("answer_raw") or "").strip()
            if not question or not answer:
                raise ValueError("question 或 answer 为空。")
            return {"question": question, "answer": answer, "answer_raw": answer_raw}
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_sleep)
    raise RuntimeError(f"QA 生成失败: {last_error}") from last_error


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _iter_with_progress(items, total: int, desc: str):
    try:
        from tqdm.auto import tqdm
        return tqdm(items, total=total, desc=desc)
    except Exception:
        return items


def build_multihop_dataset(
    kg_store_dir: Path,
    *,
    model: str,
    max_retries: int,
    retry_sleep: float,
    limit: Optional[int],
    type_limit: Optional[int],
    source_filter: str,
) -> List[Dict[str, Any]]:
    records = _load_metadata(kg_store_dir)
    if source_filter:
        records = [r for r in records if source_filter in (r.get("source_path") or "")]
    print(f"[multihop] 加载 requirement 数: {len(records)}")

    # 挖掘候选
    cond_candidates = _build_condition_agg_candidates(records)
    metric_candidates = _build_multi_metric_candidates(records)
    cross_candidates = _build_cross_entity_candidates(records)

    print(f"[multihop] 候选数 — 条件聚合: {len(cond_candidates)}, "
          f"跨指标: {len(metric_candidates)}, 跨场所对比: {len(cross_candidates)}")

    # 按类型限制数量，保证三种类型均衡
    if type_limit:
        cond_candidates = cond_candidates[:type_limit]
        metric_candidates = metric_candidates[:type_limit]
        cross_candidates = cross_candidates[:type_limit]

    all_candidates = cond_candidates + metric_candidates + cross_candidates
    if limit and limit < len(all_candidates):
        # 均衡采样
        import random
        random.shuffle(all_candidates)
        all_candidates = all_candidates[:limit]

    print(f"[multihop] 实际生成候选数: {len(all_candidates)}")

    client = _get_openai_client()
    output: List[Dict[str, Any]] = []
    failures = 0

    iterator = _iter_with_progress(all_candidates, total=len(all_candidates), desc="MultiHop QA Gen")
    for candidate in iterator:
        try:
            qa = _generate_qa(
                client, candidate,
                model=model,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
            )
        except Exception as exc:
            failures += 1
            print(f"[multihop] 跳过候选 ({candidate['type']}): {exc}")
            continue

        req_ids = [r["requirement_id"] for r in candidate["requirements"]]
        record: Dict[str, Any] = {
            "id": f"mh_{len(output) + 1:05d}",
            "hop_type": candidate["type"],
            "question": qa["question"],
            "answer": qa["answer"],
            "answer_raw": qa["answer_raw"],
            "requirement_ids": req_ids,
            "source_path": candidate["source_path"],
        }
        # 附加类型特有字段
        if candidate["type"] == "condition_agg":
            record["entity"] = candidate["entity"]
        elif candidate["type"] == "multi_metric":
            record["entity"] = candidate["entity"]
            record["metrics"] = list(candidate["metric_req_map"].keys())
        else:
            record["entity_1"] = candidate["entity_1"]
            record["entity_2"] = candidate["entity_2"]

        output.append(record)

    print(f"[multihop] 成功生成: {len(output)}, 失败: {failures}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="生成多跳 QA 数据集")
    parser.add_argument("--kg_store", default="kg_store", help="kg_store 目录路径")
    parser.add_argument(
        "--output", default="eval/qa_multihop_dataset.json", help="输出路径"
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CAD_AGENT_DEEPSEEK_MODEL", "deepseek-chat").strip(),
        help="DeepSeek 模型名",
    )
    parser.add_argument("--limit", type=int, default=None, help="总候选数上限")
    parser.add_argument(
        "--type_limit", type=int, default=50,
        help="每种多跳类型的候选数上限（默认50，三种类型共最多150条）"
    )
    parser.add_argument(
        "--source_filter", default="建筑照明",
        help="只处理 source_path 包含该字符串的记录（默认：建筑照明）"
    )
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=1.5)
    args = parser.parse_args()

    kg_store_dir = Path(args.kg_store)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_multihop_dataset(
        kg_store_dir,
        model=args.model,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        limit=args.limit,
        type_limit=args.type_limit,
        source_filter=args.source_filter,
    )

    output_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已保存到: {output_path}（共 {len(dataset)} 条）")

    if dataset:
        print("\n示例（前3条）：")
        for item in dataset[:3]:
            print(f"  [{item['id']}] [{item['hop_type']}] {item['question']}")
            print(f"    答案: {item['answer_raw']}")


if __name__ == "__main__":
    main()
