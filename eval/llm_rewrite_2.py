"""
使用 LLM 对 QA 数据集中的问题进行受控改写，生成更适合检索评测的数据集。

设计目标：
- 不修改原始事实：entity / metric / condition / answer 必须保持对应同一条 requirement
- 比原模板题更接近真实用户提问
- 避免明显错误术语替换（例如 照度 -> 亮度）污染 gold label
- 让 query 同时保留部分词面信号和部分语义改写，更容易拉开 BM25 / 向量 / 混合检索差异

默认行为：
- 读取 eval/qa_dataset.json
- 对每条问题调用 DeepSeek 兼容接口
- 生成 3 条受控改写问题（轻改写 / 中改写 / 强改写）
- 输出到 eval/qa_llm_dataset_v2.json

用法：
    python eval/llm_rewrite_2.py
    python eval/llm_rewrite_2.py --limit 20
    python eval/llm_rewrite_2.py --model deepseek-chat --max-retries 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.I)

SYSTEM_PROMPT = """
你是建筑规范 QA 检索评测数据改写助手。

你的任务是：
针对一条建筑规范问答样本，生成 3 条“受控改写”问题，用于检索评测。
目标不是随意扩写，而是生成：
- 比原模板问法更自然；
- 但仍严格对应同一条 requirement；
- 并且保留部分规范术语，让 BM25、向量、混合检索都能体现差异。

硬性要求：
1. 必须保持原问题查询的事实不变，不能改答案，不能改适用条件，不能改指标含义。
2. 必须生成恰好 3 条不同的改写问题，对应三种难度：
   - easy: 轻改写。尽量保留实体、指标、条件，只调整语序和句式。
   - medium: 中改写。允许实体做有限泛化或常见别名替换，指标允许有限自然化表达，但必须保留核心专业含义。
   - hard: 强改写。更口语化、更像真实用户提问，但仍必须能唯一对应原始事实。
3. 必须返回严格 JSON，格式如下：
{
  "rewrites": [
    {"difficulty": "easy", "question": "..."},
    {"difficulty": "medium", "question": "..."},
    {"difficulty": "hard", "question": "..."}
  ]
}
4. 不能输出解释，不能输出答案，不能输出额外字段。

术语约束：
5. 对 metric 的改写必须保持专业含义：
   - “照度标准值” 可以改成 “照度要求” / “照度标准” / “照度值要求”
   - “Ra” 可以改成 “显色指数” / “Ra值”
   - “UGR” 可以改成 “眩光值” / “统一眩光值”
6. 严禁错误替换：
   - 严禁把“照度”改成“亮度”
   - 严禁把“Ra”改成模糊的“灯光效果”
   - 严禁引入“推荐 / 最低 / 最高 / 普通 / 高档”等原样本没有的限定词
7. entity 允许有限泛化，但必须可回指到原场所：
   - “普通办公室” -> “办公室” 可以
   - “起居室” -> “客厅” 可以
   - 但不要改成过于宽泛、生活化、歧义过大的表达
8. condition 如果原样本存在，改写后不能丢失该条件语义。
9. 改写问题必须仍然是“问数值/问标准”的问题，不能改成解释型问题。
""".strip()


FORBIDDEN_PATTERNS = [
    "亮度",
    "灯光效果",
    "推荐",
    "最低",
    "最高",
]

METRIC_HINTS = {
    "照度标准值": ["照度", "照度标准", "照度要求"],
    "Ra": ["ra", "显色指数", "ra值"],
    "UGR": ["ugr", "眩光", "眩光值", "统一眩光值"],
}


def _json_load_loose(text: str) -> Optional[Dict[str, Any]]:
    payload = (text or "").strip()
    match = JSON_BLOCK_RE.search(payload)
    if match:
        payload = match.group(1).strip()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _normalize_question(text: str) -> str:
    normalized = re.sub(r"\s+", "", (text or "").strip()).lower()
    normalized = normalized.replace("？", "?")
    return normalized


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"输入文件不是列表: {path}")
    records: List[Dict[str, Any]] = []
    required_keys = {
        "id",
        "question",
        "answer",
        "answer_raw",
        "requirement_id",
        "entity",
        "metric",
        "condition",
        "source_path",
        "page_no",
    }
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"第 {idx} 条样本不是对象。")
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"第 {idx} 条样本缺少字段: {sorted(missing)}")
        records.append(item)
    return records


def _get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("缺少 openai 依赖，请安装 `openai`。") from exc

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip() or "https://api.deepseek.com/v1"
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY。")
    return OpenAI(api_key=api_key, base_url=base_url)


def _metric_guidance(metric: str) -> str:
    if metric == "照度标准值":
        return "指标改写时必须保留“照度”这个词，可使用：照度要求 / 照度标准 / 照度值要求；严禁改成‘亮度’。"
    if metric == "Ra":
        return "指标改写时可使用：Ra / Ra值 / 显色指数；不能改成模糊表达。"
    if metric == "UGR":
        return "指标改写时可使用：UGR / 眩光值 / 统一眩光值。"
    return f"指标“{metric}”必须保留其核心专业含义，不能改成泛化描述。"


def _build_user_prompt(record: Dict[str, Any]) -> str:
    condition = record.get("condition")
    condition_text = f"条件: {condition}\n" if condition else "条件: 无\n"
    return (
        "请对下面这条建筑规范 QA 问题生成 3 条受控改写，分别是 easy / medium / hard。\n\n"
        f"原问题: {record['question']}\n"
        f"场所(entity): {record['entity']}\n"
        f"指标(metric): {record['metric']}\n"
        f"{condition_text}"
        f"答案: {record['answer_raw']}\n\n"
        "请严格遵守：\n"
        "- 必须仍然对应同一条 requirement；\n"
        "- 如果有条件，必须保留条件语义；\n"
        "- 问题必须仍然是问数值/问标准，不要变成解释型问法；\n"
        "- easy：轻改写，只改语序和句式；\n"
        "- medium：中改写，可做有限实体泛化/常见别名替换；\n"
        "- hard：强改写，更口语，但不能引入歧义；\n"
        f"- {_metric_guidance(str(record['metric']))}\n"
        "- 严禁出现：亮度、灯光效果、推荐、最低、最高；\n"
        "- 只输出严格 JSON。\n"
    )


def _contains_required_metric_signal(question: str, metric: str) -> bool:
    norm = _normalize_question(question)
    hints = METRIC_HINTS.get(metric)
    if not hints:
        return True
    return any(hint in norm for hint in hints)


def _validate_rewrite(question: str, record: Dict[str, Any]) -> Optional[str]:
    text = (question or "").strip()
    if not text:
        return "空问题"

    norm = _normalize_question(text)
    if not norm:
        return "空问题"

    for bad in FORBIDDEN_PATTERNS:
        if bad.lower() in norm:
            return f"包含禁用词: {bad}"

    if "多少" not in text and "几" not in text and "达到" not in text and "要求" not in text and "标准" not in text:
        return "不像数值/标准提问"

    if not _contains_required_metric_signal(text, str(record.get("metric") or "")):
        return "缺少指标核心信号"

    condition = (record.get("condition") or "").strip()
    if condition and condition not in text:
        cond_norm = _normalize_question(condition)
        text_norm = _normalize_question(text)
        if cond_norm and cond_norm not in text_norm:
            return "丢失条件语义"

    return None


def _parse_rewrites(payload: Dict[str, Any], record: Dict[str, Any]) -> List[Dict[str, str]]:
    items = payload.get("rewrites")
    if not isinstance(items, list):
        raise ValueError("LLM 返回中缺少 rewrites 列表。")

    expected = ["easy", "medium", "hard"]
    seen_diff = set()
    seen_question = set()
    rewrites: List[Dict[str, str]] = []
    original_norm = _normalize_question(str(record["question"]))

    for item in items:
        if not isinstance(item, dict):
            continue
        difficulty = str(item.get("difficulty") or "").strip().lower()
        question = str(item.get("question") or "").strip()
        if difficulty not in expected or difficulty in seen_diff:
            continue
        norm = _normalize_question(question)
        if not norm or norm == original_norm or norm in seen_question:
            continue
        error = _validate_rewrite(question, record)
        if error:
            continue
        seen_diff.add(difficulty)
        seen_question.add(norm)
        rewrites.append({"difficulty": difficulty, "question": question})

    if {item["difficulty"] for item in rewrites} != set(expected):
        raise ValueError(f"有效改写不完整，实际得到: {[item['difficulty'] for item in rewrites]}")

    rewrites.sort(key=lambda x: expected.index(x["difficulty"]))
    return rewrites


def rewrite_question(
    client: Any,
    record: Dict[str, Any],
    *,
    model: str,
    max_retries: int,
    retry_sleep: float,
) -> List[Dict[str, str]]:
    prompt = _build_user_prompt(record)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = completion.choices[0].message.content or ""
            payload = _json_load_loose(content)
            if payload is None:
                raise ValueError(f"LLM 未返回合法 JSON。原始输出: {content[:500]}")
            return _parse_rewrites(payload, record)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_sleep)

    raise RuntimeError(
        f"问题改写失败。question={record['question']!r} error={last_error}"
    ) from last_error


def _iter_with_progress(records: Iterable[Dict[str, Any]], total: int, desc: str):
    try:
        from tqdm.auto import tqdm

        return tqdm(records, total=total, desc=desc)
    except Exception:
        return records


def build_rewrite_dataset(
    records: List[Dict[str, Any]],
    *,
    model: str,
    max_retries: int,
    retry_sleep: float,
) -> List[Dict[str, Any]]:
    client = _get_openai_client()
    output: List[Dict[str, Any]] = []

    iterator = _iter_with_progress(records, total=len(records), desc="LLM Rewriting v2")
    failures = 0
    for record in iterator:
        try:
            rewrites = rewrite_question(
                client,
                record,
                model=model,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
            )
        except Exception as exc:
            failures += 1
            print(f"[llm_rewrite_2] 跳过样本 {record['id']}: {exc}")
            continue

        for item in rewrites:
            output.append({
                "id": f"qa_llm2_{len(output) + 1:06d}",
                "question": item["question"],
                "difficulty": item["difficulty"],
                "answer": record["answer"],
                "answer_raw": record["answer_raw"],
                "requirement_id": record["requirement_id"],
                "entity": record["entity"],
                "metric": record["metric"],
                "condition": record["condition"],
                "source_path": record["source_path"],
                "page_no": record["page_no"],
            })

    print(f"[llm_rewrite_2] 原始样本数: {len(records)}")
    print(f"[llm_rewrite_2] 失败样本数: {failures}")
    print(f"[llm_rewrite_2] 生成改写样本数: {len(output)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 LLM 对 QA 数据集问题做 3 条受控改写")
    parser.add_argument("--input", default="eval/qa_dataset.json", help="输入 QA 数据集路径")
    parser.add_argument("--output", default="eval/qa_llm_dataset_v2.json", help="输出改写数据集路径")
    parser.add_argument(
        "--model",
        default=os.getenv("CAD_AGENT_DEEPSEEK_MODEL", "deepseek-chat").strip(),
        help="可选：覆盖默认 DeepSeek 模型名",
    )
    parser.add_argument("--limit", type=int, default=None, help="可选：仅处理前 N 条样本")
    parser.add_argument("--start", type=int, default=0, help="可选：从第 start 条样本开始处理")
    parser.add_argument("--max-retries", type=int, default=3, help="单条样本的最大重试次数")
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=1.5,
        help="失败重试前的等待秒数",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = _load_dataset(input_path)
    if args.start:
        records = records[args.start:]
    if args.limit is not None:
        records = records[:args.limit]

    rewritten = build_rewrite_dataset(
        records,
        model=args.model,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )
    output_path.write_text(json.dumps(rewritten, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已保存到: {output_path}")
    if rewritten:
        print("\n示例（前5条）：")
        for item in rewritten[:5]:
            print(f"  [{item['id']}] ({item['difficulty']}) {item['question']}")


if __name__ == "__main__":
    main()
