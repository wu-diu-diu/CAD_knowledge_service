"""
使用 LLM 对 QA 数据集中的问题进行改写，生成独立的新数据集。

默认行为：
- 读取 eval/qa_dataset.json
- 对每条问题调用 DeepSeek 兼容接口
- 生成 5 条改写问题
- 输出到 eval/qa_llm_dataset.json

输出样本字段结构与原 qa_dataset.json 完全一致，仅 id 和 question 不同。

用法：
    python eval/llm_rewrite.py
    python eval/llm_rewrite.py --limit 10
    python eval/llm_rewrite.py --model deepseek-chat --max-retries 4
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
你是建筑规范 QA 数据增强助手。
你的任务是对一个问题做 5 条中文改写，用于检索评测数据扩充。

硬性要求：
1. 必须保持原问题要查询的事实不变，不能改答案，不能改适用条件。
2. 必须生成恰好 5 条不同的改写问题。
3. 改写要更口语化、更像真实用户提问。
4. 允许对场所做合理泛化或模糊替换，例如“高档办公室”可以改写成“办公室”。
5. 允许把专业术语解释成更易懂的说法，例如：
   - Ra -> 显色指数
   - UGR -> 统一眩光值 / 眩光控制值
   - 照度标准值 -> 照度要求 / 亮度要求（仅在不改变原意时）
6. 如果原问题中包含多个并列场所，可以：
   - 拆成单个场所提问；
   - 或改写成“这些区域/这些地方……”的汇总问法。
   但无论如何，语义必须仍对应同一条原始事实记录。
7. 不能输出解释，不能输出答案，不能输出字段说明。
8. 必须返回严格 JSON，格式如下：
{
  "rewrites": [
    "改写1",
    "改写2",
    "改写3",
    "改写4",
    "改写5"
  ]
}
""".strip()


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
    normalized = re.sub(r"\s+", "", (text or "").strip())
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


def _build_user_prompt(record: Dict[str, Any]) -> str:
    condition = record.get("condition")
    condition_text = f"条件: {condition}\n" if condition else "条件: 无\n"
    return (
        "请对下面这条建筑规范 QA 问题生成 5 条改写。\n\n"
        f"原问题: {record['question']}\n"
        f"场所: {record['entity']}\n"
        f"指标: {record['metric']}\n"
        f"{condition_text}"
        f"答案: {record['answer_raw']}\n\n"
        "请确保：\n"
        "- 保持要查询的事实与答案完全不变；\n"
        "- 更口语化；\n"
        "- 可以做合理的场所泛化；\n"
        "- 可以把专业名词解释成更自然的表达；\n"
        "- 如果原问题含多个并列场所，可以拆分或汇总；\n"
        "- 只输出严格 JSON。\n"
    )


def _parse_rewrites(payload: Dict[str, Any], original_question: str) -> List[str]:
    items = payload.get("rewrites")
    if not isinstance(items, list):
        raise ValueError("LLM 返回中缺少 rewrites 列表。")

    seen = set()
    rewrites: List[str] = []
    original_norm = _normalize_question(original_question)
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        norm = _normalize_question(text)
        if not norm or norm == original_norm or norm in seen:
            continue
        seen.add(norm)
        rewrites.append(text)

    if len(rewrites) != 5:
        raise ValueError(f"有效改写数量不是 5，而是 {len(rewrites)}。")
    return rewrites


def rewrite_question(
    client: Any,
    record: Dict[str, Any],
    *,
    model: str,
    max_retries: int,
    retry_sleep: float,
) -> List[str]:
    prompt = _build_user_prompt(record)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=0.9,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = completion.choices[0].message.content or ""
            payload = _json_load_loose(content)
            if payload is None:
                raise ValueError(f"LLM 未返回合法 JSON。原始输出: {content[:500]}")
            return _parse_rewrites(payload, str(record["question"]))
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

    iterator = _iter_with_progress(records, total=len(records), desc="LLM Rewriting")
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
            print(f"[llm_rewrite] 跳过样本 {record['id']}: {exc}")
            continue

        for rewrite in rewrites:
            output.append({
                "id": f"qa_llm_{len(output) + 1:06d}",
                "question": rewrite,
                "answer": record["answer"],
                "answer_raw": record["answer_raw"],
                "requirement_id": record["requirement_id"],
                "entity": record["entity"],
                "metric": record["metric"],
                "condition": record["condition"],
                "source_path": record["source_path"],
                "page_no": record["page_no"],
            })

    print(f"[llm_rewrite] 原始样本数: {len(records)}")
    print(f"[llm_rewrite] 失败样本数: {failures}")
    print(f"[llm_rewrite] 生成改写样本数: {len(output)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 LLM 对 QA 数据集问题做 5 条改写")
    parser.add_argument("--input", default="eval/qa_dataset.json", help="输入 QA 数据集路径")
    parser.add_argument("--output", default="eval/qa_llm_dataset.json", help="输出改写数据集路径")
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
        print("\n示例（前3条）：")
        for item in rewritten[:3]:
            print(f"  [{item['id']}] {item['question']}")


if __name__ == "__main__":
    main()
