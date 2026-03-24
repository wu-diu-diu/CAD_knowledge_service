"""
评测指标：Recall@K、MRR、Exact Match。
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional


def _extract_numeric(text: str) -> Optional[str]:
    """提取文本中第一个数值（含小数、范围），去除空格后返回。"""
    m = re.search(r"\d+(?:\.\d+)?(?:\s*[~\-]\s*\d+(?:\.\d+)?)?", (text or ""))
    return m.group(0).replace(" ", "") if m else None


def recall_at_k(results: List[Dict[str, Any]], gold_req_id: str, k: int) -> float:
    """top-K 结果中是否包含正确的 requirement_id。"""
    for r in results[:k]:
        if r.get("requirement_id") == gold_req_id:
            return 1.0
    return 0.0


def mrr(results: List[Dict[str, Any]], gold_req_id: str) -> float:
    """正确答案排在第几位的倒数（从1开始计）。"""
    for rank, r in enumerate(results, start=1):
        if r.get("requirement_id") == gold_req_id:
            return 1.0 / rank
    return 0.0


def exact_match(predicted: str, gold: str) -> float:
    """
    提取两个字符串中的数值后比较。
    "300lx" vs "300" → 1.0
    "300" vs "500"   → 0.0
    """
    pred_num = _extract_numeric(predicted or "")
    gold_num = _extract_numeric(gold or "")
    if pred_num is None or gold_num is None:
        return 1.0 if (predicted or "").strip() == (gold or "").strip() else 0.0
    return 1.0 if pred_num == gold_num else 0.0


def evaluate_retrieval(
    method_fn: Callable[[str], List[Dict[str, Any]]],
    dataset: List[Dict[str, Any]],
    top_k: int = 10,
) -> Dict[str, float]:
    """
    对检索方法在数据集上计算 Recall@1/5/K 和 MRR。

    method_fn: 接受 question 字符串，返回结果列表（每条含 requirement_id）
    dataset:   QA 数据集，每条含 requirement_id（gold）
    """
    r1 = r5 = rk = mrr_sum = 0.0
    n = len(dataset)
    if n == 0:
        return {"recall@1": 0.0, "recall@5": 0.0, f"recall@{top_k}": 0.0, "mrr": 0.0}

    for item in dataset:
        gold_id = item["requirement_id"]
        results = method_fn(item["question"])
        r1 += recall_at_k(results, gold_id, 1)
        r5 += recall_at_k(results, gold_id, 5)
        rk += recall_at_k(results, gold_id, top_k)
        mrr_sum += mrr(results, gold_id)

    return {
        "recall@1": round(r1 / n, 4),
        "recall@5": round(r5 / n, 4),
        f"recall@{top_k}": round(rk / n, 4),
        "mrr": round(mrr_sum / n, 4),
    }


def evaluate_generation(
    method_fn: Callable[[str], str],
    dataset: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    对生成方法（直接返回答案字符串）计算 Exact Match。

    method_fn: 接受 question 字符串，返回答案字符串
    """
    n = len(dataset)
    if n == 0:
        return {"exact_match": 0.0}
    em_sum = sum(exact_match(method_fn(item["question"]), item["answer"]) for item in dataset)
    return {"exact_match": round(em_sum / n, 4)}
