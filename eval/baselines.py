"""
三条基线方法实现：
  A. BM25 稀疏检索（rank_bm25 + jieba 分词）
  B. 纯向量检索（Requirement 粒度，req_index.faiss）
  C. LLM 直接回答（无检索）
  运行代码：python eval/run_eval.py --kg_store kg_store --dataset eval/qa_dataset.json --top_k 10 --skip_llm
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from kg.vector_store import search_requirements


# ---------------------------------------------------------------------------
# 基线 A：BM25
# ---------------------------------------------------------------------------

class BM25Baseline:
    """
    基于 rank_bm25 + jieba 的稀疏检索基线。
    语料与向量索引相同：所有 Requirement 节点的 search_text。
    """

    def __init__(self, kg_store_dir: Path) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise RuntimeError("请安装 rank-bm25：pip install rank-bm25") from e
        try:
            import jieba
        except ImportError as e:
            raise RuntimeError("请安装 jieba：pip install jieba") from e

        import jieba as _jieba
        from rank_bm25 import BM25Okapi as _BM25

        meta_path = kg_store_dir / "req_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"req_metadata.json 不存在，请先调用 build_requirement_index()。路径: {meta_path}"
            )
        self._metadata: List[dict] = json.loads(meta_path.read_text(encoding="utf-8"))
        tokenized = [list(_jieba.cut(r["search_text"])) for r in self._metadata]
        self._bm25 = _BM25(tokenized)
        self._jieba = _jieba

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        tokens = list(self._jieba.cut(query))
        scores = self._bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            rec = dict(self._metadata[idx])
            rec["score"] = float(scores[idx])
            results.append(rec)
        return results


# ---------------------------------------------------------------------------
# 基线 B：纯向量检索（Requirement 粒度）
# ---------------------------------------------------------------------------

class VectorRAGBaseline:
    """
    直接使用 Requirement 向量索引做纯向量检索，
    不做 BM25 融合，也不做 entity / metric 结构过滤。
    这样可以与 BM25、KG 混合检索在同一 Requirement 粒度上公平对比。
    """

    def __init__(
        self,
        kg_store_dir: Path,
        embed_fn: Callable[[List[str]], np.ndarray],
    ) -> None:
        self._kg_store_dir = kg_store_dir
        self._embed_fn = embed_fn

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return search_requirements(
            query,
            self._kg_store_dir,
            self._embed_fn,
            top_k=top_k,
        )


# ---------------------------------------------------------------------------
# 基线 C：LLM 直接回答（无检索）
# ---------------------------------------------------------------------------

class LLMDirectBaseline:
    """
    不做任何检索，直接把问题发给 LLM，返回答案字符串。
    复用 kg/llm_query.py 中的 _chat_text()。
    """

    SYSTEM_PROMPT = (
        "你是建筑规范专家，请直接回答问题，给出具体数值和单位。"
        "如果不确定，请说'不知道'，不要编造数据。"
    )

    def __init__(self, model: Optional[str] = None) -> None:
        self._model = model or os.getenv("KG_LLM_MODEL", "qwen-plus")

    def answer(self, question: str) -> str:
        # 延迟导入，避免在没有 API key 时报错
        from kg.llm_query import _chat_text
        try:
            return _chat_text(question, self.SYSTEM_PROMPT, self._model)
        except Exception as e:
            return f"[ERROR] {e}"
