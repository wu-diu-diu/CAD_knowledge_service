"""
三条基线方法实现：
  A. BM25 稀疏检索（rank_bm25 + jieba 分词）
  B. 纯向量 RAG（现有 rag_store/index.faiss，chunk 粒度，无图谱）
  C. LLM 直接回答（无检索）
  运行代码：python eval/run_eval.py --kg_store kg_store --dataset eval/qa_dataset.json --top_k 10 --skip_llm
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


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
# 基线 B：纯向量 RAG（chunk 粒度，无图谱）
# ---------------------------------------------------------------------------

class VectorRAGBaseline:
    """
    直接使用现有 rag_store/index.faiss（chunk 粒度）做向量检索，
    不做任何图谱结构过滤。结果不含 requirement_id，无法计算 Recall@K/MRR，
    但可以计算 Exact Match（从 chunk 文本中提取数值）。
    """

    def __init__(
        self,
        rag_store_dir: Path,
        embed_fn: Callable[[List[str]], np.ndarray],
    ) -> None:
        try:
            import faiss as _faiss
        except ImportError as e:
            raise RuntimeError("请安装 faiss-cpu。") from e

        index_path = rag_store_dir / "index.faiss"
        meta_path = rag_store_dir / "metadata.json"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS 索引不存在: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"元数据不存在: {meta_path}")

        self._index = _faiss.read_index(str(index_path))
        self._metadata: List[dict] = json.loads(meta_path.read_text(encoding="utf-8"))
        self._embed_fn = embed_fn

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        q_vec = self._embed_fn([query])
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            rec = dict(self._metadata[idx])
            rec["score"] = float(score)
            # chunk 粒度没有 requirement_id，填 None 以便统一接口
            rec.setdefault("requirement_id", None)
            results.append(rec)
        return results


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
