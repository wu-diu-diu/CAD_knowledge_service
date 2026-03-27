"""
评测主脚本：对比 BM25、纯向量检索、KG混合检索 三种方法。

用法：
    python eval/run_eval.py \
        --kg_store kg_store \
        --rag_store rag_store \
        --dataset eval/qa_dataset.json \
        --top_k 10 \
        --output eval/results.json \
        [--rebuild]    # 强制重建向量索引
        [--qa_llm_dataset|--qa_all_dataset] [--BM25|--VECTOR|--HYBRID]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# 确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.baselines import BM25Baseline, LLMDirectBaseline, VectorRAGBaseline
from eval.metrics import evaluate_generation, evaluate_retrieval, exact_match
from kg.hybrid_query import hybrid_query
from kg.vector_store import build_requirement_index, requirement_index_exists


def _load_embed_fn():
    """加载 sentence-transformers 模型，返回 embed_fn。"""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    device = os.getenv("RAG_EMBED_DEVICE", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    batch_size = int(os.getenv("RAG_EMBED_BATCH_SIZE", "64"))

    print(f"[eval] 加载 embedding 模型: {model_name} (device={device})")
    model = SentenceTransformer(model_name, device=device)

    def embed_fn(texts: List[str]) -> np.ndarray:
        vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    return embed_fn


def _print_table(rows: List[Dict[str, Any]], top_k: int) -> None:
    headers = ["方法", "Recall@1", "Recall@5", f"Recall@{top_k}", "MRR", "Exact Match"]
    col_w = [max(len(h), 16) for h in headers]
    col_w[0] = 20

    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    header_row = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, col_w)) + "|"

    print(sep)
    print(header_row)
    print(sep)
    for row in rows:
        vals = [
            row.get("method", ""),
            f"{row.get('recall@1', 0):.4f}",
            f"{row.get('recall@5', 0):.4f}",
            f"{row.get(f'recall@{top_k}', 0):.4f}",
            f"{row.get('mrr', 0):.4f}",
            f"{row.get('exact_match', 0):.4f}",
        ]
        print("|" + "|".join(f" {v:<{w}} " for v, w in zip(vals, col_w)) + "|")
    print(sep)


def run_eval(
    kg_store_dir: Path,
    dataset_path: Path,
    top_k: int = 10,
    rebuild: bool = False,
    run_bm25: bool = True,
    run_vector: bool = True,
    run_hybrid: bool = True,
    rrf_k_values: List[int] = (60,),
    limit: int = 0,
) -> List[Dict[str, Any]]:

    # 1. 加载数据集
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_path}，请先运行 eval/dataset_builder.py")
    dataset: List[dict] = json.loads(dataset_path.read_text(encoding="utf-8"))
    if limit and limit < len(dataset):
        import random
        dataset = random.sample(dataset, limit)
        print(f"[eval] 数据集: {len(dataset)} 条 QA 对（随机采样 {limit} 条）")
    else:
        print(f"[eval] 数据集: {len(dataset)} 条 QA 对")

    # 2. 加载 embedding 模型
    embed_fn = _load_embed_fn()

    # 3. 构建/检查 Requirement 向量索引
    if rebuild or not requirement_index_exists(kg_store_dir):
        print("[eval] 构建 Requirement 向量索引...")
        t0 = time.time()
        n = build_requirement_index(kg_store_dir, embed_fn)
        print(f"[eval] 索引构建完成，共 {n} 条 Requirement，耗时 {time.time()-t0:.1f}s")
    else:
        print("[eval] 使用已有 Requirement 向量索引")

    all_results: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # 方法 1：BM25
    # -----------------------------------------------------------------------
    if run_bm25:
        print("\n[eval] 评测 BM25...")
        try:
            bm25 = BM25Baseline(kg_store_dir)
            bm25_metrics = evaluate_retrieval(
                lambda q: bm25.search(q, top_k=top_k),
                dataset,
                top_k=top_k,
                show_progress=True,
                progress_desc="BM25 Retrieval",
            )

            def bm25_answer(q: str) -> str:
                res = bm25.search(q, top_k=1)
                vals = res[0].get("values", [])
                return vals[0] if vals else ""

            bm25_em = evaluate_generation(
                bm25_answer,
                dataset,
                show_progress=True,
                progress_desc="BM25 Exact Match",
            )
            bm25_metrics.update(bm25_em)
            # bm25_metrics.update(
            #     _evaluate_breakdowns(
            #         dataset,
            #         top_k,
            #         retrieval_fn=lambda q: bm25.search(q, top_k=top_k),
            #         answer_fn=bm25_answer,
            #     )
            # )
            bm25_metrics["method"] = "BM25"
            all_results.append(bm25_metrics)
            print(f"  Recall@1={bm25_metrics['recall@1']:.4f}  MRR={bm25_metrics['mrr']:.4f}")
            # _print_slice_summary("BM25", bm25_metrics["by_semantic_slice"], top_k)
        except Exception as e:
            print(f"  [跳过] BM25 初始化失败: {e}")

    # -----------------------------------------------------------------------
    # 方法 2：纯向量检索（Requirement 粒度）
    # -----------------------------------------------------------------------
    if run_vector:
        print("\n[eval] 评测纯向量检索（Requirement 粒度）...")
        try:
            rag = VectorRAGBaseline(kg_store_dir, embed_fn)

            def rag_answer(q: str) -> str:
                res = rag.search(q, top_k=1)
                return res[0]["values"][0] if res and res[0].get("values") else ""

            rag_metrics = evaluate_retrieval(
                lambda q: rag.search(q, top_k=top_k),
                dataset,
                top_k=top_k,
                show_progress=True,
                progress_desc="Vector Retrieval",
            )
            rag_em = evaluate_generation(
                rag_answer,
                dataset,
                show_progress=True,
                progress_desc="Vector Exact Match",
            )
            rag_metrics.update(rag_em)
            # rag_metrics.update(
            #     _evaluate_breakdowns(
            #         dataset,
            #         top_k,
            #         retrieval_fn=lambda q: rag.search(q, top_k=top_k),
            #         answer_fn=rag_answer,
            #     )
            # )
            rag_metrics["method"] = "纯向量检索(Req粒度)"
            all_results.append(rag_metrics)
            print(f"  Recall@1={rag_metrics['recall@1']:.4f}  MRR={rag_metrics['mrr']:.4f}")
            # _print_slice_summary("纯向量检索(Req粒度)", rag_metrics["by_semantic_slice"], top_k)
        except Exception as e:
            print(f"  [跳过] 纯向量检索初始化失败: {e}")

    # -----------------------------------------------------------------------
    # 方法 3：KG BM25+向量混合检索（Requirement 粒度），支持多个 rrf_k 值
    # -----------------------------------------------------------------------
    if run_hybrid:
        for rrf_k in rrf_k_values:
            print(f"\n[eval] 评测 KG BM25+向量混合检索（rrf_k={rrf_k}）...")

            def kg_vec_search(q: str, _k: int = rrf_k) -> List[dict]:
                result = hybrid_query(q, kg_store_dir, embed_fn, top_k=top_k, rrf_k=_k)
                return result.get("results", [])

            kg_vec_metrics = evaluate_retrieval(
                kg_vec_search,
                dataset,
                top_k=top_k,
                show_progress=True,
                progress_desc=f"Hybrid(k={rrf_k}) Retrieval",
            )

            def kg_vec_answer(q: str, _k: int = rrf_k) -> str:
                res = kg_vec_search(q, _k)
                return res[0]["values"][0] if res and res[0].get("values") else ""

            kg_vec_em = evaluate_generation(
                kg_vec_answer,
                dataset,
                show_progress=True,
                progress_desc=f"Hybrid(k={rrf_k}) Exact Match",
            )
            kg_vec_metrics.update(kg_vec_em)
            kg_vec_metrics["method"] = f"KG混合检索(rrf_k={rrf_k})"
            all_results.append(kg_vec_metrics)
            print(f"  Recall@1={kg_vec_metrics['recall@1']:.4f}  MRR={kg_vec_metrics['mrr']:.4f}")
        # _print_slice_summary("KG BM25+向量混合检索(Req粒度)", kg_vec_metrics["by_semantic_slice"], top_k)

    # -----------------------------------------------------------------------
    # 方法 4：LLM 直接回答（可选）
    # -----------------------------------------------------------------------
    # if not skip_llm:
    #     print("\n[eval] 评测 LLM 直接回答...")
    #     try:
    #         llm = LLMDirectBaseline()
    #         llm_em = evaluate_generation(llm.answer, dataset)
    #         llm_metrics: Dict[str, Any] = {
    #             "method": "LLM直接回答",
    #             "recall@1": 0.0,
    #             "recall@5": 0.0,
    #             f"recall@{top_k}": 0.0,
    #             "mrr": 0.0,
    #         }
    #         llm_metrics.update(llm_em)
    #         llm_metrics.update(
    #             _evaluate_breakdowns(
    #                 dataset,
    #                 top_k,
    #                 retrieval_fn=None,
    #                 answer_fn=llm.answer,
    #             )
    #         )
    #         all_results.append(llm_metrics)
    #         print(f"  EM={llm_metrics['exact_match']:.4f}")
    #         _print_slice_summary("LLM直接回答", llm_metrics["by_semantic_slice"], top_k)
    #     except Exception as e:
    #         print(f"  [跳过] LLM 基线失败: {e}")
    # else:
    #     print("\n[eval] 跳过 LLM 直接回答基线（--skip_llm）")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="KG 检索方法评测")
    parser.add_argument("--kg_store", default="kg_store")
    parser.add_argument("--rag_store", default="rag_store")
    parser.add_argument("--dataset", default="eval/qa_dataset.json")
    parser.add_argument("--qa_dataset", action="store_true", help="使用 eval/qa_dataset.json 作为数据集")
    parser.add_argument("--qa_llm_dataset", action="store_true", help="使用 eval/qa_llm_dataset.json 作为数据集")
    parser.add_argument("--qa_all_dataset", action="store_true", help="使用 eval/qa_all_dataset.json 作为数据集")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output", default="eval/results.json")
    parser.add_argument("--skip_llm", default=False, action="store_true", help="跳过 LLM 直接回答基线")
    parser.add_argument("--rebuild", default=False, action="store_true", help="强制重建 Requirement 向量索引")
    parser.add_argument("--BM25", dest="run_bm25", default=False, action="store_true", help="只选择 BM25 方法之一")
    parser.add_argument("--VECTOR", dest="run_vector", default=False, action="store_true", help="只选择纯向量 RAG 方法之一")
    parser.add_argument("--HYBRID", dest="run_hybrid", default=False, action="store_true", help="只选择 KG 混合检索方法之一")
    parser.add_argument("--rrf_k", type=int, nargs="+", default=[60], help="RRF 的 k 值，可传多个，如 --rrf_k 10 30 60 100")
    parser.add_argument("--limit", type=int, default=0, help="随机采样数据集条数，0 表示使用全部")
    args = parser.parse_args()

    chosen_dataset_flags = sum(bool(x) for x in [args.qa_dataset, args.qa_llm_dataset, args.qa_all_dataset])
    if chosen_dataset_flags > 1:
        raise ValueError("--qa_dataset、--qa_llm_dataset、--qa_all_dataset 只能指定一个。")

    dataset_path = Path(args.dataset)
    if args.qa_dataset:
        dataset_path = Path("eval/qa_dataset.json")
    elif args.qa_llm_dataset:
        dataset_path = Path("eval/qa_llm_dataset.json")
    elif args.qa_all_dataset:
        dataset_path = Path("eval/qa_all_dataset.json")

    selected_any_method = args.run_bm25 or args.run_vector or args.run_hybrid
    run_bm25 = args.run_bm25 or not selected_any_method
    run_vector = args.run_vector or not selected_any_method
    run_hybrid = args.run_hybrid or not selected_any_method

    results = run_eval(
        kg_store_dir=Path(args.kg_store),
        dataset_path=dataset_path,
        top_k=args.top_k,
        rebuild=args.rebuild,
        run_bm25=run_bm25,
        run_vector=run_vector,
        run_hybrid=run_hybrid,
        rrf_k_values=args.rrf_k,
        limit=args.limit,
    )

    print("\n" + "=" * 70)
    print("评测结果汇总")
    print("=" * 70)
    _print_table(results, args.top_k)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
