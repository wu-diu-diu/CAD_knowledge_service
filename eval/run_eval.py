"""
评测主脚本：对比 BM25、纯向量检索、KG混合检索、混合+图扩展 四种方法。

用法：
    python eval/run_eval.py \
        --kg_store kg_store \
        --dataset eval/qa_dataset.json \
        --top_k 10 \
        [--rebuild]
        [--qa_llm_dataset|--qa_all_dataset|--qa_multihop_dataset]
        [--BM25|--VECTOR|--HYBRID|--GRAPH]
        [--multihop]   # 启用多跳评测模式（Full Recall@K）

多跳数据集上测试：python eval/run_eval.py --qa_multihop_dataset --BM25 --HYBRID --GRAPH --multihop
单跳数据集上测评：python eval/run_eval.py --qa_dataset --BM25 --HYBRID --GRAPH
全部数据集上测评：python eval/run_eval.py --qa_all_dataset --BM25 --HYBRID --GRAPH
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
from kg.hybrid_graph_query import hybrid_graph_query
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


def _print_multihop_table(rows: List[Dict[str, Any]], top_k: int) -> None:
    """多跳评测专用表格：Full Recall@K + Partial Recall@K。"""
    headers = ["方法", f"Full R@{top_k}", f"Partial R@{top_k}", "Avg Coverage"]
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
            f"{row.get(f'full_recall@{top_k}', 0):.4f}",
            f"{row.get(f'partial_recall@{top_k}', 0):.4f}",
            f"{row.get('avg_coverage', 0):.4f}",
        ]
        print("|" + "|".join(f" {v:<{w}} " for v, w in zip(vals, col_w)) + "|")
    print(sep)


def evaluate_multihop_retrieval(
    method_fn,
    dataset: List[Dict[str, Any]],
    top_k: int = 10,
    show_progress: bool = False,
    progress_desc: str = "",
) -> Dict[str, float]:
    """
    多跳检索评测：每条样本有多个 gold requirement_ids。

    - Full Recall@K：top-K 中包含所有 gold requirement_ids 才算命中
    - Partial Recall@K：top-K 中至少包含一个 gold requirement_id 算命中
    - Avg Coverage：平均每条样本召回了多少比例的 gold requirement_ids
    """
    n = len(dataset)
    if n == 0:
        return {f"full_recall@{top_k}": 0.0, f"partial_recall@{top_k}": 0.0, "avg_coverage": 0.0}

    iterator = dataset
    if show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(dataset, total=n, desc=progress_desc or "MultiHop Retrieval", leave=False)
        except Exception:
            pass

    full_hit = partial_hit = coverage_sum = 0.0
    for item in iterator:
        gold_ids = set(item.get("requirement_ids", []))
        if not gold_ids:
            continue
        results = method_fn(item["question"])
        # 不截断：图扩展方法返回的结果数 > top_k，截断会丢失扩展节点
        retrieved_ids = {r.get("requirement_id") for r in results}
        matched = gold_ids & retrieved_ids
        coverage = len(matched) / len(gold_ids)
        coverage_sum += coverage
        if len(matched) == len(gold_ids):
            full_hit += 1.0
        if len(matched) >= 1:
            partial_hit += 1.0

    return {
        f"full_recall@{top_k}": round(full_hit / n, 4),
        f"partial_recall@{top_k}": round(partial_hit / n, 4),
        "avg_coverage": round(coverage_sum / n, 4),
    }


def run_eval(
    kg_store_dir: Path,
    dataset_path: Path,
    top_k: int = 10,
    rebuild: bool = False,
    run_bm25: bool = True,
    run_vector: bool = True,
    run_hybrid: bool = True,
    run_graph: bool = False,
    rrf_k_values: List[int] = (60,),
    limit: int = 0,
    multihop: bool = False,
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
            if multihop:
                bm25_metrics = evaluate_multihop_retrieval(
                    lambda q: bm25.search(q, top_k=top_k),
                    dataset, top_k=top_k, show_progress=True, progress_desc="BM25 MultiHop",
                )
                bm25_metrics["method"] = "BM25"
                all_results.append(bm25_metrics)
                print(f"  Full R@{top_k}={bm25_metrics[f'full_recall@{top_k}']:.4f}  "
                      f"Partial R@{top_k}={bm25_metrics[f'partial_recall@{top_k}']:.4f}  "
                      f"Avg Coverage={bm25_metrics['avg_coverage']:.4f}")
            else:
                bm25_metrics = evaluate_retrieval(
                    lambda q: bm25.search(q, top_k=top_k),
                    dataset, top_k=top_k, show_progress=True, progress_desc="BM25 Retrieval",
                )

                def bm25_answer(q: str) -> str:
                    res = bm25.search(q, top_k=1)
                    vals = res[0].get("values", [])
                    return vals[0] if vals else ""

                bm25_em = evaluate_generation(
                    bm25_answer, dataset, show_progress=True, progress_desc="BM25 Exact Match",
                )
                bm25_metrics.update(bm25_em)
                bm25_metrics["method"] = "BM25"
                all_results.append(bm25_metrics)
                print(f"  Recall@1={bm25_metrics['recall@1']:.4f}  MRR={bm25_metrics['mrr']:.4f}")
        except Exception as e:
            print(f"  [跳过] BM25 失败: {e}")

    # -----------------------------------------------------------------------
    # 方法 2：纯向量检索（Requirement 粒度）
    # -----------------------------------------------------------------------
    if run_vector:
        print("\n[eval] 评测纯向量检索（Requirement 粒度）...")
        try:
            rag = VectorRAGBaseline(kg_store_dir, embed_fn)
            if multihop:
                rag_metrics = evaluate_multihop_retrieval(
                    lambda q: rag.search(q, top_k=top_k),
                    dataset, top_k=top_k, show_progress=True, progress_desc="Vector MultiHop",
                )
                rag_metrics["method"] = "纯向量检索(Req粒度)"
                all_results.append(rag_metrics)
                print(f"  Full R@{top_k}={rag_metrics[f'full_recall@{top_k}']:.4f}  "
                      f"Partial R@{top_k}={rag_metrics[f'partial_recall@{top_k}']:.4f}  "
                      f"Avg Coverage={rag_metrics['avg_coverage']:.4f}")
            else:
                def rag_answer(q: str) -> str:
                    res = rag.search(q, top_k=1)
                    return res[0]["values"][0] if res and res[0].get("values") else ""

                rag_metrics = evaluate_retrieval(
                    lambda q: rag.search(q, top_k=top_k),
                    dataset, top_k=top_k, show_progress=True, progress_desc="Vector Retrieval",
                )
                rag_em = evaluate_generation(
                    rag_answer, dataset, show_progress=True, progress_desc="Vector Exact Match",
                )
                rag_metrics.update(rag_em)
                rag_metrics["method"] = "纯向量检索(Req粒度)"
                all_results.append(rag_metrics)
                print(f"  Recall@1={rag_metrics['recall@1']:.4f}  MRR={rag_metrics['mrr']:.4f}")
        except Exception as e:
            print(f"  [跳过] 纯向量检索失败: {e}")

    # -----------------------------------------------------------------------
    # 方法 3：KG BM25+向量混合检索，支持多个 rrf_k 值
    # -----------------------------------------------------------------------
    if run_hybrid:
        for rrf_k in rrf_k_values:
            print(f"\n[eval] 评测 KG BM25+向量混合检索（rrf_k={rrf_k}）...")

            def kg_vec_search(q: str, _k: int = rrf_k) -> List[dict]:
                result = hybrid_query(q, kg_store_dir, embed_fn, top_k=top_k, rrf_k=_k)
                return result.get("results", [])

            if multihop:
                kg_vec_metrics = evaluate_multihop_retrieval(
                    kg_vec_search, dataset, top_k=top_k, show_progress=True,
                    progress_desc=f"Hybrid(k={rrf_k}) MultiHop",
                )
                kg_vec_metrics["method"] = f"KG混合检索(rrf_k={rrf_k})"
                all_results.append(kg_vec_metrics)
                print(f"  Full R@{top_k}={kg_vec_metrics[f'full_recall@{top_k}']:.4f}  "
                      f"Partial R@{top_k}={kg_vec_metrics[f'partial_recall@{top_k}']:.4f}  "
                      f"Avg Coverage={kg_vec_metrics['avg_coverage']:.4f}")
            else:
                kg_vec_metrics = evaluate_retrieval(
                    kg_vec_search, dataset, top_k=top_k, show_progress=True,
                    progress_desc=f"Hybrid(k={rrf_k}) Retrieval",
                )

                def kg_vec_answer(q: str, _k: int = rrf_k) -> str:
                    res = kg_vec_search(q, _k)
                    return res[0]["values"][0] if res and res[0].get("values") else ""

                kg_vec_em = evaluate_generation(
                    kg_vec_answer, dataset, show_progress=True,
                    progress_desc=f"Hybrid(k={rrf_k}) Exact Match",
                )
                kg_vec_metrics.update(kg_vec_em)
                kg_vec_metrics["method"] = f"KG混合检索(rrf_k={rrf_k})"
                all_results.append(kg_vec_metrics)
                print(f"  Recall@1={kg_vec_metrics['recall@1']:.4f}  MRR={kg_vec_metrics['mrr']:.4f}")

    # -----------------------------------------------------------------------
    # 方法 4：混合检索 + 图邻域扩展
    # -----------------------------------------------------------------------
    if run_graph:
        for rrf_k in rrf_k_values:
            print(f"\n[eval] 评测 混合+图扩展（rrf_k={rrf_k}）...")

            def graph_search(q: str, _k: int = rrf_k) -> List[dict]:
                result = hybrid_graph_query(q, kg_store_dir, embed_fn, top_k=top_k, rrf_k=_k)
                return result.get("results", [])

            if multihop:
                graph_mh = evaluate_multihop_retrieval(
                    graph_search,
                    dataset,
                    top_k=top_k,
                    show_progress=True,
                    progress_desc=f"Graph(k={rrf_k}) MultiHop",
                )
                graph_mh["method"] = f"混合+图扩展(rrf_k={rrf_k})"
                all_results.append(graph_mh)
                print(f"  Full R@{top_k}={graph_mh[f'full_recall@{top_k}']:.4f}  "
                      f"Partial R@{top_k}={graph_mh[f'partial_recall@{top_k}']:.4f}  "
                      f"Avg Coverage={graph_mh['avg_coverage']:.4f}")
            else:
                graph_metrics = evaluate_retrieval(
                    graph_search,
                    dataset,
                    top_k=top_k,
                    show_progress=True,
                    progress_desc=f"Graph(k={rrf_k}) Retrieval",
                )

                def graph_answer(q: str, _k: int = rrf_k) -> str:
                    res = graph_search(q, _k)
                    return res[0]["values"][0] if res and res[0].get("values") else ""

                graph_em = evaluate_generation(
                    graph_answer,
                    dataset,
                    show_progress=True,
                    progress_desc=f"Graph(k={rrf_k}) Exact Match",
                )
                graph_metrics.update(graph_em)
                graph_metrics["method"] = f"混合+图扩展(rrf_k={rrf_k})"
                all_results.append(graph_metrics)
                print(f"  Recall@1={graph_metrics['recall@1']:.4f}  MRR={graph_metrics['mrr']:.4f}")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="KG 检索方法评测")
    parser.add_argument("--kg_store", default="kg_store")
    parser.add_argument("--rag_store", default="rag_store")
    parser.add_argument("--dataset", default="eval/qa_dataset.json")
    parser.add_argument("--qa_dataset", action="store_true", help="使用 eval/qa_dataset.json")
    parser.add_argument("--qa_llm_dataset", action="store_true", help="使用 eval/qa_llm_dataset.json")
    parser.add_argument("--qa_all_dataset", action="store_true", help="使用 eval/qa_all_dataset.json")
    parser.add_argument("--qa_multihop_dataset", action="store_true", help="使用 eval/qa_multihop_dataset.json（多跳评测）")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output", default="eval/results.json")
    parser.add_argument("--rebuild", default=False, action="store_true", help="强制重建 Requirement 向量索引")
    parser.add_argument("--BM25", dest="run_bm25", default=False, action="store_true")
    parser.add_argument("--VECTOR", dest="run_vector", default=False, action="store_true")
    parser.add_argument("--HYBRID", dest="run_hybrid", default=False, action="store_true")
    parser.add_argument("--GRAPH", dest="run_graph", default=False, action="store_true", help="混合+图邻域扩展")
    parser.add_argument("--rrf_k", type=int, nargs="+", default=[60], help="RRF k 值，可传多个")
    parser.add_argument("--limit", type=int, default=0, help="随机采样条数，0 表示全部")
    parser.add_argument("--multihop", default=False, action="store_true", help="启用多跳评测模式（Full/Partial Recall@K）")
    args = parser.parse_args()

    chosen_dataset_flags = sum(bool(x) for x in [
        args.qa_dataset, args.qa_llm_dataset, args.qa_all_dataset, args.qa_multihop_dataset
    ])
    if chosen_dataset_flags > 1:
        raise ValueError("数据集参数只能指定一个。")

    dataset_path = Path(args.dataset)
    if args.qa_dataset:
        dataset_path = Path("eval/dataset/qa_dataset.json")
    elif args.qa_llm_dataset:
        dataset_path = Path("eval/dataset/qa_llm_dataset.json")
    elif args.qa_all_dataset:
        dataset_path = Path("eval/dataset/qa_all_dataset.json")
    elif args.qa_multihop_dataset:
        dataset_path = Path("eval/qa_multihop_test.json")

    # --qa_multihop_dataset 自动开启 multihop 模式
    multihop = args.multihop or args.qa_multihop_dataset

    selected_any_method = args.run_bm25 or args.run_vector or args.run_hybrid or args.run_graph
    run_bm25 = args.run_bm25 or not selected_any_method
    run_vector = args.run_vector or not selected_any_method
    run_hybrid = args.run_hybrid or not selected_any_method
    run_graph = args.run_graph

    results = run_eval(
        kg_store_dir=Path(args.kg_store),
        dataset_path=dataset_path,
        top_k=args.top_k,
        rebuild=args.rebuild,
        run_bm25=run_bm25,
        run_vector=run_vector,
        run_hybrid=run_hybrid,
        run_graph=run_graph,
        rrf_k_values=args.rrf_k,
        limit=args.limit,
        multihop=multihop,
    )

    print("\n" + "=" * 70)
    print("评测结果汇总")
    print("=" * 70)
    if multihop:
        _print_multihop_table(results, args.top_k)
    else:
        _print_table(results, args.top_k)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
