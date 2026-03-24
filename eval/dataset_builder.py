"""
从知识图谱自动生成 QA 评测数据集。

只处理 table_cell_constraint 类型的 Requirement，因为这类节点有明确的
entity + metric + value，适合生成精确问答。

用法：
    python eval/dataset_builder.py --kg_store kg_store --output eval/qa_dataset.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _iter_graph_docs(store_dir: Path):
    docs_dir = store_dir / "doc_graphs"
    if not docs_dir.exists():
        return
    for path in sorted(docs_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            yield path, data


# 无意义的 metric 名称模式（正则），匹配到则跳过
_NOISE_METRIC_RE = re.compile(
    r"^(序号|编号|Z/X|备注|说明|类别|类型|名称|项目|项|列\d+)$", re.I
)

# 有实际规范意义的 metric 关键词（包含任意一个则保留）
_VALID_METRIC_KEYWORDS = [
    "照度", "Ra", "UGR", "U0", "功率密度", "色温", "GR", "显色",
    "眩光", "照明", "亮度", "光通量", "功率因数", "效率", "寿命",
]


def _is_valid_metric(metric: str) -> bool:
    if not metric:
        return False
    if _NOISE_METRIC_RE.match(metric.strip()):
        return False
    # 跳过重复列后缀 (#2, #3...)，只保留第一列
    if re.search(r"#\d+$", metric.strip()):
        return False
    return any(kw in metric for kw in _VALID_METRIC_KEYWORDS)


def _has_digit(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


def _extract_numeric(text: str) -> Optional[str]:
    """提取文本中的第一个数值（含小数、范围），用于 Exact Match 比较。"""
    m = re.search(r"\d+(?:\.\d+)?(?:\s*[~\-]\s*\d+(?:\.\d+)?)?", (text or ""))
    return m.group(0).strip() if m else None


def build_qa_dataset(kg_store_dir: Path) -> List[Dict[str, Any]]:
    """
    遍历图谱，生成 QA 对列表。

    每条 QA 对格式：
    {
        "id": "qa_001",
        "question": "办公室的照度标准值是多少？",
        "answer": "300",          # 提取的数值，用于 Exact Match
        "answer_raw": "300lx",    # 原始单元格文本
        "requirement_id": "req:...",
        "entity": "办公室",
        "metric": "照度标准值",
        "condition": "...",       # 可选，子场景条件
        "source_path": "...",
        "page_no": 15,
    }
    """
    records: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()  # (entity, metric) 去重

    for _, doc in _iter_graph_docs(kg_store_dir):
        node_map: Dict[str, dict] = {
            n["id"]: n for n in doc.get("nodes", []) if isinstance(n, dict)
        }
        edges = [e for e in doc.get("edges", []) if isinstance(e, dict)]
        out_map: Dict[str, List[dict]] = {}
        for e in edges:
            out_map.setdefault(e.get("source", ""), []).append(e)

        for node in node_map.values():
            if node.get("label") != "Requirement":
                continue
            props = node.get("props") or {}
            if props.get("requirement_type") != "table_cell_constraint":
                continue

            req_id = node["id"]
            raw_cell = str(props.get("raw_cell") or "")
            if not _has_digit(raw_cell):
                continue

            page_no = props.get("page_no")
            source_path = str(props.get("source_path") or doc.get("source_path", ""))

            entity_name: Optional[str] = None
            metric_name: Optional[str] = None
            value_raw: Optional[str] = None
            condition_text: Optional[str] = None

            for e in out_map.get(req_id, []):
                etype = e.get("type")
                target = node_map.get(e.get("target", ""))
                if not target:
                    continue
                tlabel = target.get("label")
                tprops = target.get("props") or {}
                if etype == "APPLIES_TO" and tlabel == "DomainEntity":
                    entity_name = str(tprops.get("canonical_name") or tprops.get("name") or "")
                elif etype == "CONSTRAINS_METRIC" and tlabel == "Metric":
                    metric_name = str(tprops.get("canonical_name") or tprops.get("name") or "")
                elif etype == "HAS_VALUE_SPEC" and tlabel == "ValueSpec":
                    value_raw = str(tprops.get("raw_text") or tprops.get("value") or "")
                elif etype == "UNDER_CONDITION" and tlabel == "Condition":
                    condition_text = str(tprops.get("text") or "")

            if not entity_name or not metric_name or not value_raw:
                continue
            if not _has_digit(value_raw):
                continue
            if not _is_valid_metric(metric_name):
                continue

            dedup_key = (entity_name, metric_name)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            question = f"{entity_name}的{metric_name}是多少？"
            if condition_text:
                question = f"{entity_name}（{condition_text}）的{metric_name}是多少？"

            records.append({
                "id": f"qa_{len(records) + 1:04d}",
                "question": question,
                "answer": _extract_numeric(value_raw) or value_raw,
                "answer_raw": value_raw,
                "requirement_id": req_id,
                "entity": entity_name,
                "metric": metric_name,
                "condition": condition_text,
                "source_path": source_path,
                "page_no": page_no,
            })

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="从知识图谱自动生成 QA 评测数据集")
    parser.add_argument("--kg_store", default="kg_store", help="kg_store 目录路径")
    parser.add_argument("--output", default="eval/qa_dataset.json", help="输出 JSON 文件路径")
    args = parser.parse_args()

    kg_store_dir = Path(args.kg_store)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = build_qa_dataset(kg_store_dir)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"生成 QA 对: {len(records)} 条")
    print(f"已保存到: {output_path}")

    if records:
        print("\n示例（前3条）：")
        for r in records[:3]:
            print(f"  [{r['id']}] Q: {r['question']}")
            print(f"         A: {r['answer_raw']}  (数值: {r['answer']})")


if __name__ == "__main__":
    main()
