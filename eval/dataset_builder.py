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


_NOISE_METRIC_RE = re.compile(
    r"^(序号|编号|Z/X|备注|说明|类别|类型|名称|项目|项|列\d+)$", re.I
)

_NOISE_ENTITY_RE = re.compile(
    r"^(序号|编号|备注|说明|类别|类型|名称|项目|项|规格|型号|参数|要求|指标|值|标准|内容|条件|场景|功能|"
    r"表\d*|图\d*|附录|章节?)$",
    re.I,
)

_NON_PLACE_ENTITY_HINT_RE = re.compile(
    r"(规格|型号|参数|灯具|光源|屏幕分类|分类|等级|功率|尺寸|色温|显色|亮度|性能|"
    r"PAR\d+|LED|T\d|金卤灯|高压钠灯|荧光灯|白炽灯)",
    re.I,
)

_PLACE_ENTITY_HINT_RE = re.compile(
    r"(房|室|厅|堂|间|区|区域|空间|通道|走廊|廊|楼梯|梯间|扶梯|平台|门厅|站厅|站台|"
    r"候车室|候诊室|车库|库房|仓库|书库|厨房|餐厅|居室|起居室|卧室|卫生间|厕所|盥洗室|浴室|"
    r"办公室|教室|阅览室|实验室|化验室|休息室|休息厅|会议室|洽谈室|报告厅|多功能厅|宴会厅|"
    r"展厅|商店|超市|营业厅|大堂|大厅|客房|宿舍|病房|诊室|控制室|主控室|调度室|机房|"
    r"泵房|风机房|配电站|变电站|站|厂房|车间|厂区|广场|道路|乐园|活动室|服务中心|"
    r"用房|商业街|市场|出入口|入口|出口|换乘厅|连接区|作业区)",
    re.I,
)

_NON_PLACE_FRAGMENT_RE = re.compile(
    r"(背景|字体|图像|屏幕|展品|设备顶部|书架|售票台|服务台|阀门|压缩机|电操作柱|"
    r"元器件|石质器物|玻璃制品|岩矿标本|竹木制品|动物标本)",
    re.I,
)

_VALID_METRIC_KEYWORDS = [
    "照度", "Ra", "UGR", "U0", "功率密度", "色温", "GR", "显色",
    "眩光", "照明", "亮度", "光通量", "功率因数", "效率", "寿命",
]


def _is_valid_metric(metric: str) -> bool:
    if not metric:
        return False
    if _NOISE_METRIC_RE.match(metric.strip()):
        return False
    if re.search(r"#\d+$", metric.strip()):
        return False
    return any(kw in metric for kw in _VALID_METRIC_KEYWORDS)


def _is_valid_entity_rule(entity: str) -> bool:
    """
    规则层的快速过滤。

    这里只拦截明显不可能作为问句主体的内容：
    - 纯数字 / 比较符号 / 范围表达，如 >5、220-240
    - 泛化表头词，如 规格、参数、类型
    - 没有中文或英文名称成分的碎片
    """
    if not entity:
        return False

    entity = entity.strip()
    if not entity:
        return False
    if _NOISE_ENTITY_RE.match(entity):
        return False
    if re.search(r"#\d+$", entity):
        return False
    if re.fullmatch(r"[<>＝=≤≥~\-–—+\d\.\s%/]+", entity):
        return False
    if re.match(r"^[<>＝=≤≥~\-–—+]?\s*\d", entity):
        return False
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", entity):
        return False
    if _NON_PLACE_ENTITY_HINT_RE.search(entity):
        return False
    if _NON_PLACE_FRAGMENT_RE.search(entity):
        return False
    if not _PLACE_ENTITY_HINT_RE.search(entity):
        return False
    return True


def _has_digit(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


def _extract_numeric(text: str) -> Optional[str]:
    """提取文本中的第一个数值（含小数、范围），用于 Exact Match 比较。"""
    m = re.search(r"\d+(?:\.\d+)?(?:\s*[~\-]\s*\d+(?:\.\d+)?)?", (text or ""))
    return m.group(0).strip() if m else None


def _load_entity_filter_cache(cache_path: Path) -> Dict[str, bool]:
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): bool(v) for k, v in payload.items()}


def _save_entity_filter_cache(cache_path: Path, cache: Dict[str, bool]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(dict(sorted(cache.items())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _llm_filter_entities(
    entities: List[str],
    *,
    cache_path: Optional[Path] = None,
    model: Optional[str] = None,
    batch_size: int = 50,
) -> Dict[str, bool]:
    """
    使用 LLM 批量判别 entity 是否适合作为 QA 问句主体。

    返回: {entity_text: is_valid}
    """
    if not entities:
        return {}

    cache: Dict[str, bool] = {}
    if cache_path is not None:
        cache = _load_entity_filter_cache(cache_path)

    result: Dict[str, bool] = {}
    unique_entities = sorted({(e or "").strip() for e in entities if (e or "").strip()})

    # 明显垃圾项直接用规则淘汰，减少 LLM 调用量。
    pending: List[str] = []
    for entity in unique_entities:
        if not _is_valid_entity_rule(entity):
            result[entity] = False
            cache[entity] = False
            continue
        if entity in cache:
            result[entity] = cache[entity]
            continue
        pending.append(entity)

    if not pending:
        if cache_path is not None:
            _save_entity_filter_cache(cache_path, cache)
        return result

    try:
        from kg.llm_query import _chat_json
    except Exception as exc:
        print(f"[dataset_builder] 无法加载 LLM 客户端，退回规则过滤: {exc}")
        for entity in pending:
            result[entity] = _is_valid_entity_rule(entity)
            cache[entity] = result[entity]
        if cache_path is not None:
            _save_entity_filter_cache(cache_path, cache)
        return result

    system_prompt = (
        "你是建筑规范 QA 数据清洗助手。"
        "你的任务是判断一个字符串是否适合作为问句主体 entity。"
        "这里只保留空间/场所/地点类实体，如房间、区域、通道、厅、走廊、楼梯间、超市、办公室、教室。"
        "凡是不是地点或场所的对象，一律判为非法。"
        "非法 entity 包括纯数字、阈值表达、比较符号、表头泛词、含义不明碎片、设备型号、产品规格、分类名称、背景/图像描述、工艺过程名称。"
        "必须严格返回 JSON，不要解释。"
    )

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        prompt = (
            "请判断以下 entity 是否适合作为问题模板“entity的metric是多少？”中的 entity。\n"
            "判定标准：\n"
            "1. 只保留地点/场所/空间/区域类实体，如：餐厅、起居室、办公室、通道、连接区、扶梯、换乘厅。\n"
            "2. 非法：纯数字或比较表达（如 >5、220-240）、表头词（如 规格、参数）、意义不明碎片、设备型号、产品类别、屏幕分类、背景/图像描述、工艺过程名称。\n"
            "3. 不是地点/场所的对象，一律判 false。\n"
            "3. 仅输出 JSON，格式如下：\n"
            "{\n"
            '  "items": [\n'
            '    {"entity": "起居室", "valid": true},\n'
            '    {"entity": ">5", "valid": false},\n'
            '    {"entity": "规格", "valid": false},\n'
            '    {"entity": "屏幕分类", "valid": false}\n'
            "  ]\n"
            "}\n\n"
            f"待判断 entity 列表：\n{json.dumps(batch, ensure_ascii=False)}"
        )
        try:
            payload = _chat_json(prompt, system_prompt, model or "qwen-plus")
            items = payload.get("items", [])
            batch_result: Dict[str, bool] = {}
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    entity = str(item.get("entity") or "").strip()
                    if entity:
                        batch_result[entity] = bool(item.get("valid"))

            for entity in batch:
                is_valid = batch_result.get(entity, _is_valid_entity_rule(entity))
                result[entity] = is_valid
                cache[entity] = is_valid
        except Exception as exc:
            print(f"[dataset_builder] LLM 过滤失败，退回规则过滤。batch_start={start} error={exc}")
            for entity in batch:
                result[entity] = _is_valid_entity_rule(entity)
                cache[entity] = result[entity]

    if cache_path is not None:
        _save_entity_filter_cache(cache_path, cache)
    return result


def build_qa_dataset(
    kg_store_dir: Path,
    *,
    entity_filter_mode: str = "llm",
    entity_filter_cache_path: Optional[Path] = None,
    entity_filter_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
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
    candidates: List[Dict[str, Any]] = []
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

            candidates.append({
                "requirement_id": req_id,
                "entity": entity_name,
                "metric": metric_name,
                "condition": condition_text,
                "answer_raw": value_raw,
                "answer": _extract_numeric(value_raw) or value_raw,
                "source_path": source_path,
                "page_no": page_no,
            })

    entities = [str(item["entity"]) for item in candidates]
    if entity_filter_mode == "llm":
        entity_valid_map = _llm_filter_entities(
            entities,
            cache_path=entity_filter_cache_path,
            model=entity_filter_model,
        )
    else:
        entity_valid_map = {entity: _is_valid_entity_rule(entity) for entity in set(entities)}

    for item in candidates:
        entity_name = str(item["entity"])
        metric_name = str(item["metric"])
        condition_text = item.get("condition")
        if not entity_valid_map.get(entity_name, False):
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
            "answer": item["answer"],
            "answer_raw": item["answer_raw"],
            "requirement_id": item["requirement_id"],
            "entity": entity_name,
            "metric": metric_name,
            "condition": condition_text,
            "source_path": item["source_path"],
            "page_no": item["page_no"],
        })

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="从知识图谱自动生成 QA 评测数据集")
    parser.add_argument("--kg_store", default="kg_store", help="kg_store 目录路径")
    parser.add_argument("--output", default="eval/qa_dataset.json", help="输出 JSON 文件路径")
    parser.add_argument(
        "--entity_filter",
        default="llm",
        choices=["llm", "rule"],
        help="entity 过滤模式：llm=LLM批量判别并带规则兜底，rule=仅规则过滤",
    )
    parser.add_argument(
        "--entity_filter_cache",
        default="eval/entity_filter_cache.json",
        help="LLM entity 过滤缓存文件路径",
    )
    parser.add_argument(
        "--entity_filter_model",
        default=None,
        help="可选：覆盖默认 entity 过滤模型名",
    )
    args = parser.parse_args()

    kg_store_dir = Path(args.kg_store)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = build_qa_dataset(
        kg_store_dir,
        entity_filter_mode=args.entity_filter,
        entity_filter_cache_path=Path(args.entity_filter_cache) if args.entity_filter_cache else None,
        entity_filter_model=args.entity_filter_model,
    )
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
