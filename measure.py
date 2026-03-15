from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from kg.md_parser import parse_markdown_blocks


DEFAULT_MD_PATH = Path(
    "/home/chen/punchy/CAD_knowledge_service/transfer_output/建筑照明设计标准_20260310_234524.md"
)

FENCE_RE = re.compile(r"^```")
DISPLAY_DOLLAR_RE = re.compile(r"\$\$[\s\S]*?\$\$")
DISPLAY_BRACKET_RE = re.compile(r"\\\[[\s\S]*?\\\]")
INLINE_PAREN_RE = re.compile(r"\\\([\s\S]*?\\\)")
PAGE_RE = re.compile(r"^\[PAGE:\d+\]\s*$")
HEADING_RE = re.compile(r"^#{1,6}\s+")
TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$")
LIST_ITEM_RE = re.compile(r"^\s*[-*+]\s+")


def strip_fenced_code_blocks(text: str) -> str:
    lines = text.splitlines()
    kept_lines: list[str] = []
    in_fence = False

    for line in lines:
        if FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            continue
        if not in_fence:
            kept_lines.append(line)
    return "\n".join(kept_lines)


def count_formulas(text: str) -> tuple[int, int, int]:
    clean_text = strip_fenced_code_blocks(text)

    display_dollar = DISPLAY_DOLLAR_RE.findall(clean_text)
    clean_text = DISPLAY_DOLLAR_RE.sub("\n", clean_text)

    display_bracket = DISPLAY_BRACKET_RE.findall(clean_text)
    clean_text = DISPLAY_BRACKET_RE.sub(" ", clean_text)

    inline_paren = INLINE_PAREN_RE.findall(clean_text)

    display_count = len(display_dollar) + len(display_bracket)
    inline_count = len(inline_paren)
    total_count = display_count + inline_count
    return total_count, display_count, inline_count


def count_text_paragraphs(text: str) -> int:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n+", normalized)
    paragraph_count = 0

    for raw_block in blocks:
        block = raw_block.strip()
        if not block:
            continue

        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        first_line = lines[0].strip()

        if FENCE_RE.match(first_line):
            continue
        if PAGE_RE.match(first_line):
            continue
        if HEADING_RE.match(first_line):
            continue
        if all(TABLE_ROW_RE.match(line.strip()) for line in lines):
            continue
        if block.startswith("$$") and block.endswith("$$"):
            continue
        if block.startswith("\\[") and block.endswith("\\]"):
            continue
        if len(lines) == 1 and (lines[0].startswith("![") or lines[0].startswith("```json")):
            continue
        if all(LIST_ITEM_RE.match(line) for line in lines):
            continue

        paragraph_count += 1

    return paragraph_count


def measure_markdown(md_path: Path) -> dict[str, object]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    blocks = parse_markdown_blocks(text)

    heading_levels: Counter[int] = Counter()
    for block in blocks:
        if block.block_type != "heading":
            continue
        try:
            level = int((block.meta or {}).get("level", 1))
        except Exception:
            level = 1
        heading_levels[level] += 1

    heading_count = sum(heading_levels.values())
    table_count = sum(1 for block in blocks if block.block_type == "table")
    paragraph_count = count_text_paragraphs(text)
    figure_json_count = sum(1 for block in blocks if block.block_type == "figure_json")
    formula_count, display_formula_count, inline_formula_count = count_formulas(text)

    return {
        "tables": table_count,
        "headings": heading_count,
        "heading_levels": dict(sorted(heading_levels.items())),
        "paragraphs": paragraph_count,
        "figure_json_blocks": figure_json_count,
        "formulas": formula_count,
        "display_formulas": display_formula_count,
        "inline_formulas": inline_formula_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 Markdown 文档中的结构块数量。")
    parser.add_argument(
        "md_path",
        nargs="?",
        default=str(DEFAULT_MD_PATH),
        help="待统计的 Markdown 文件路径",
    )
    args = parser.parse_args()

    md_path = Path(args.md_path).expanduser().resolve()
    if not md_path.exists():
        raise FileNotFoundError(f"文件不存在: {md_path}")

    stats = measure_markdown(md_path)

    print(f"文件: {md_path}")
    print(f"表格数量: {stats['tables']}")
    print(f"标题数量: {stats['headings']}")
    heading_levels = stats.get("heading_levels", {})
    if isinstance(heading_levels, dict):
        for level, count in heading_levels.items():
            print(f"{level}级标题数量: {count}")
    print(f"文本段落数量: {stats['paragraphs']}")
    print(f"图片 JSON 块数量: {stats['figure_json_blocks']}")
    print(f"公式数量: {stats['formulas']}")
    print(f"  其中块公式数量: {stats['display_formulas']}")
    print(f"  其中行内公式数量: {stats['inline_formulas']}")


if __name__ == "__main__":
    main()
