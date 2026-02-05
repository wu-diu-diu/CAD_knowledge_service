import base64
import json
import os
import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

TABLE_RE = re.compile(r"<table>.*?</table>", re.S | re.I)
IMAGE_LINE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
CAPTION_RE = re.compile(r"^\s*(\(?[a-zA-Z]\)|图|Figure|Fig\.|表)", re.I)

SYSTEM_PROMPT = """
你是专注于理解学术论文插图的研究助手。你的任务是分析输入图片，并输出该图在论文中的含义与作用的结构化解释。

你不需要逐字转录图片中的全部文字，而是要理解图像的视觉内容、学术语境和整体结构，从而概括其目的和功能。

请严格按照以下 JSON 结构输出：

{
  "purpose": "(图的核心意图：这张图想表达什么？)",
  "role_in_paper": "(图在论文中的作用：例如展示模型结构、呈现实验结果、对比方法等)",
  "key_insights": [
    "(关键结论 1：例如模型在 GSM8K 上表现更好)",
    "(关键结论 2：例如 MoE 降低推理成本)",
    "..."
  ],
  "visual_elements_summary": "(简要描述图中包含的元素：结构、曲线、流程、子图等。不要逐字抄写，只需概括要点)"
}

请使用简洁、专业、准确的中文表述。输出内容应适合用于下游 RAG 知识系统的检索与问答。不要编造图片中没有明确体现的信息。如有难以辨认的内容，请明确说明：“部分内容难以辨认”。  
"""

def get_max_physical_columns(table_soup):
    """计算表格的最大物理列数（考虑 colspan）"""
    max_cols = 0
    rows = table_soup.find_all('tr')
    for tr in rows:
        current_cols = 0
        cells = tr.find_all(['td', 'th'])
        if not cells: continue
        for cell in cells:
            current_cols += int(cell.get('colspan', 1))
        max_cols = max(max_cols, current_cols)
    return max_cols

def adjust_first_column_span(table_soup, diff_amount):
    """给表格每一行的第一个单元格增加 colspan，用于撑宽较窄的表格"""
    if diff_amount <= 0: return
    for tr in table_soup.find_all('tr'):
        cells = tr.find_all(['td', 'th'])
        if cells:
            first_cell = cells[0]
            current_span = int(first_cell.get('colspan', 1))
            first_cell['colspan'] = current_span + diff_amount

def html_table_to_md_grid(table_str):
    """将 HTML 字符串转换为 Markdown 管道符表格"""
    soup = BeautifulSoup(table_str, 'html.parser')
    table_tag = soup.find('table')
    if not table_tag: return table_str

    rows = table_tag.find_all('tr')
    if not rows: return ""

    matrix = {}
    max_r, max_c = 0, 0
    curr_r = 0
    
    for tr in rows:
        curr_c = 0
        cells = tr.find_all(['td', 'th'])
        if not cells: continue

        for cell in cells:
            while (curr_r, curr_c) in matrix:
                curr_c += 1
            
            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))
            # 文本清洗
            text = cell.get_text(strip=True).replace('|', '\\|').replace('\n', '<br>')
            
            for r in range(curr_r, curr_r + rowspan):
                for c in range(curr_c, curr_c + colspan):
                    matrix[(r, c)] = text
                    max_c = max(max_c, c)
            max_r = max(max_r, curr_r + rowspan - 1)
            curr_c += colspan
        curr_r += 1

    md_lines = []
    for r in range(max_r + 1):
        row_content = [matrix.get((r, c), " ") for c in range(max_c + 1)]
        md_lines.append("| " + " | ".join(row_content) + " |")
        if r == 0:
            md_lines.append("| " + " | ".join(["---"] * (max_c + 1)) + " |")

    return "\n" + "\n".join(md_lines) + "\n"


def _get_openai_client() -> Optional[object]:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def _guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"


def explain_image(image_path: Path, model_name: str = "qwen-vl-plus") -> Optional[str]:
    client = _get_openai_client()
    if client is None:
        print("Skipping image explanation: missing DASHSCOPE_API_KEY or openai package.")
        return None

    img_bytes = image_path.read_bytes()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    mime_type = _guess_mime_type(image_path)
    data_url = f"data:{mime_type};base64,{img_base64}"

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Explain the figure"},
                ],
            },
        ],
    )

    response_json = json.loads(completion.model_dump_json())
    return response_json["choices"][0]["message"]["content"]


def extract_json_from_markdown(text: str) -> Optional[dict]:
    """
    从 markdown 代码块中提取 JSON 字符串并解析为字典
    """
    match = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print("⚠️ JSON解析失败：", e)
            return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("⚠️ 未找到 JSON 代码块")
        return None


def process_images_in_markdown(md_content: str, md_path: Path) -> str:
    """
    将 Markdown 中的图片替换为结构化 JSON 描述块。
    """
    lines = md_content.splitlines()
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = IMAGE_LINE_RE.match(line.strip())
        if not match:
            new_lines.append(line)
            i += 1
            continue

        image_rel_path = match.group("path")
        image_path = Path(image_rel_path)
        if not image_path.is_absolute():
            image_path = (md_path.parent / image_path).resolve()

        caption = None
        skip_next = False
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and (CAPTION_RE.match(next_line) or ": " in next_line):
                caption = next_line.split(": ", 1)[1].strip() if ": " in next_line else next_line
                skip_next = True

        if not image_path.exists():
            print(f"Image not found: {image_path}")
            new_lines.append(line)
            i += 1
            continue

        explanation = explain_image(image_path)
        if explanation is None:
            new_lines.append(line)
            i += 1
            continue

        structured = extract_json_from_markdown(explanation) or {"raw": explanation}
        if caption:
            structured = {"figure_title": caption, **structured}
        json_block = "```json\n" + json.dumps(structured, ensure_ascii=False, indent=2) + "\n```"
        new_lines.append(json_block)
        i += 2 if skip_next else 1

    return "\n".join(new_lines)

def merge_html_tables(html1, html2):
    """合并两个 HTML 表格字符串，返回合并后的 HTML 字符串"""
    soup1 = BeautifulSoup(html1, 'html.parser')
    soup2 = BeautifulSoup(html2, 'html.parser')
    t1 = soup1.find('table')
    t2 = soup2.find('table')

    if t1 and t2:
        # 1. 双向列宽对齐
        cols1 = get_max_physical_columns(t1)
        cols2 = get_max_physical_columns(t2)
        
        if cols2 > cols1:
            adjust_first_column_span(t1, cols2 - cols1)
        elif cols1 > cols2:
            adjust_first_column_span(t2, cols1 - cols2)

        # 2. 数据合并
        rows2 = t2.find_all('tr')
        if len(rows2) > 0:
            # 跳过表2的表头(第一行)，追加剩余行
            for row in rows2[1:]:
                t1.append(row)
                
    return str(t1)

def process_markdown_file(md_content):
    """
    基于切片的处理逻辑：
    1. 按 <table> 标签切分文档。
    2. 扫描切片，发现“续表”则合并相邻切片。
    3. 最后统一转 Markdown。
    """
    
    # 1. 切分文档
    # 使用 capturing group () 保留分割符，这样列表里既有文本也有表格
    # parts 结构: [文本, 表格, 文本, 表格, 文本...]
    # 索引为奇数的是表格，偶数的是文本
    pattern = re.compile(r'(<table[^>]*>.*?</table>)', re.DOTALL | re.IGNORECASE)
    parts = re.split(pattern, md_content)
    
    # 2. 循环扫描并合并
    # 我们使用 while 循环，因为列表长度会在合并过程中变短
    has_merged = True
    while has_merged:
        has_merged = False
        i = 1 # 从第一个表格开始遍历 (索引0是开头的文本)
        
        while i < len(parts) - 2:
            current_part = parts[i]
            gap_text = parts[i+1]
            next_part = parts[i+2]
            
            # 检查结构：[表格] + [含续表的文本] + [表格]
            is_curr_table = current_part.strip().lower().startswith('<table')
            is_next_table = next_part.strip().lower().startswith('<table')
            has_continuation_mark = "续表" in gap_text
            
            if is_curr_table and is_next_table and has_continuation_mark:
                # --- 执行合并 ---
                merged_html = merge_html_tables(current_part, next_part)
                
                # 更新当前表格为合并后的表格
                parts[i] = merged_html
                
                # 删除中间的间隔文本(去除"续表"字样) 和 下一个表格
                # 我们将其设为多余的空字符串，稍后拼接时会消失，或者直接从列表删除
                del parts[i+2] # 删除后一个表格
                del parts[i+1] # 删除中间的"续表"文本
                
                # 标记发生了合并，需要重新检查（处理多级续表）
                has_merged = True
                # 注意：i 不需要增加，因为后面的元素前移了，我们需要检查新的 parts[i+1]
            else:
                # 如果没合并，移动到下一个表格
                i += 2

    # 3. 转换与重组
    final_content = []
    for part in parts:
        # 如果是表格，转 Markdown；如果是文本，保持原样
        if part.strip().lower().startswith('<table'):
            final_content.append(html_table_to_md_grid(part))
        else:
            final_content.append(part)
            
    return "".join(final_content)

def fix_md_headings(md_path):
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # 处理 2.3.4. 这种多级编号，转为3级标题
        m = re.match(r"^#+\s+(\d+\.\d+\.\d+\.)\s+(.*)", line)
        if m:
            section, title = m.groups()
            new_lines.append(f"### {section} {title}\n")
            continue

        # 处理 4.1. 这种两级编号，转为2级标题
        m = re.match(r"^#+\s+(\d+\.\d+\.)\s+(.*)", line)
        if m:
            section, title = m.groups()
            new_lines.append(f"## {section} {title}\n")
            continue
        # 处理数字编号标题，如 # 2.3 或 # 2.3.1
        m = re.match(r"^(#+)\s+(\d+(?:\.\d+)+)\s+(.*)", line)
        if m:
            hashes, section, title = m.groups()
            level = section.count('.') + 1
            new_line = "#" * level + f" {section} {title}\n"
            new_lines.append(new_line)
            continue

        # 处理罗马数字 III.，转为3级标题
        m = re.match(r"^#+\s+III\.\s+(.*)", line)
        if m:
            title = m.group(1)
            new_lines.append(f"### III. {title}\n")
            continue

        # 处理罗马数字 II.，转为2级标题
        m = re.match(r"^#+\s+II\.\s+(.*)", line)
        if m:
            title = m.group(1)
            new_lines.append(f"## II. {title}\n")
            continue

        # 处理 A.1/B.1/C.1 这类标题，转为2级标题
        m = re.match(r"^#+\s+([A-Z])\.(\d+)\s+(.*)", line)
        if m:
            letter, num, title = m.groups()
            new_lines.append(f"## {letter}.{num} {title}\n")
            continue

        new_lines.append(line)

    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def fix_md_tables(md_path):
    content = Path(md_path).read_text(encoding="utf-8", errors="ignore")
    merged = process_markdown_file(content)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(merged)

md_path = Path("/home/chen/punchy/CAD_knowledge_service/output/建筑照明设计标准_260202_124455/hybrid_auto/test.md")

# fix_md_headings(md_path)
# fix_md_tables(md_path)
# new_content = process_images_in_markdown(md_path.read_text(encoding="utf-8"), md_path)
# with open(md_path, "w", encoding="utf-8") as f:
#     f.write(new_content)
