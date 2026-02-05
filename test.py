import re
from bs4 import BeautifulSoup

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

# --- 以下是你提供的业务代码，无需修改，直接运行即可 ---

if __name__ == "__main__":
    input_file = "/home/chen/punchy/CAD_knowledge_service/test_files/test1.md"
    output_file = "/home/chen/punchy/CAD_knowledge_service/test_files/test1_processed.md"
    
    # 1. 读取
    with open(input_file, "r", encoding="utf-8") as f:
        test_markdown = f.read()

    # 2. 处理 (直接调用上面的函数)
    processed = process_markdown_file(test_markdown)

    # 3. 保存
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(processed)

    print(f"处理完成！文件已保存至: {output_file}")