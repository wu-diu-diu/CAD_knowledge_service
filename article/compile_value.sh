#!/bin/bash
# 编译 model_value.tex → PDF → PNG
# 用法: cd article && bash compile_value.sh [output_name]
# 默认输出 model_value.png，可指定其他名称

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

# 使用项目 venv
PYTHON="../.venv/bin/python"

# 输出文件名（默认或自定义）
OUTPUT_NAME="${1:-model_value}"

echo "[1/3] 生成 .tex ..."
$PYTHON plot_value.py

echo "[2/3] pdflatex 编译 ..."
cd "$RESULTS_DIR"
pdflatex -interaction=nonstopmode model_value.tex

echo "[3/3] PDF → PNG (300dpi) ..."
pdftoppm -r 300 -png model_value.pdf model_value_page
mv model_value_page-1.png "${OUTPUT_NAME}.png"
# 清理 LaTeX 中间文件
rm -f model_value.aux model_value.log

echo "done → article/results/${OUTPUT_NAME}.png"
