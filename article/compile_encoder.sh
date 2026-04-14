#!/bin/bash
# 编译 model_encoder.tex → PDF → PNG
# 用法: cd article && bash compile_arch.sh [output_name]
# 默认输出 model_encoder.png

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

# 使用项目 venv
PYTHON="../.venv/bin/python"

# 输出文件名（默认或自定义）
OUTPUT_NAME="${1:-model_encoder}"

echo "[1/3] 生成 .tex ..."
$PYTHON plot_encoder.py

echo "[2/3] xelatex 编译 ..."
cd "$RESULTS_DIR"
xelatex -interaction=nonstopmode model_encoder.tex

echo "[3/3] PDF → PNG (300dpi) ..."
pdftoppm -r 300 -png model_encoder.pdf model_encoder_page
mv model_encoder_page-1.png "${OUTPUT_NAME}.png"

# 清理 LaTeX 中间文件
rm -f model_encoder.aux model_encoder.log

echo "done → article/results/${OUTPUT_NAME}.png"
