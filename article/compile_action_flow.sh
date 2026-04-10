#!/bin/bash
# 编译 action_flow.tex → PDF → PNG
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

PYTHON="../.venv/bin/python"

echo "[1/3] 生成 .tex ..."
$PYTHON plot_action_flow.py

echo "[2/3] pdflatex 编译 ..."
cd "$RESULTS_DIR"
pdflatex -interaction=nonstopmode action_flow.tex

echo "[3/3] PDF → PNG (300dpi) ..."
pdftoppm -r 300 -png action_flow.pdf action_flow_page
mv action_flow_page-1.png action_flow.png
rm -f action_flow.aux action_flow.log

echo "done → article/results/action_flow.png"
