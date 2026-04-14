#!/bin/bash
# 编译 wiring_*.tex → PDF → PNG（3张）
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

PYTHON="../.venv/bin/python"

echo "[1/3] 生成 .tex ..."
$PYTHON plot_wiring.py

cd "$RESULTS_DIR"

for NAME in wiring_encoder wiring_policy wiring_value; do
    echo "[compile] $NAME ..."
    xelatex -interaction=nonstopmode ${NAME}.tex
    pdftoppm -r 300 -png ${NAME}.pdf ${NAME}_page
    mv ${NAME}_page-1.png ${NAME}.png
    rm -f ${NAME}.aux ${NAME}.log
done

echo "done → article/results/wiring_encoder.png, wiring_policy.png, wiring_value.png"
