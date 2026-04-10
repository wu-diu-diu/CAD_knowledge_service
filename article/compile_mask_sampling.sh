#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

PYTHON="../.venv/bin/python"

echo "[1/3] generating tex ..."
$PYTHON plot_mask_sampling.py

echo "[2/3] pdflatex ..."
cd "$RESULTS_DIR"
pdflatex -interaction=nonstopmode mask_sampling.tex

echo "[3/3] pdf -> png ..."
pdftoppm -r 300 -png mask_sampling.pdf mask_sampling_page
mv mask_sampling_page-1.png mask_sampling.png
rm -f mask_sampling.aux mask_sampling.log

echo "done -> article/results/mask_sampling.png"
