#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="output/reframing_results/all_dinov2_20260522_full_mlm_lrt"
mkdir -p "$OUT_DIR"

python src/run_all_dinov2_full_mlm_lrt.py \
  --out-dir "$OUT_DIR" \
  --chunk-size 32 \
  --top-k 500 \
  --lrt-solver GEMMA \
  --lrt-batch-size 2048

python src/cluster_dinov2_gwas_signals.py \
  --out-dir "$OUT_DIR" \
  --window-bp 200000
