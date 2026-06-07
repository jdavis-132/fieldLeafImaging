#!/usr/bin/env bash
set -euo pipefail

cd /home/james/leaf_imaging/fieldLeafImaging_github

date
python /home/james/leaf_imaging/fieldLeafImaging_github/src/run_all_sam3_full_mlm_lrt.py \
  --chunk-size 32 \
  --top-k 500 \
  --lrt-solver GEMMA \
  --lrt-batch-size 2048
python /home/james/leaf_imaging/fieldLeafImaging_github/src/cluster_sam3_gwas_signals.py \
  --window-bp 200000
date
