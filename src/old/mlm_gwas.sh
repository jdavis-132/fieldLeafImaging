#!/bin/bash
set -e  # Exit on error
set -x  # Print commands (for debugging)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PANICLE

for i in $(seq 1 256); do
  python src/run_GWAS.py \
    --phenotype "output/sam3_blues_${i}.csv" \
    --phenotype-id-column genotype \
    --genotype data/genotype/sam3_rs.vcf \
    --methods MLM \
    --n-pcs 5 \
    --compute-effective-tests \
    --outputs all_marker_pvalues \
    --outputdir output/mlm_20260512
done
