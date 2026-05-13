#!/bin/bash
set -e  # Exit on error
set -x  # Print commands (for debugging)

#Rscript src/R/sampleBLUEsForResampling.R

#conda activate PANICLE

python src/run_GWAS.py \
  --phenotype output/sam3_blues_highfi_samples.csv \
  --phenotype-id-column genotype \
  --genotype data/genotype/sam3_rs.vcf \
  --methods FarmCPU \
  --n-pcs 5 \
  --compute-effective-tests \
  --outputs significant_marker_pvalues \
  --outputdir output/farmcpu_20260512 | \
tee logs/farmcpu_20260512.txt
