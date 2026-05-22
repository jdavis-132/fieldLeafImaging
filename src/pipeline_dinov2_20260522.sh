#!/bin/bash
set -e  # Exit on error
set -x  # Print commands (for debugging)

source /home/schnable/anaconda3/etc/profile.d/conda.sh
conda activate jupyterlab-debugger

# get DINOv2 embeddings
python src/dinov2/extract_dinov2_features_20260522.py "data/processed/ne2025/device*/cropped*/"  output/dinov2_20260522.csv
# run these if dino outperforms sam
#python src/dinov2/extract_dinov2_features_20260522.py "data/processed/aamu2025/block*/cropped*/"  output/dinov2_20260522_aamu.csv
#python src/dinov2/extract_dinov2_features_20260522.py "data/processed/fvsu2025/cropped*/"  output/dinov2_20260522_fvsu.csv

# Run BLUEs & prep files for RF
Rscript src/R/LV_quantGen.R -image -output/dinov2_20260522.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -mean,std -dinov2_20260522_all -0.01 -data/genotype_conversion_table.csv -data/ne2025/images_keep_all.csv > logs/dinov2_quantgen_20260522.Rout 2>&1
Rscript src/R/LV_quantGen.R -image -output/dinov2_20260522.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -mean -dinov2_20260522_mean -0.01 -data/genotype_conversion_table.csv -data/ne2025/images_keep_all.csv > logs/dinov2_quantgen_20260522.Rout 2>&1
Rscript src/R/LV_quantGen.R -image -output/dinov2_20260522.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -std -dinov2_20260522_std -0.01 -data/genotype_conversion_table.csv -data/ne2025/images_keep_all.csv > logs/dinov2_quantgen_20260522.Rout 2>&1

# run these if dino outperforms sam
#Rscript src/R/LV_quantGen.R -image -output/dinov2_20260522_aamu.csv -data/aamu2025/aamu_field_index.csv -mean,std -dinov2_20260522_all_aamu -0.01 -data/genotype_conversion_table.csv -data/aamu2025/image_ids_keep.csv > logs/dinov2_quantgen_20260522.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -output/dinov2_20260522_aamu.csv -data/aamu2025/aamu_field_index.csv -mean -dinov2_20260522_mean_aamu -0.01 -data/genotype_conversion_table.csv -data/aamu2025/image_ids_keep.csv > logs/dinov2_quantgen_20260522.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -output/dinov2_20260522_aamu.csv -data/aamu2025/aamu_field_index.csv -std -dinov2_20260522_std_aamu -0.01 -data/genotype_conversion_table.csv -data/aamu2025/image_ids_keep.csv > logs/dinov2_quantgen_20260522.Rout 2>&1

#Rscript src/R/LV_quantGen.R -image -output/dinov2_20260522_fvsu.csv -data/fvsu2025/fvsu_field_index.csv -mean,std -dinov2_20260522_all_fvsu -0.01 -data/genotype_conversion_table.csv -data/fvsu2025/image_ids_keep.csv > logs/dinov2_quantgen_20260522.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -output/dinov2_20260522_fvsu.csv -data/fvsu2025/fvsu_field_index.csv -mean -dinov2_20260522_mean_fvsu -0.01 -data/genotype_conversion_table.csv -data/fvsu2025/image_ids_keep.csv > logs/dinov2_quantgen_20260522.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -output/dinov2_20260522_fvsu.csv -data/fvsu2025/fvsu_field_index.csv -std -dinov2_20260522_std_fvsu -0.01 -data/genotype_conversion_table.csv -data/fvsu2025/image_ids_keep.csv > logs/dinov2_quantgen_20260522.Rout 2>&1
