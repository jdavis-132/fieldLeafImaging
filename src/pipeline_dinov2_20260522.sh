#!/bin/bash
set -e  # Exit on error
set -x  # Print commands (for debugging)

source /home/schnable/anaconda3/etc/profile.d/conda.sh
conda activate jupyterlab-debugger

# get DINOv2 embeddings
python src/dinov2/extract_dinov2_features_20260522.py "data/processed/ne2025/device*/cropped*/"  output/dinov2_20260522.csv
python src/dinov2/extract_dinov2_features_20260522.py "data/processed/aamu2025/block*/cropped*/"  output/dinov2_20260522_aamu.csv
python src/dinov2/extract_dinov2_features_20260522.py "data/processed/fvsu2025/cropped*/"  output/dinov2_20260522_fvsu.csv


