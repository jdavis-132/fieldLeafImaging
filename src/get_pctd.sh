#!/bin/bash
set -e  # Exit on error
set -x  # Print commands (for debugging)

source /home/schnable/anaconda3/etc/profile.d/conda.sh
conda activate jupyterlab-debugger

python src/extract_pctd.py --images data/processed/ne2025/device1/cropped/ --masks data/processed/ne2025/device1/masks_cropped/ --output data/processed/ne2025/device1/pctd_all.csv
python src/extract_pctd.py --images data/processed/ne2025/device2/cropped/ --masks data/processed/ne2025/device2/masks_cropped/ --output data/processed/ne2025/device2/pctd_all.csv
python src/extract_pctd.py --images data/processed/ne2025/device3/cropped/ --masks data/processed/ne2025/device3/masks_cropped/ --output data/processed/ne2025/device3/pctd_all.csv
python src/extract_pctd.py --images data/processed/ne2025/device4/cropped/ --masks data/processed/ne2025/device4/masks_cropped/ --output data/processed/ne2025/device4/pctd_all.csv
python src/extract_pctd.py --images data/processed/ne2025/device5/cropped/ --masks data/processed/ne2025/device5/masks_cropped/ --output data/processed/ne2025/device5/pctd_all.csv
python src/extract_pctd.py --images data/processed/ne2025/device6/cropped/ --masks data/processed/ne2025/device6/masks_cropped/ --output data/processed/ne2025/device6/pctd_all.csv
python src/extract_pctd.py --images data/processed/ne2025/device7/cropped/ --masks data/processed/ne2025/device7/masks_cropped/ --output data/processed/ne2025/device7/pctd_all.csv
python src/extract_pctd.py --images data/processed/ne2025/device8/cropped/ --masks data/processed/ne2025/device8/masks_cropped/ --output data/processed/ne2025/device8/pctd_all.csv

python src/extract_pctd.py --images data/processed/aamu2025/block1/cropped/ --masks data/processed/aamu2025/block1/masks_cropped/ --output data/processed/aamu2025/block1/pctd_all.csv
python src/extract_pctd.py --images data/processed/aamu2025/block2/cropped/ --masks data/processed/aamu2025/block2/masks_cropped/ --output data/processed/aamu2025/block2/pctd_all.csv

python src/extract_pctd.py --images data/processed/fvsu2025/cropped/ --masks data/processed/fvsu2025/masks_cropped/ --output data/processed/fvsu2025/pctd_all.csv
