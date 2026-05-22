import os
import cv2
from pathlib import Path
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('raw_dir', help='Input directory with raw images')
parser.add_argument('mask_dir', help='Input directory containing masks for raw images')
parser.add_argument('out_dir', help='Output directory to save segmented images to')

args = parser.parse_args()

raw_images = glob.glob(args.raw_dir + '/*.*')
print(f'Found {len(raw_images)} images to segment')

for idx, image in enumerate(raw_images):
    raw_img = cv2.imread(image)
    rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    basename = os.path.basename(image)
    mask_path = os.path.join(args.mask_dir, str(basename).replace('.jpg', '.png'))
    mask = cv2.imread(mask_path)
    binary_mask = mask / 255.0
    segmented = raw_img * binary_mask
    cv2.imwrite(os.path.join(args.out_dir, str(basename).replace('.jpg', '.png')), segmented)