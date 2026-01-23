#!/usr/bin/env python3
"""
Script to count white pixels in mask images and save results to CSV.
"""

import os
import glob
import csv
import cv2
import numpy as np
from pathlib import Path


def count_white_pixels(image_path):
    """
    Count the number of white pixels in an image.

    Args:
        image_path: Path to the image file

    Returns:
        Number of white pixels (pixels with value 255)
    """
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return 0

    # Count pixels with value 255 (white)
    white_pixel_count = np.sum(img == 255)

    return int(white_pixel_count)


def main():
    # Get script directory and set base path
    script_dir = Path(__file__).parent
    base_dir = script_dir / "data" / "processed"

    # Find all images in masks directories
    mask_pattern = str(base_dir / "*" / "*" / "masks" / "*")
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']

    # Get all mask files
    all_files = glob.glob(mask_pattern)
    image_files = [f for f in all_files if any(f.endswith(ext) for ext in image_extensions)]

    print(f"Found {len(image_files)} mask images")

    # Process each image and collect results
    results = []
    for i, image_path in enumerate(image_files):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(image_files)}...")

        white_pixel_count = count_white_pixels(image_path)
        results.append({
            'image_path': image_path,
            'total_pixels': white_pixel_count
        })

    # Save results to CSV
    output_file = script_dir / "output" / "white_pixel_counts.csv"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['image_path', 'total_pixels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_file}")
    print(f"Total images processed: {len(results)}")

    # Print summary statistics
    if results:
        total_pixels = [r['total_pixels'] for r in results]
        print(f"Mean white pixels per image: {np.mean(total_pixels):.2f}")
        print(f"Min white pixels: {np.min(total_pixels)}")
        print(f"Max white pixels: {np.max(total_pixels)}")


if __name__ == "__main__":
    main()
