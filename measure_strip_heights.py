#!/usr/bin/env python3
"""
Script to measure heights of segmented green strips and save to CSV
"""

import argparse
import os
import csv
import cv2
import numpy as np
from pathlib import Path
import glob


def measure_strip_height(mask):
    """
    Measure height statistics of a segmented strip

    Args:
        mask: Binary mask (2D numpy array)

    Returns:
        dict with height statistics
    """
    # Find all y-coordinates where mask is True
    rows, cols = np.where(mask > 0)

    if len(rows) == 0:
        return {
            'mean_height': 0,
            'median_height': 0,
            'min_height': 0,
            'max_height': 0,
            'height_range': 0,
            'num_pixels': 0
        }

    # Calculate height for each column
    heights = []
    unique_cols = np.unique(cols)

    for col in unique_cols:
        # Get all rows for this column
        col_rows = rows[cols == col]
        # Height is the difference between max and min row in this column
        height = col_rows.max() - col_rows.min() + 1
        heights.append(height)

    heights = np.array(heights)

    return {
        'mean_height': float(np.mean(heights)),
        'median_height': float(np.median(heights)),
        'min_height': float(np.min(heights)),
        'max_height': float(np.max(heights)),
        'height_range': float(np.max(heights) - np.min(heights)),
        'num_pixels': int(np.sum(mask > 0))
    }


def load_mask(mask_path):
    """Load a binary mask from file"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask


def process_strip_directory(strip_dir, image_name):
    """
    Process all strips in a directory and collect measurements

    Args:
        strip_dir: Directory containing strip masks
        image_name: Name of the original image

    Returns:
        list of measurement dictionaries
    """
    measurements = []

    # Find all mask files
    mask_files = sorted(glob.glob(os.path.join(strip_dir, "strip_*_mask.png")))

    print(f"Found {len(mask_files)} strip masks")

    for mask_file in mask_files:
        # Extract strip number from filename
        basename = os.path.basename(mask_file)
        strip_num = basename.split('_')[1]

        print(f"Processing strip {strip_num}...")

        # Load mask
        mask = load_mask(mask_file)

        # Measure heights
        stats = measure_strip_height(mask)

        # Load box coordinates if available
        box_file = mask_file.replace('_mask.png', '_box.txt')
        box_coords = None
        score = None

        if os.path.exists(box_file):
            with open(box_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('Box:'):
                        box_coords = line.split('Box:')[1].strip()
                    elif line.startswith('Score:'):
                        score = float(line.split('Score:')[1].strip())

        # Combine all data
        measurement = {
            'image_name': image_name,
            'strip_id': int(strip_num),
            'mean_height_px': stats['mean_height'],
            'median_height_px': stats['median_height'],
            'min_height_px': stats['min_height'],
            'max_height_px': stats['max_height'],
            'height_range_px': stats['height_range'],
            'area_px': stats['num_pixels'],
            'box_coords': box_coords,
            'sam_score': score
        }

        measurements.append(measurement)

    return measurements


def save_to_csv(measurements, output_path):
    """
    Save measurements to CSV file

    Args:
        measurements: List of measurement dictionaries
        output_path: Path to output CSV file
    """
    if not measurements:
        print("No measurements to save")
        return

    # Define CSV columns
    fieldnames = [
        'image_name',
        'strip_id',
        'mean_height_px',
        'median_height_px',
        'min_height_px',
        'max_height_px',
        'height_range_px',
        'area_px',
        'box_coords',
        'sam_score'
    ]

    # Write CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(measurements)

    print(f"✅ Saved {len(measurements)} measurements to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Measure heights of segmented green strips")
    parser.add_argument("--input-dir", "-i", type=str, required=True,
                        help="Directory containing strip segmentation results")
    parser.add_argument("--output-csv", "-o", type=str, default="strip_measurements.csv",
                        help="Output CSV file")
    parser.add_argument("--batch", action="store_true",
                        help="Process multiple images (subdirectories) in input-dir")

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    all_measurements = []

    if args.batch:
        # Process multiple image directories
        subdirs = [d for d in os.listdir(args.input_dir)
                   if os.path.isdir(os.path.join(args.input_dir, d))]

        print(f"Processing {len(subdirs)} images in batch mode...")

        for subdir in subdirs:
            strip_dir = os.path.join(args.input_dir, subdir)
            print(f"\n--- Processing {subdir} ---")
            measurements = process_strip_directory(strip_dir, subdir)
            all_measurements.extend(measurements)

    else:
        # Process single image directory
        # Get parent directory name as image name
        image_name = os.path.basename(os.path.normpath(args.input_dir))
        measurements = process_strip_directory(args.input_dir, image_name)
        all_measurements.extend(measurements)

    # Save to CSV
    if all_measurements:
        save_to_csv(all_measurements, args.output_csv)

        # Print summary statistics
        print("\n=== Summary Statistics ===")
        print(f"Total strips measured: {len(all_measurements)}")

        if len(all_measurements) > 0:
            mean_heights = [m['mean_height_px'] for m in all_measurements]
            print(f"Overall mean height: {np.mean(mean_heights):.2f} px")
            print(f"Overall median height: {np.median(mean_heights):.2f} px")
            print(f"Height std dev: {np.std(mean_heights):.2f} px")
    else:
        print("No measurements collected")


if __name__ == "__main__":
    main()
