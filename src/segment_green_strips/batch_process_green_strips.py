#!/usr/bin/env python3
"""
Batch process all images in subdirectories with green strip detection
"""

import argparse
import os
import subprocess
import glob
from pathlib import Path
import csv
import cv2
import numpy as np


def find_images(base_dir, extensions=None):
    """
    Find all images in subdirectories

    Args:
        base_dir: Base directory to search
        extensions: List of image extensions to search for

    Returns:
        list of tuples: (image_path, subdirectory_name)
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    images = []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        # Get relative path from base_dir
        rel_path = os.path.relpath(root, base_dir)

        # Skip if we're in the base directory itself
        if rel_path == '.':
            continue

        # Use the immediate subdirectory name as identifier
        subdir_parts = rel_path.split(os.sep)
        subdir_id = subdir_parts[0] if subdir_parts else 'unknown'

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in extensions:
                image_path = os.path.join(root, file)
                images.append((image_path, subdir_id))

    return images


def detect_green_boxes(image_path, min_area=5000):
    """
    Detect green strips using improved detection algorithm

    Args:
        image_path: Path to image
        min_area: Minimum area for detection

    Returns:
        int: Number of boxes detected, or -1 on error
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return -1

        # Import the improved detection function
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from detect_green_strips import detect_green_strips

        # Run detection
        boxes, _ = detect_green_strips(image)

        return len(boxes)

    except Exception as e:
        print(f"  Error detecting boxes: {e}")
        return -1


def run_detection(image_path, output_dir, device='cuda', model_path='models/sam2.1_hiera_tiny.pt'):
    """
    Run detect_green_strips.py on a single image

    Args:
        image_path: Path to image
        output_dir: Output directory
        device: Device to use (cuda/cpu)
        model_path: Path to SAM2 model

    Returns:
        bool: True if successful, False otherwise
    """
    cmd = [
        'python3',
        'detect_green_strips.py',
        '--image', image_path,
        '--output-dir', output_dir,
        '--device', device,
        '--model', model_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            return True
        else:
            print(f"  Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  Timeout processing image")
        return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def measure_strips(strip_dir, image_identifier):
    """
    Measure strips in a directory and return measurements

    Args:
        strip_dir: Directory containing strip masks
        image_identifier: Identifier for the image

    Returns:
        list of measurement dicts
    """
    cmd = [
        'python3',
        'measure_strip_heights.py',
        '--input-dir', strip_dir,
        '--output-csv', '/tmp/temp_measurements.csv'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and os.path.exists('/tmp/temp_measurements.csv'):
            # Read measurements
            measurements = []
            with open('/tmp/temp_measurements.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Add image identifier
                    row['subdirectory'] = image_identifier
                    measurements.append(row)

            # Clean up temp file
            os.remove('/tmp/temp_measurements.csv')
            return measurements
        else:
            return []
    except Exception as e:
        print(f"  Error measuring strips: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Batch process images with green strip detection")
    parser.add_argument("--data-dir", "-d", type=str, default="data/ne2025",
                        help="Base data directory containing subdirectories")
    parser.add_argument("--output-dir", "-o", type=str, default="batch_green_strips_output",
                        help="Output directory for all results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--model", type=str, default="models/sam2.1_hiera_tiny.pt",
                        help="Path to SAM2 model")
    parser.add_argument("--measurements-csv", "-m", type=str, default="all_strip_measurements.csv",
                        help="Output CSV file for all measurements")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images that have already been processed")
    parser.add_argument("--min-boxes", type=int, default=2,
                        help="Minimum number of boxes required (skip if fewer)")
    parser.add_argument("--max-boxes", type=int, default=2,
                        help="Maximum number of boxes allowed (skip if more)")
    parser.add_argument("--skipped-csv", type=str, default="skipped_images.csv",
                        help="CSV file to save skipped image paths")
    parser.add_argument("--fallback-csv", type=str, default="fallback_images.csv",
                        help="CSV file to save images that used fallback boxes")

    args = parser.parse_args()

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all images
    print(f"Scanning for images in {args.data_dir}...")
    images = find_images(args.data_dir)

    if not images:
        print("No images found!")
        return

    print(f"Found {len(images)} images to process\n")

    # Track results
    successful = 0
    failed = 0
    skipped = 0
    skipped_too_many_boxes = 0
    all_measurements = []
    skipped_images = []
    fallback_images = []

    # Process each image
    for i, (image_path, subdir_id) in enumerate(images, 1):
        image_name = Path(image_path).stem

        # Create identifier combining subdirectory and image name
        full_identifier = f"{subdir_id}_{image_name}"

        print(f"[{i}/{len(images)}] Processing: {subdir_id}/{image_name}")

        # Check if already processed
        expected_output_dir = os.path.join(args.output_dir, full_identifier)
        if args.skip_existing and os.path.exists(expected_output_dir):
            mask_files = glob.glob(os.path.join(expected_output_dir, "strip_*_mask.png"))
            if mask_files:
                print(f"  Skipping (already processed)")
                skipped += 1

                # Still collect measurements if available
                measurements = measure_strips(expected_output_dir, full_identifier)
                all_measurements.extend(measurements)
                continue

        # Check number of boxes before running SAM2
        num_boxes = detect_green_boxes(image_path)

        if num_boxes < 0:
            print(f"  ✗ Error detecting boxes")
            failed += 1
            continue

        print(f"  Detected {num_boxes} green strips")

        # Track images that need fallback boxes (0 or 1 box detected)
        needs_fallback = num_boxes < 2
        if needs_fallback:
            fallback_images.append({
                'image_path': image_path,
                'subdirectory': subdir_id,
                'image_name': image_name,
                'num_boxes_detected': num_boxes,
                'reason': 'fallback_boxes_needed'
            })
            print(f"  ⚠ Warning: Image will use fallback boxes ({num_boxes} detected)")

        if num_boxes < args.min_boxes:
            print(f"  ⊗ Skipping (fewer than {args.min_boxes} boxes detected)")
            skipped_too_many_boxes += 1
            skipped_images.append({
                'image_path': image_path,
                'subdirectory': subdir_id,
                'image_name': image_name,
                'num_boxes_detected': num_boxes,
                'reason': f'fewer_than_{args.min_boxes}_boxes'
            })
            continue

        if num_boxes > args.max_boxes:
            print(f"  ⊗ Skipping (more than {args.max_boxes} boxes detected)")
            skipped_too_many_boxes += 1
            skipped_images.append({
                'image_path': image_path,
                'subdirectory': subdir_id,
                'image_name': image_name,
                'num_boxes_detected': num_boxes,
                'reason': f'more_than_{args.max_boxes}_boxes'
            })
            continue

        # Create subdirectory-specific output directory
        subdir_output = os.path.join(args.output_dir, subdir_id)
        os.makedirs(subdir_output, exist_ok=True)

        # Run detection
        success = run_detection(image_path, subdir_output, args.device, args.model)

        if success:
            print(f"  ✓ Detection complete")
            successful += 1

            # Find the output directory for this image
            # detect_green_strips.py creates a subdirectory with the image stem
            strip_output_dir = os.path.join(subdir_output, image_name)

            if os.path.exists(strip_output_dir):
                # Rename to include subdirectory identifier
                new_strip_dir = os.path.join(args.output_dir, full_identifier)

                # Move to new location with full identifier
                if not os.path.exists(new_strip_dir):
                    os.rename(strip_output_dir, new_strip_dir)
                    print(f"  ✓ Renamed to: {full_identifier}")
                else:
                    print(f"  ! Output directory already exists: {new_strip_dir}")

                # Measure strips
                measurements = measure_strips(new_strip_dir, full_identifier)
                all_measurements.extend(measurements)

                if measurements:
                    print(f"  ✓ Measured {len(measurements)} strips")
        else:
            print(f"  ✗ Detection failed")
            failed += 1

    # Save all measurements to CSV
    if all_measurements:
        # Define all fieldnames
        fieldnames = [
            'subdirectory',
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

        output_csv = os.path.join(args.output_dir, args.measurements_csv)

        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_measurements)

        print(f"\n✅ Saved {len(all_measurements)} total measurements to {output_csv}")

    # Save skipped images to CSV
    if skipped_images:
        skipped_csv_path = os.path.join(args.output_dir, args.skipped_csv)

        with open(skipped_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['image_path', 'subdirectory', 'image_name', 'num_boxes_detected', 'reason']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(skipped_images)

        print(f"✅ Saved {len(skipped_images)} skipped images to {skipped_csv_path}")

    # Save fallback images to CSV
    if fallback_images:
        fallback_csv_path = os.path.join(args.output_dir, args.fallback_csv)

        with open(fallback_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['image_path', 'subdirectory', 'image_name', 'num_boxes_detected', 'reason']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fallback_images)

        print(f"⚠ Saved {len(fallback_images)} fallback images to {fallback_csv_path}")

    # Print summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total images found: {len(images)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (already processed): {skipped}")
    print(f"Skipped (wrong number of boxes): {skipped_too_many_boxes}")
    print(f"Total strips measured: {len(all_measurements)}")
    print(f"\nResults saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
