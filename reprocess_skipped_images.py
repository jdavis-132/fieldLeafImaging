#!/usr/bin/env python3
"""
Reprocess images that were previously skipped using the improved detection algorithm
"""

import argparse
import csv
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Reprocess skipped images with improved detection"
    )
    parser.add_argument(
        "--skipped-csv",
        type=str,
        default="batch_green_strips_output/skipped_images.csv",
        help="Path to skipped images CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reprocessed_output",
        help="Output directory for reprocessed images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/sam2.1_hiera_tiny.pt",
        help="Path to SAM2 model"
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only visualize boxes, don't run SAM2"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing)"
    )

    args = parser.parse_args()

    # Check if CSV exists
    if not os.path.exists(args.skipped_csv):
        print(f"Error: CSV file '{args.skipped_csv}' not found.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Read skipped images from CSV
    skipped_images = []
    with open(args.skipped_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            skipped_images.append(row)

    print(f"Found {len(skipped_images)} skipped images in {args.skipped_csv}")

    if args.max_images:
        skipped_images = skipped_images[:args.max_images]
        print(f"Processing first {args.max_images} images (--max-images set)\n")

    # Track results
    successful = 0
    failed = 0
    still_wrong_count = 0
    now_correct = 0

    # Create CSV to track reprocessing results
    results_csv = os.path.join(args.output_dir, "reprocessing_results.csv")
    results_file = open(results_csv, 'w', newline='')
    results_writer = csv.DictWriter(
        results_file,
        fieldnames=['image_path', 'subdirectory', 'image_name',
                   'old_num_boxes', 'new_num_boxes', 'status', 'reason']
    )
    results_writer.writeheader()

    # Process each image
    for i, row in enumerate(skipped_images, 1):
        image_path = row['image_path']
        subdir_id = row['subdirectory']
        image_name = row['image_name']
        old_num_boxes = int(row['num_boxes_detected'])
        old_reason = row['reason']

        print(f"[{i}/{len(skipped_images)}] Processing: {image_name}")
        print(f"  Original detection: {old_num_boxes} boxes ({old_reason})")

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"  ✗ Error: Image not found at {image_path}")
            failed += 1
            results_writer.writerow({
                'image_path': image_path,
                'subdirectory': subdir_id,
                'image_name': image_name,
                'old_num_boxes': old_num_boxes,
                'new_num_boxes': -1,
                'status': 'failed',
                'reason': 'image_not_found'
            })
            continue

        # Build command
        cmd = [
            'python3',
            'detect_green_strips.py',
            '--image', image_path,
            '--output-dir', args.output_dir,
            '--device', args.device,
            '--model', args.model
        ]

        if args.visualize_only:
            cmd.append('--visualize-only')

        try:
            # Run detection
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                # Parse output to get number of boxes detected
                output_lines = result.stdout.split('\n')
                new_num_boxes = None

                for line in output_lines:
                    if 'Found' in line and 'green strips' in line:
                        # Extract number from "Found X green strips"
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                new_num_boxes = int(parts[1])
                            except (ValueError, IndexError):
                                pass
                        break

                if new_num_boxes is None:
                    new_num_boxes = -1
                    status = 'unknown'
                    reason = 'could_not_parse_output'
                elif new_num_boxes == 2:
                    status = 'success_now_correct'
                    reason = 'now_detects_2_boxes'
                    now_correct += 1
                else:
                    status = 'success_still_wrong'
                    reason = f'still_detects_{new_num_boxes}_boxes'
                    still_wrong_count += 1

                print(f"  ✓ New detection: {new_num_boxes} boxes ({status})")
                successful += 1

                results_writer.writerow({
                    'image_path': image_path,
                    'subdirectory': subdir_id,
                    'image_name': image_name,
                    'old_num_boxes': old_num_boxes,
                    'new_num_boxes': new_num_boxes,
                    'status': status,
                    'reason': reason
                })

            else:
                print(f"  ✗ Error: {result.stderr[:200]}")
                failed += 1
                results_writer.writerow({
                    'image_path': image_path,
                    'subdirectory': subdir_id,
                    'image_name': image_name,
                    'old_num_boxes': old_num_boxes,
                    'new_num_boxes': -1,
                    'status': 'failed',
                    'reason': 'detection_error'
                })

        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout processing image")
            failed += 1
            results_writer.writerow({
                'image_path': image_path,
                'subdirectory': subdir_id,
                'image_name': image_name,
                'old_num_boxes': old_num_boxes,
                'new_num_boxes': -1,
                'status': 'failed',
                'reason': 'timeout'
            })
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            failed += 1
            results_writer.writerow({
                'image_path': image_path,
                'subdirectory': subdir_id,
                'image_name': image_name,
                'old_num_boxes': old_num_boxes,
                'new_num_boxes': -1,
                'status': 'failed',
                'reason': str(e)
            })

    # Close results file
    results_file.close()

    # Print summary
    print("\n" + "="*60)
    print("REPROCESSING SUMMARY")
    print("="*60)
    print(f"Total images reprocessed: {len(skipped_images)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Now correct (2 boxes): {now_correct}")
    print(f"Still wrong count: {still_wrong_count}")
    if successful > 0:
        improvement_rate = (now_correct / successful) * 100
        print(f"Improvement rate: {improvement_rate:.1f}%")
    print(f"\nResults saved to: {results_csv}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
