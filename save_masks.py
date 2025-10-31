#!/usr/bin/env python3
"""
Script to save SAM2.1 Tiny masks to output directory
"""

import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path
from sam2_tiny import SAM2TinyPredictor


def save_masks_to_directory(result, output_dir, image_name, mode="auto"):
    """
    Save masks to output directory organized by image name

    Args:
        result (dict): Result from SAM2 prediction
        output_dir (str): Base output directory
        image_name (str): Original image filename (without extension)
        mode (str): Prediction mode (auto, point, box)
    """
    # Create subdirectory for this image
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    masks = result['masks']

    if mode == "auto":
        # For automatic segmentation, masks is a list of dicts
        print(f"Saving {len(masks)} masks to {image_output_dir}/")

        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']

            # Save binary mask
            mask_filename = f"mask_{i:04d}.png"
            mask_path = os.path.join(image_output_dir, mask_filename)
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

            # Save mask metadata
            metadata_filename = f"mask_{i:04d}_metadata.txt"
            metadata_path = os.path.join(image_output_dir, metadata_filename)
            with open(metadata_path, 'w') as f:
                f.write(f"Area: {mask_data.get('area', 'N/A')}\n")
                f.write(f"Bbox: {mask_data.get('bbox', 'N/A')}\n")
                f.write(f"Predicted IOU: {mask_data.get('predicted_iou', 'N/A')}\n")
                f.write(f"Stability Score: {mask_data.get('stability_score', 'N/A')}\n")
                f.write(f"Crop Box: {mask_data.get('crop_box', 'N/A')}\n")
    else:
        # For point/box segmentation, masks is a numpy array
        print(f"Saving {len(masks)} masks to {image_output_dir}/")

        for i, mask in enumerate(masks):
            # Save binary mask
            mask_filename = f"mask_{i:04d}.png"
            mask_path = os.path.join(image_output_dir, mask_filename)
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

            # Save score if available
            if 'scores' in result:
                score = result['scores'][i]
                score_filename = f"mask_{i:04d}_score.txt"
                score_path = os.path.join(image_output_dir, score_filename)
                with open(score_path, 'w') as f:
                    f.write(f"Score: {score:.6f}\n")

    print(f"✅ Saved {len(masks)} masks to {image_output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Run SAM 2.1 Tiny and save masks")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", "-o", type=str, default="output_masks",
                        help="Output directory for masks")
    parser.add_argument("--mode", "-m", type=str, choices=["point", "box", "auto"], default="auto",
                        help="Segmentation mode: point, box, or auto")
    parser.add_argument("--points", "-p", type=str, help="Point coordinates as 'x1,y1;x2,y2' (for point mode)")
    parser.add_argument("--labels", "-l", type=str, help="Point labels as '1,0' (1=positive, 0=negative, for point mode)")
    parser.add_argument("--box", "-b", type=str, help="Box coordinates as 'x1,y1,x2,y2' (for box mode)")
    parser.add_argument("--model", type=str, default="models/sam2.1_hiera_tiny.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run on (cuda/cpu/auto)")

    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        sys.exit(1)

    # Get image name without extension
    image_path = Path(args.image)
    image_name = image_path.stem

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize predictor
    try:
        predictor = SAM2TinyPredictor(model_path=args.model, device=args.device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)

    # Run prediction based on mode
    try:
        if args.mode == "point":
            if not args.points or not args.labels:
                print("Error: Point mode requires --points and --labels arguments")
                sys.exit(1)

            # Parse points
            points_str = args.points.split(';')
            points = []
            for point_str in points_str:
                x, y = map(int, point_str.split(','))
                points.append([x, y])

            # Parse labels
            labels = list(map(int, args.labels.split(',')))

            if len(points) != len(labels):
                print("Error: Number of points must match number of labels")
                sys.exit(1)

            print(f"Running point-based segmentation...")
            result = predictor.predict_with_points(args.image, points, labels)
            save_masks_to_directory(result, args.output_dir, image_name, mode="point")

        elif args.mode == "box":
            if not args.box:
                print("Error: Box mode requires --box argument")
                sys.exit(1)

            # Parse box coordinates
            box_coords = list(map(int, args.box.split(',')))
            if len(box_coords) != 4:
                print("Error: Box coordinates must be in format 'x1,y1,x2,y2'")
                sys.exit(1)

            print(f"Running box-based segmentation...")
            result = predictor.predict_with_box(args.image, box_coords)
            save_masks_to_directory(result, args.output_dir, image_name, mode="box")

        elif args.mode == "auto":
            print(f"Running automatic segmentation...")
            result = predictor.predict_everything(args.image)
            save_masks_to_directory(result, args.output_dir, image_name, mode="auto")

        print(f"\n✨ Segmentation completed successfully!")
        print(f"📁 Masks saved to: {os.path.join(args.output_dir, image_name)}/")

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
