#!/usr/bin/env python3
"""
Simple command-line interface for SAM 2.1 Tiny model
"""

import argparse
import os
import sys
from sam2_tiny import SAM2TinyPredictor

def main():
    parser = argparse.ArgumentParser(description="Run SAM 2.1 Tiny model on images")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--mode", "-m", type=str, choices=["point", "box", "auto"], default="auto",
                        help="Segmentation mode: point, box, or auto")
    parser.add_argument("--points", "-p", type=str, help="Point coordinates as 'x1,y1;x2,y2' (for point mode)")
    parser.add_argument("--labels", "-l", type=str, help="Point labels as '1,0' (1=positive, 0=negative, for point mode)")
    parser.add_argument("--box", "-b", type=str, help="Box coordinates as 'x1,y1,x2,y2' (for box mode)")
    parser.add_argument("--output", "-o", type=str, help="Output path for visualization")
    parser.add_argument("--model", type=str, default="models/sam2.1_hiera_tiny.pt", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (cuda/cpu/auto)")

    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        sys.exit(1)

    # Set device
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize predictor
    try:
        predictor = SAM2TinyPredictor(model_path=args.model, device=device)
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

            result = predictor.predict_with_points(args.image, points, labels)
            predictor.visualize_prediction(result, args.output)

        elif args.mode == "box":
            if not args.box:
                print("Error: Box mode requires --box argument")
                sys.exit(1)

            # Parse box coordinates
            box_coords = list(map(int, args.box.split(',')))
            if len(box_coords) != 4:
                print("Error: Box coordinates must be in format 'x1,y1,x2,y2'")
                sys.exit(1)

            result = predictor.predict_with_box(args.image, box_coords)
            predictor.visualize_prediction(result, args.output)

        elif args.mode == "auto":
            result = predictor.predict_everything(args.image)
            predictor.visualize_everything_prediction(result, args.output)

    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

    print("Segmentation completed successfully!")

if __name__ == "__main__":
    main()