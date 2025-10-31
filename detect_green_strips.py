#!/usr/bin/env python3
"""
Script to detect green leaf strips and segment them with SAM2
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from sam2_tiny import SAM2TinyPredictor


def find_midrib_location(image, hsv, box, mask):
    """
    Find the midrib vein location within a box using brightness analysis

    Args:
        image: Original BGR image
        hsv: HSV version of image
        box: Bounding box [x1, y1, x2, y2]
        mask: Green detection mask

    Returns:
        y-coordinate of midrib, or None if not found
    """
    x1, y1, x2, y2 = box

    # Extract ROI from Value channel (brightness)
    roi_v = hsv[y1:y2, x1:x2, 2]  # V channel
    roi_mask = mask[y1:y2, x1:x2]

    # Only consider green pixels
    roi_v_masked = roi_v.copy()
    roi_v_masked[roi_mask == 0] = 0

    # Compute horizontal projection (average brightness per row)
    row_brightness = []
    for i in range(roi_v_masked.shape[0]):
        row = roi_v_masked[i, :]
        green_pixels = row[row > 0]
        if len(green_pixels) > 0:
            row_brightness.append(np.mean(green_pixels))
        else:
            row_brightness.append(0)

    row_brightness = np.array(row_brightness)

    # Apply smoothing to reduce noise
    try:
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(row_brightness, sigma=5)
    except (ImportError, Exception):
        # Fallback if scipy not available or incompatible
        kernel_size = 11
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(row_brightness, kernel, mode='same')

    # Find peaks (bright regions likely to be midrib)
    # The midrib should be in the middle third of the box
    mid_start = len(smoothed) // 3
    mid_end = 2 * len(smoothed) // 3
    mid_region = smoothed[mid_start:mid_end]

    if len(mid_region) == 0:
        return None

    # Find the brightest row in the middle region
    local_max_idx = np.argmax(mid_region)
    global_max_idx = mid_start + local_max_idx

    # Check if this peak is significantly brighter than average
    avg_brightness = np.mean(smoothed[smoothed > 0]) if np.any(smoothed > 0) else 0
    peak_brightness = smoothed[global_max_idx]

    # Midrib should be at least 5% brighter than average
    if peak_brightness > avg_brightness * 1.05:
        return y1 + global_max_idx

    return None


def detect_green_strips(image):
    """
    Detect green strips in the image using improved color thresholding and spatial analysis

    Args:
        image: BGR image

    Returns:
        list of bounding boxes [x1, y1, x2, y2]
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for green color
    # Adjust these values based on your specific green
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply aggressive morphological operations to bridge gaps and merge fragments
    # Use larger kernels to connect discontinuous leaf regions
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    # Additional horizontal closing to connect leaf strips along their length
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_horizontal)

    # Clean up small noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and score contours
    candidates = []
    min_area = 5000  # Reduced threshold to catch smaller fragments

    img_height, img_width = image.shape[:2]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate features for scoring
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if (w * h) > 0 else 0

            # Horizontal strips should have high width-to-height ratio
            # Score based on how well it matches expected strip characteristics
            score = area  # Start with area as base score

            # Prefer wider, more horizontal shapes (aspect ratio > 2)
            if aspect_ratio > 2:
                score *= 1.5
            elif aspect_ratio > 4:
                score *= 2.0

            # Prefer shapes that fill their bounding box well
            if extent > 0.5:
                score *= 1.3

            # Prefer larger boxes (they're more likely to be complete strips)
            if area > 50000:
                score *= 1.2

            candidates.append({
                'box': [x, y, x + w, y + h],
                'area': area,
                'score': score,
                'centroid_y': y + h/2,
                'aspect_ratio': aspect_ratio,
                'extent': extent
            })

    # Sort candidates by score (descending)
    candidates.sort(key=lambda c: c['score'], reverse=True)

    # Select top candidates with spatial separation
    selected_boxes = []
    min_vertical_separation = img_height * 0.1  # At least 10% of image height apart

    for candidate in candidates:
        # Check if this candidate is spatially separated from already selected boxes
        is_separated = True
        for selected in selected_boxes:
            # Check vertical separation between centroids
            y_diff = abs(candidate['centroid_y'] - selected['centroid_y'])
            if y_diff < min_vertical_separation:
                is_separated = False
                break

        if is_separated:
            selected_boxes.append(candidate)

        # Stop once we have 2 boxes
        if len(selected_boxes) >= 2:
            break

    # If we have fewer than 2 boxes, try to split the largest box
    if len(selected_boxes) == 1 and len(candidates) == 1:
        candidate = candidates[0]
        box = candidate['box']
        x1, y1, x2, y2 = box

        # Strategy 1: Try to find midrib vein using brightness analysis
        # This handles the case where midrib is light green (not a gap)
        midrib_y = find_midrib_location(image, hsv, box, mask)

        if midrib_y is not None:
            # Found a bright midrib - split there
            split_y = int(midrib_y)
            # Add small margin around midrib
            margin = int((y2 - y1) * 0.02)  # 2% of box height
            box1 = [x1, y1, x2, max(y1, split_y - margin)]
            box2 = [x1, min(y2, split_y + margin), x2, y2]

            selected_boxes = [
                {'box': box1, 'area': (x2-x1) * (box1[3]-box1[1]), 'score': 1.0},
                {'box': box2, 'area': (x2-x1) * (box2[3]-box2[1]), 'score': 1.0}
            ]
            print(f"  Split using midrib detection at y={split_y}")
        else:
            # Strategy 2: Try to detect gap between strips using projection
            roi = mask[y1:y2, x1:x2]

            # Compute horizontal projection (sum of pixels in each row)
            h_projection = np.sum(roi, axis=1) / 255

            # Find local minima that could indicate a gap between strips
            try:
                from scipy.signal import find_peaks
                # Invert to find valleys
                valleys, _ = find_peaks(-h_projection, prominence=roi.shape[1] * 0.1)

                if len(valleys) > 0:
                    # Find the most prominent valley (likely gap between strips)
                    mid_valley = valleys[len(valleys)//2]
                    split_y = y1 + mid_valley

                    # Create two boxes by splitting horizontally
                    box1 = [x1, y1, x2, split_y]
                    box2 = [x1, split_y, x2, y2]

                    selected_boxes = [
                        {'box': box1, 'area': (x2-x1) * (split_y-y1), 'score': 1.0},
                        {'box': box2, 'area': (x2-x1) * (y2-split_y), 'score': 1.0}
                    ]
                    print(f"  Split using valley detection at y={split_y}")
            except ImportError:
                # scipy not available, use simple midpoint split
                if (y2 - y1) > img_height * 0.3:  # Only split if box is large enough
                    split_y = (y1 + y2) // 2
                    box1 = [x1, y1, x2, split_y]
                    box2 = [x1, split_y, x2, y2]

                    selected_boxes = [
                        {'box': box1, 'area': (x2-x1) * (split_y-y1), 'score': 1.0},
                        {'box': box2, 'area': (x2-x1) * (y2-split_y), 'score': 1.0}
                    ]
                    print(f"  Split using midpoint at y={split_y}")

    # If we still don't have exactly 2 boxes, handle edge cases
    boxes = [b['box'] for b in selected_boxes]

    # Check for nested/overlapping boxes (one box inside another)
    if len(boxes) == 2:
        box1, box2 = boxes
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Check if box2 is inside box1
        if (x1_2 >= x1_1 and x2_2 <= x2_1 and y1_2 >= y1_1 and y2_2 <= y2_1):
            # box2 is nested in box1 - this is the light-green midrib case
            # Split the larger box (box1) using midrib detection
            print(f"  Detected nested boxes - attempting to split larger box")
            midrib_y = find_midrib_location(image, hsv, box1, mask)

            if midrib_y is not None:
                split_y = int(midrib_y)
                margin = int((y2_1 - y1_1) * 0.02)
                box_top = [x1_1, y1_1, x2_1, max(y1_1, split_y - margin)]
                box_bottom = [x1_1, min(y2_1, split_y + margin), x2_1, y2_1]
                boxes = [box_top, box_bottom]
                print(f"  Split nested case using midrib at y={split_y}")
            else:
                # Fallback: split at midpoint
                split_y = (y1_1 + y2_1) // 2
                box_top = [x1_1, y1_1, x2_1, split_y]
                box_bottom = [x1_1, split_y, x2_1, y2_1]
                boxes = [box_top, box_bottom]
                print(f"  Split nested case at midpoint y={split_y}")

        # Check if box1 is inside box2 (less common but possible)
        elif (x1_1 >= x1_2 and x2_1 <= x2_2 and y1_1 >= y1_2 and y2_1 <= y2_2):
            print(f"  Detected nested boxes - attempting to split larger box")
            midrib_y = find_midrib_location(image, hsv, box2, mask)

            if midrib_y is not None:
                split_y = int(midrib_y)
                margin = int((y2_2 - y1_2) * 0.02)
                box_top = [x1_2, y1_2, x2_2, max(y1_2, split_y - margin)]
                box_bottom = [x1_2, min(y2_2, split_y + margin), x2_2, y2_2]
                boxes = [box_top, box_bottom]
                print(f"  Split nested case using midrib at y={split_y}")
            else:
                split_y = (y1_2 + y2_2) // 2
                box_top = [x1_2, y1_2, x2_2, split_y]
                box_bottom = [x1_2, split_y, x2_2, y2_2]
                boxes = [box_top, box_bottom]
                print(f"  Split nested case at midpoint y={split_y}")

    if len(boxes) > 2:
        # More than 2 - keep only top 2 by score
        boxes = boxes[:2]

    # Sort boxes top to bottom for consistency
    boxes.sort(key=lambda b: b[1])

    print(f"Found {len(boxes)} green strips")
    return boxes, mask


def visualize_boxes(image, boxes, output_path=None):
    """
    Visualize detected bounding boxes on the image
    """
    vis_image = image.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(vis_image, f"Strip {i+1}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Box visualization saved to {output_path}")

    return vis_image


def save_masks_from_boxes(predictor, image_path, boxes, output_dir, image_name):
    """
    Run SAM2 with box prompts and save individual masks
    """
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    print(f"\nSegmenting {len(boxes)} strips with SAM2...")

    for i, box in enumerate(boxes):
        print(f"Processing strip {i+1}/{len(boxes)}...")

        # Get segmentation from SAM2
        result = predictor.predict_with_box(image_path, box)

        # Save the mask (typically returns 1 mask for box prompt)
        mask = result['masks'][0]
        mask_filename = f"strip_{i:04d}_mask.png"
        mask_path = os.path.join(image_output_dir, mask_filename)
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

        # Save box coordinates
        box_filename = f"strip_{i:04d}_box.txt"
        box_path = os.path.join(image_output_dir, box_filename)
        with open(box_path, 'w') as f:
            f.write(f"Box: {box}\n")
            f.write(f"Score: {result['scores'][0]:.6f}\n")

        # Save visualization of this strip
        vis = result['image'].copy()
        cv2.rectangle(vis, (int(box[0]), int(box[1])),
                     (int(box[2]), int(box[3])), (0, 255, 0), 3)
        overlay = vis.copy()
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([0, 255, 0]) * 0.5
        vis_filename = f"strip_{i:04d}_vis.jpg"
        vis_path = os.path.join(image_output_dir, vis_filename)
        cv2.imwrite(vis_path, cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))

    print(f"\n✅ Saved {len(boxes)} strip masks to {image_output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Detect green strips and segment with SAM2")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", "-o", type=str, default="green_strips_output",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="models/sam2.1_hiera_tiny.pt",
                        help="Path to SAM2 model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--min-area", type=int, default=10000,
                        help="Minimum area for green strip detection")
    parser.add_argument("--visualize-only", action="store_true",
                        help="Only visualize boxes, don't run SAM2")

    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get image name
    image_name = Path(args.image).stem

    # Load image
    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect green strips
    print("Detecting green strips...")
    boxes, mask = detect_green_strips(image)

    if len(boxes) == 0:
        print("No green strips detected. Try adjusting the color thresholds.")
        return

    # Save detection mask
    mask_path = os.path.join(args.output_dir, f"{image_name}_green_mask.png")
    cv2.imwrite(mask_path, mask)
    print(f"Green detection mask saved to {mask_path}")

    # Visualize boxes
    box_vis_path = os.path.join(args.output_dir, f"{image_name}_boxes.jpg")
    visualize_boxes(image, boxes, box_vis_path)

    if args.visualize_only:
        print("Visualization complete (--visualize-only flag set)")
        return

    # Initialize SAM2
    print(f"\nInitializing SAM2 on {args.device.upper()}...")
    predictor = SAM2TinyPredictor(model_path=args.model, device=args.device)

    # Segment each strip with SAM2
    save_masks_from_boxes(predictor, image_rgb, boxes, args.output_dir, image_name)

    print(f"\n✨ Complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
