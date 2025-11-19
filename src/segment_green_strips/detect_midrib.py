#!/usr/bin/env python3
"""
Script to detect and segment the midrib (central vein) in sorghum leaf images.
The midrib is the bright central structure running between the two green leaf lamina strips.
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path

# Set matplotlib backend to non-interactive (must be before pyplot import)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def detect_green_strips(image):
    """
    Detect green strips in the image using color thresholding and spatial analysis.
    Adapted from detect_green_strips.py

    Args:
        image: BGR image

    Returns:
        list of bounding boxes [x1, y1, x2, y2], green mask
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for green color
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphological operations to connect leaf regions
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_horizontal)

    # Clean up small noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours
    candidates = []
    min_area = 5000
    img_height, img_width = image.shape[:2]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if (w * h) > 0 else 0

            score = area
            if aspect_ratio > 2:
                score *= 1.5
            elif aspect_ratio > 4:
                score *= 2.0
            if extent > 0.5:
                score *= 1.3
            if area > 50000:
                score *= 1.2

            candidates.append({
                'box': [x, y, x + w, y + h],
                'area': area,
                'score': score,
                'centroid_y': y + h/2
            })

    # Sort by score
    candidates.sort(key=lambda c: c['score'], reverse=True)

    # Select top 2 candidates with spatial separation
    selected_boxes = []
    min_vertical_separation = img_height * 0.1

    for candidate in candidates:
        is_separated = True
        for selected in selected_boxes:
            y_diff = abs(candidate['centroid_y'] - selected['centroid_y'])
            if y_diff < min_vertical_separation:
                is_separated = False
                break

        if is_separated:
            selected_boxes.append(candidate)

        if len(selected_boxes) >= 2:
            break

    # Extract boxes and sort top to bottom
    boxes = [b['box'] for b in selected_boxes]
    boxes.sort(key=lambda b: b[1])

    return boxes, mask, hsv


def create_background_mask(image, hsv):
    """
    Create a mask identifying white/bright background pixels to exclude from midrib detection.

    Args:
        image: Original BGR image
        hsv: HSV version of image

    Returns:
        Binary mask where 255 = background (to be excluded), 0 = potential leaf tissue
    """
    h, w = image.shape[:2]

    # Convert to LAB for better brightness analysis
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Method 1: High brightness + low saturation = white background
    # White pixels have high V (brightness) and low S (saturation)
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    # Background is typically: very bright (L > 200 or V > 220) AND low saturation (S < 50)
    bright_mask = (l_channel > 200) | (v_channel > 220)
    low_sat_mask = s_channel < 50
    white_background = bright_mask & low_sat_mask

    # Method 2: Check for very high RGB values (close to 255)
    b, g, r = cv2.split(image)
    very_bright = (r > 240) & (g > 240) & (b > 240)

    # Combine both methods
    background_mask = (white_background | very_bright).astype(np.uint8) * 255

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)

    return background_mask


def detect_midrib_region(image, hsv, boxes, green_mask):
    """
    Detect the midrib region between two green leaf strips.

    Args:
        image: Original BGR image
        hsv: HSV version of image
        boxes: List of 2 bounding boxes for the green strips [x1, y1, x2, y2]
        green_mask: Binary mask of green regions

    Returns:
        Binary mask of the midrib (255 = midrib, 0 = background)
    """
    h, w = image.shape[:2]
    midrib_mask = np.zeros((h, w), dtype=np.uint8)

    if len(boxes) < 2:
        print("  Warning: Need exactly 2 green strips to detect midrib")
        return midrib_mask

    # Get the two strip boxes
    box1, box2 = boxes[0], boxes[1]
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Define the region of interest (ROI) between the two strips
    # The midrib should be in the gap between the two green strips
    roi_y1 = max(0, y2_1 - 20)  # Extend slightly into first strip
    roi_y2 = min(h, y1_2 + 20)  # Extend slightly into second strip
    roi_x1 = max(x1_1, x1_2)
    roi_x2 = min(x2_1, x2_2)

    # Handle case where boxes don't overlap horizontally
    if roi_x1 >= roi_x2:
        roi_x1 = min(x1_1, x1_2)
        roi_x2 = max(x2_1, x2_2)

    # Validate ROI dimensions
    if roi_y1 >= roi_y2 or roi_x1 >= roi_x2:
        print(f"  Warning: Invalid ROI dimensions: x=[{roi_x1}, {roi_x2}], y=[{roi_y1}, {roi_y2}]")
        print(f"  Box1: y=[{y1_1}, {y2_1}], Box2: y=[{y1_2}, {y2_2}]")
        # Use a wider margin or full space between boxes
        roi_y1 = max(0, y2_1)
        roi_y2 = min(h, y1_2)
        if roi_y1 >= roi_y2:
            # Boxes may be overlapping or touching - use full image width
            roi_y1 = 0
            roi_y2 = h
            print(f"  Using full image height for ROI")

    # Ensure minimum ROI size
    if roi_y2 - roi_y1 < 10:
        print(f"  Warning: ROI height too small ({roi_y2 - roi_y1}px), expanding...")
        mid_y = (roi_y1 + roi_y2) // 2
        roi_y1 = max(0, mid_y - 50)
        roi_y2 = min(h, mid_y + 50)

    if roi_x2 - roi_x1 < 10:
        print(f"  Warning: ROI width too small ({roi_x2 - roi_x1}px), expanding...")
        mid_x = (roi_x1 + roi_x2) // 2
        roi_x1 = max(0, mid_x - 50)
        roi_x2 = min(w, mid_x + 50)

    print(f"  ROI between strips: x=[{roi_x1}, {roi_x2}], y=[{roi_y1}, {roi_y2}]")

    # Step 1: Create background mask to exclude white background
    print("  Creating background exclusion mask...")
    background_mask = create_background_mask(image, hsv)
    roi_background = background_mask[roi_y1:roi_y2, roi_x1:roi_x2]

    # Step 2: Convert to LAB color space for better brightness/color separation
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    roi_lab = lab[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_l = roi_lab[:, :, 0]  # L channel (lightness)
    roi_a = roi_lab[:, :, 1]  # A channel (green-red)
    roi_b = roi_lab[:, :, 2]  # B channel (blue-yellow)

    # Extract HSV ROI
    roi_hsv = hsv[roi_y1:roi_y2, roi_x1:roi_x2]
    h_channel = roi_hsv[:, :, 0]
    s_channel = roi_hsv[:, :, 1]
    v_channel = roi_hsv[:, :, 2]

    # Step 3: Compute reference statistics from green lamina
    green_pixels_l = lab[:, :, 0][green_mask > 0]
    green_pixels_a = lab[:, :, 1][green_mask > 0]
    green_pixels_b = lab[:, :, 2][green_mask > 0]
    green_pixels_v = hsv[:, :, 2][green_mask > 0]

    if len(green_pixels_v) == 0:
        print("  Warning: No green pixels found for reference")
        return midrib_mask, background_mask

    mean_green_l = np.mean(green_pixels_l)
    std_green_l = np.std(green_pixels_l)
    mean_green_v = np.mean(green_pixels_v)
    std_green_v = np.std(green_pixels_v)
    mean_green_a = np.mean(green_pixels_a)
    mean_green_b = np.mean(green_pixels_b)

    print(f"  Green lamina: L={mean_green_l:.1f}±{std_green_l:.1f}, V={mean_green_v:.1f}±{std_green_v:.1f}")
    print(f"  Green lamina: A={mean_green_a:.1f}, B={mean_green_b:.1f}")

    # Step 4: Brightness-based filtering
    # Midrib should be significantly brighter than green lamina
    # Midrib is typically 1.5-2.5x brighter than lamina
    # Reduced from 2.0*std to 1.5*std to catch more midribs
    l_threshold = min(200, mean_green_l + 1.5 * std_green_l)
    v_threshold = min(220, mean_green_v + 1.5 * std_green_v)

    print(f"  Midrib thresholds (STRICT): L>{l_threshold:.1f}, V>{v_threshold:.1f}")

    # Brightness mask: much brighter than lamina, but not pure white
    brightness_mask = ((roi_l > l_threshold) & (roi_l < 240) &
                      (v_channel > v_threshold) & (v_channel < 250))

    # Step 5: COLOR-based filtering (more permissive)
    # Key insight: Brightness is the most reliable indicator
    # Color helps discriminate, but shouldn't be overly strict

    # Get saturation stats
    mean_green_s = np.mean(hsv[:,:,1][green_mask > 0])
    std_green_s = np.std(hsv[:,:,1][green_mask > 0])

    # Adaptive saturation threshold
    # If lamina has very high saturation (>100), be more permissive
    if mean_green_s > 100:
        max_saturation = min(80, mean_green_s * 0.6)
    else:
        max_saturation = min(70, mean_green_s * 0.55)

    # Broader hue range to accommodate different lighting conditions
    # Include yellow-orange (10°) to yellow-green (70°)
    hue_mask = (h_channel >= 8) & (h_channel <= 75)

    # Create multiple color criteria (use OR logic for more permissiveness)
    # Option 1: Low saturation (typical midrib)
    low_sat_mask = (s_channel < max_saturation) & (s_channel > 5) & hue_mask

    # Option 2: Moderate saturation but much brighter (some midribs retain color)
    moderate_sat_mask = (s_channel < max_saturation * 1.5) & (roi_l > l_threshold + 20) & hue_mask

    # Combine color options with OR
    color_mask = low_sat_mask | moderate_sat_mask

    print(f"  Green saturation: S={mean_green_s:.1f}±{std_green_s:.1f}")
    print(f"  Color thresholds: S<{max_saturation:.1f} (or S<{max_saturation*1.5:.1f} if very bright), H=8-75")

    # Step 6: Exclude high-saturation green lamina regions
    # More conservative: only exclude pixels that are clearly green lamina
    lamina_threshold = max(50, mean_green_s * 0.7)  # Adaptive lamina threshold
    lamina_mask = ((s_channel > lamina_threshold) & (h_channel >= 35) & (h_channel <= 65))
    not_lamina = ~lamina_mask

    # Step 7: Combine constraints
    # Strategy: Brightness is REQUIRED, but be flexible on color
    not_background = (roi_background == 0)

    # Primary: Bright + good color + not lamina
    combined_strict = (brightness_mask & color_mask & not_lamina & not_background)

    # Fallback: If strict filter gets very few pixels, allow bright + not-lamina (ignore color)
    combined_relaxed = (brightness_mask & not_lamina & not_background)

    combined_strict_count = np.sum(combined_strict)
    combined_relaxed_count = np.sum(combined_relaxed)

    # Use strict filter if it finds reasonable amount, otherwise use relaxed
    if combined_strict_count > 100:
        combined = combined_strict.astype(np.uint8) * 255
        mode = "strict"
    elif combined_relaxed_count > 100:
        combined = combined_relaxed.astype(np.uint8) * 255
        mode = "relaxed (color-agnostic)"
    else:
        combined = combined_strict.astype(np.uint8) * 255
        mode = "strict (insufficient pixels)"

    print(f"  Background pixels in ROI: {np.sum(roi_background > 0)}")
    print(f"  Brightness-filtered pixels: {np.sum(brightness_mask > 0)}")
    print(f"  Color-filtered pixels: {np.sum(color_mask > 0)}")
    print(f"  Not-lamina pixels: {np.sum(not_lamina > 0)}")
    print(f"  Combined pixels ({mode}): {np.sum(combined > 0)}")

    # Step 8: Morphological cleanup - emphasize THIN horizontal structures
    # First, close gaps along horizontal direction (midrib is continuous)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_horizontal, iterations=2)

    # Then remove small isolated noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)

    # Step 9: Select best component based on SHAPE, POSITION, and SIZE
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)

    if num_labels > 1:  # More than just background
        roi_height = roi_y2 - roi_y1
        roi_width = roi_x2 - roi_x1
        roi_area = roi_height * roi_width
        roi_center_y = roi_height / 2

        # First pass: filter out components that are clearly too large to be midrib
        # Midrib should be at most 10% of ROI area
        max_midrib_area = roi_area * 0.10

        best_component = None
        best_score = -1

        candidates = []

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            # Skip components that are too large
            if area > max_midrib_area:
                continue

            # Calculate features
            aspect_ratio = w / h if h > 0 else 0
            centroid_y = centroids[i][1]

            # Distance from ROI center (midrib should be centered)
            center_distance = abs(centroid_y - roi_center_y)
            center_score = 1.0 - (center_distance / roi_height)

            # Aspect ratio score (midrib is very horizontal)
            aspect_score = min(1.0, aspect_ratio / 10.0)  # Prefer aspect ratio > 10

            # Width score (midrib spans most of leaf width)
            width_coverage = w / roi_width
            width_score = min(1.0, width_coverage)

            # Thinness score (midrib is relatively thin)
            # Use absolute height threshold to handle cases where ROI is abnormally large
            absolute_height = h
            if absolute_height < 20:  # Very thin
                thinness_score = 1.0
            elif absolute_height < 50:  # Moderately thin
                thinness_score = 0.7
            elif absolute_height < 100:  # Acceptable
                thinness_score = 0.4
            else:  # Too thick
                relative_height = h / roi_height
                thinness_score = max(0.0, 1.0 - relative_height * 5)

            # Combined score
            score = (center_score * 0.25 +
                    aspect_score * 0.35 +  # Increased weight on aspect ratio
                    width_score * 0.25 +
                    thinness_score * 0.15)

            # More permissive minimum requirements:
            # - Must be somewhat horizontal (AR >= 2.0, lowered from 2.5)
            # - Must span reasonable width (>= 15% of ROI width OR >150 pixels, lowered from 20%/200px)
            # - Must not be too thick (height < 200 pixels, increased from 150px)
            meets_requirements = (aspect_ratio >= 2.0 and
                                (width_coverage >= 0.15 or w >= 150) and
                                absolute_height < 200)

            if meets_requirements:
                candidates.append({
                    'id': i,
                    'area': area,
                    'score': score,
                    'ar': aspect_ratio,
                    'center': center_score,
                    'width': width_coverage,
                    'thin': thinness_score,
                    'height': absolute_height
                })

        # Sort candidates by score
        candidates.sort(key=lambda c: c['score'], reverse=True)

        # Print top candidates
        for candidate in candidates[:10]:
            print(f"    Component {candidate['id']}: area={candidate['area']}, " +
                  f"AR={candidate['ar']:.1f}, h={candidate['height']}px, " +
                  f"center={candidate['center']:.2f}, width={candidate['width']:.2f}, " +
                  f"thin={candidate['thin']:.2f}, score={candidate['score']:.3f}")

        if len(candidates) > 0:
            best_component = candidates[0]['id']
            best_score = candidates[0]['score']
            combined = (labels == best_component).astype(np.uint8) * 255
            print(f"  Selected component {best_component} with score {best_score:.3f}")
        else:
            # No components met requirements - try to diagnose why and use fallback
            num_too_large = np.sum([stats[i][4] > max_midrib_area for i in range(1, num_labels)])

            # Check shape requirements for all components
            shape_failures = []
            for i in range(1, min(num_labels, 6)):  # Check first 5 components
                x, y, w, h, area = stats[i]
                if area > max_midrib_area:
                    continue
                ar = w / h if h > 0 else 0
                wc = w / roi_width
                shape_failures.append(f"comp{i}: AR={ar:.1f}(<2.0?), w={wc*100:.1f}%(<15%?), h={h}px(>200?)")

            print(f"  Warning: No component met requirements (total: {num_labels-1}, too_large: {num_too_large})")
            if shape_failures:
                print(f"  Shape failures: {'; '.join(shape_failures[:3])}")

            # Fallback: Select the most horizontal component that's not too large
            fallback_candidates = []
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                if area > max_midrib_area:
                    continue
                ar = w / h if h > 0 else 0
                if ar >= 1.5:  # Very relaxed: just needs to be wider than tall
                    fallback_candidates.append({'id': i, 'ar': ar, 'area': area})

            if fallback_candidates:
                # Sort by aspect ratio (most horizontal)
                fallback_candidates.sort(key=lambda c: c['ar'], reverse=True)
                best_component = fallback_candidates[0]['id']
                combined = (labels == best_component).astype(np.uint8) * 255
                print(f"  Fallback: Selected component {best_component} (AR={fallback_candidates[0]['ar']:.1f})")
            else:
                print(f"  Fallback failed: No horizontal components found")
                combined = np.zeros_like(combined)

    # Place the detected midrib in the full image mask
    midrib_mask[roi_y1:roi_y2, roi_x1:roi_x2] = combined

    # Final cleanup: slight dilation to make midrib more visible
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    midrib_mask = cv2.dilate(midrib_mask, kernel_final, iterations=1)

    # Calculate statistics
    midrib_area = np.sum(midrib_mask > 0)
    total_area = h * w
    midrib_percentage = (midrib_area / total_area) * 100
    print(f"  Midrib area: {midrib_area} pixels ({midrib_percentage:.2f}% of image)")

    return midrib_mask, background_mask


def create_visualization(image, boxes, green_mask, midrib_mask, output_path, background_mask=None):
    """
    Create a comprehensive visualization showing the detection process.

    Args:
        image: Original BGR image
        boxes: List of bounding boxes for green strips
        green_mask: Binary mask of green regions
        midrib_mask: Binary mask of midrib
        output_path: Path to save the visualization
        background_mask: Optional background exclusion mask for debugging
    """
    # Create figure with subplots (3x3 grid if background mask provided, otherwise 2x3)
    if background_mask is not None:
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Green strips with bounding boxes
    ax2 = fig.add_subplot(gs[0, 1])
    vis_boxes = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(vis_boxes, f"Strip {i+1}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    ax2.imshow(cv2.cvtColor(vis_boxes, cv2.COLOR_BGR2RGB))
    ax2.set_title('Detected Green Strips', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Green mask
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(green_mask, cmap='gray')
    ax3.set_title('Green Detection Mask', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Midrib mask
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(midrib_mask, cmap='gray')
    ax4.set_title('Midrib Mask', fontsize=14, fontweight='bold')
    ax4.axis('off')

    # Overlay: midrib on original
    ax5 = fig.add_subplot(gs[1, 1])
    overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    midrib_colored = np.zeros_like(overlay)
    midrib_colored[midrib_mask > 0] = [255, 0, 0]  # Red
    overlay = cv2.addWeighted(overlay, 0.7, midrib_colored, 0.3, 0)
    ax5.imshow(overlay)
    ax5.set_title('Midrib Overlay (Red)', fontsize=14, fontweight='bold')
    ax5.axis('off')

    # Combined: all detections
    ax6 = fig.add_subplot(gs[1, 2])
    combined = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    # Green strips in green
    green_colored = np.zeros_like(combined)
    green_colored[green_mask > 0] = [0, 255, 0]
    combined = cv2.addWeighted(combined, 0.7, green_colored, 0.3, 0)
    # Midrib in red
    midrib_colored = np.zeros_like(combined)
    midrib_colored[midrib_mask > 0] = [255, 0, 0]
    combined = cv2.addWeighted(combined, 0.8, midrib_colored, 0.2, 0)
    # Draw bounding boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
    ax6.imshow(combined)
    ax6.set_title('All Detections', fontsize=14, fontweight='bold')
    ax6.axis('off')

    # Add background mask visualization if provided
    if background_mask is not None:
        # Background mask
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.imshow(background_mask, cmap='gray')
        ax7.set_title('Background Exclusion Mask', fontsize=14, fontweight='bold')
        ax7.axis('off')

        # Inverted background (leaf tissue only)
        ax8 = fig.add_subplot(gs[2, 1])
        leaf_only = cv2.bitwise_not(background_mask)
        ax8.imshow(leaf_only, cmap='gray')
        ax8.set_title('Leaf Tissue Mask (Not Background)', fontsize=14, fontweight='bold')
        ax8.axis('off')

        # Midrib with background highlighted
        ax9 = fig.add_subplot(gs[2, 2])
        debug_vis = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        # Show background in blue
        background_colored = np.zeros_like(debug_vis)
        background_colored[background_mask > 0] = [100, 100, 255]
        debug_vis = cv2.addWeighted(debug_vis, 0.7, background_colored, 0.3, 0)
        # Show midrib in red
        midrib_colored = np.zeros_like(debug_vis)
        midrib_colored[midrib_mask > 0] = [255, 0, 0]
        debug_vis = cv2.addWeighted(debug_vis, 0.8, midrib_colored, 0.2, 0)
        ax9.imshow(debug_vis)
        ax9.set_title('Midrib (Red) + Background (Blue)', fontsize=14, fontweight='bold')
        ax9.axis('off')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {output_path}")


def process_image(image_path, output_dir, visualize=True):
    """
    Process a single image to detect and segment the midrib.

    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        visualize: Whether to create visualization

    Returns:
        midrib_mask: Binary mask of the midrib
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Error: Could not load image {image_path}")
        return None

    image_name = Path(image_path).stem
    print(f"\nProcessing: {image_name}")

    # Detect green strips
    print("  Detecting green strips...")
    boxes, green_mask, hsv = detect_green_strips(image)

    if len(boxes) < 2:
        print(f"  Warning: Only detected {len(boxes)} green strip(s), need 2")
        return None

    print(f"  Found {len(boxes)} green strips")

    # Detect midrib
    print("  Detecting midrib...")
    midrib_mask, background_mask = detect_midrib_region(image, hsv, boxes, green_mask)

    # Save binary mask
    mask_filename = f"{image_name}_midrib_mask.png"
    mask_path = os.path.join(output_dir, mask_filename)
    cv2.imwrite(mask_path, midrib_mask)
    print(f"  Midrib mask saved to {mask_path}")

    # Create visualization
    if visualize:
        vis_filename = f"{image_name}_midrib_visualization.png"
        vis_path = os.path.join(output_dir, vis_filename)
        create_visualization(image, boxes, green_mask, midrib_mask, vis_path, background_mask)

    return midrib_mask


def main():
    parser = argparse.ArgumentParser(
        description="Detect and segment midrib (central vein) in sorghum leaf images"
    )
    parser.add_argument(
        "--image", "-i", type=str,
        help="Path to single input image (or use --image-list)"
    )
    parser.add_argument(
        "--image-list", "-l", type=str,
        help="Path to text file with list of image paths (one per line)"
    )
    parser.add_argument(
        "--images", nargs='+',
        help="List of image paths"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="midrib_output",
        help="Output directory for masks and visualizations"
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip visualization generation (faster)"
    )

    args = parser.parse_args()

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_list:
        with open(args.image_list, 'r') as f:
            image_paths.extend([line.strip() for line in f if line.strip()])
    if args.images:
        image_paths.extend(args.images)

    if len(image_paths) == 0:
        print("Error: No images specified. Use --image, --image-list, or --images")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Processing {len(image_paths)} image(s)...")

    # Process each image
    success_count = 0
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"\nWarning: Image not found: {image_path}")
            continue

        try:
            mask = process_image(image_path, args.output_dir, visualize=not args.no_viz)
            if mask is not None:
                success_count += 1
        except Exception as e:
            print(f"  Error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"✅ Successfully processed {success_count}/{len(image_paths)} images")
    print(f"Results saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
