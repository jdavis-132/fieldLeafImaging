"""
LEAF SEGMENTATION PIPELINE - JAMES METHOD
==========================================

OBJECTIVE:
Segment sorghum leaves with severe anthracnose by removing the white background
instead of trying to detect the leaf (which has highly variable colors).

TWO-STAGE METHODOLOGY:
1. STAGE 1: Remove white background using connected components
2. STAGE 2: Identify the leaf as the largest connected component

KEY CONCEPTS TO UNDERSTAND:
---------------------------
- CONNECTED COMPONENTS: Groups of pixels of the same color that are
  touching each other. cv2.connectedComponents() labels each group with
  a different number.

- SEED POINTS: Starting points from which we search for the white background.
  James suggests using center-top and center-bottom because we know the
  white background is there in all images.

- PROBLEM INVERSION: Instead of searching "what is leaf?" (difficult),
  we search "what is NOT leaf?" (easy: the white background).
"""

import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys


# =============================================================================
# STAGE 1: REMOVE WHITE BACKGROUND
# =============================================================================

def remove_white_background(image, visualize=True):
    """
    STAGE 1 of James's algorithm: Remove the white background.

    DETAILED PROCESS:
    -----------------
    1. Convert image to grayscale (simplifies analysis)
    2. Create binary mask: bright pixels (background) = 1, dark pixels (leaf) = 0
    3. Find connected components (groups of white pixels)
    4. Identify background components using seed points
    5. Remove those components

    PARAMETERS:
    ----------
    image : numpy.ndarray
        Input BGR image (OpenCV format)
    visualize : bool
        If True, shows intermediate visualizations

    RETURNS:
    -------
    mask_no_background : numpy.ndarray
        Binary mask where background=0, possible_leaf=255
    debug_info : dict
        Debugging information with intermediate visualizations
    """
    print("\n" + "="*70)
    print("STAGE 1: REMOVING WHITE BACKGROUND")
    print("="*70)

    # Image dimensions
    height, width = image.shape[:2]
    print(f"Image dimensions: {width} x {height} pixels")

    # -------------------------------------------------------------------------
    # STEP 1.1: Convert to grayscale
    # -------------------------------------------------------------------------
    # Why? Simplifies analysis. Instead of 3 channels (R,G,B),
    # we work with 1 channel (intensity from 0-255).
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"✓ Image converted to grayscale")

    # -------------------------------------------------------------------------
    # STEP 1.2: Create binary mask of white background
    # -------------------------------------------------------------------------
    # Adaptive or fixed threshold? James suggests starting with fixed threshold.
    # White/beige background has high values (typically >200).
    # Leaf (even with spots) has lower values.

    # We'll use OTSU to find the optimal threshold automatically
    # OTSU is a method that finds the best threshold to separate two classes
    # (in our case: bright background vs dark leaf)
    threshold_value, white_mask = cv2.threshold(
        gray,
        0,          # Ignored when using OTSU
        255,        # Maximum value (white)
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"✓ OTSU threshold calculated: {threshold_value:.1f}")
    print(f"  Pixels > {threshold_value:.1f} are considered white background")

    # -------------------------------------------------------------------------
    # STEP 1.3: Find connected components
    # -------------------------------------------------------------------------
    # cv2.connectedComponents() labels each group of white pixels
    # that are connected to each other with a unique number.
    #
    # Returns:
    #   - num_labels: how many components were found (includes background=0)
    #   - labels: matrix where each pixel has the ID of its component
    num_labels, labels = cv2.connectedComponents(white_mask, connectivity=8)
    print(f"✓ Connected components found: {num_labels - 1}")
    print(f"  (connectivity=8 means pixels connect diagonally too)")

    # -------------------------------------------------------------------------
    # STEP 1.4: Identify background components using SEED POINTS
    # -------------------------------------------------------------------------
    # James says: use points at center-top and center-bottom
    # because we know the white background is there in all images.

    # Top seed point (upper middle)
    seed_top_x = width // 2
    seed_top_y = height // 4  # 25% from top
    label_top = labels[seed_top_y, seed_top_x]

    # Bottom seed point (lower middle)
    seed_bottom_x = width // 2
    seed_bottom_y = 3 * height // 4  # 75% from top
    label_bottom = labels[seed_bottom_y, seed_bottom_x]

    print(f"\n  Seed points to identify background:")
    print(f"  - Top: ({seed_top_x}, {seed_top_y}) → component {label_top}")
    print(f"  - Bottom: ({seed_bottom_x}, {seed_bottom_y}) → component {label_bottom}")

    # Components we identify as background
    background_labels = set([label_top, label_bottom])
    print(f"  Components marked as BACKGROUND: {background_labels}")

    # -------------------------------------------------------------------------
    # STEP 1.5: Remove background components
    # -------------------------------------------------------------------------
    # Create mask where we remove background components
    mask_no_background = np.ones_like(gray) * 255  # Start all white

    for bg_label in background_labels:
        mask_no_background[labels == bg_label] = 0  # Mark background as black

    # Count how many pixels remain
    pixels_remaining = np.sum(mask_no_background > 0)
    percentage = (pixels_remaining / (width * height)) * 100
    print(f"\n✓ Background removed")
    print(f"  Remaining pixels: {pixels_remaining:,} ({percentage:.1f}% of image)")

    # -------------------------------------------------------------------------
    # Prepare debugging info
    # -------------------------------------------------------------------------
    debug_info = {
        'gray': gray,
        'white_mask': white_mask,
        'threshold_value': threshold_value,
        'labels': labels,
        'num_labels': num_labels,
        'background_labels': background_labels,
        'seed_points': [(seed_top_x, seed_top_y), (seed_bottom_x, seed_bottom_y)]
    }

    return mask_no_background, debug_info


# =============================================================================
# STAGE 2: IDENTIFY THE LEAF
# =============================================================================

def identify_leaf(mask_no_background, image, visualize=True):
    """
    STAGE 2 of James's algorithm: Identify the leaf.

    DETAILED PROCESS:
    -----------------
    1. From mask without background, find connected components
    2. Calculate statistics for each component (area, bbox, etc.)
    3. Find the LARGEST component
    4. Verify it touches both sides (left AND right) of the image
    5. That's the leaf

    PARAMETERS:
    ----------
    mask_no_background : numpy.ndarray
        Binary mask without background (output of STAGE 1)
    image : numpy.ndarray
        Original BGR image
    visualize : bool
        If True, shows visualizations

    RETURNS:
    -------
    leaf_mask : numpy.ndarray
        Final binary mask of the leaf
    debug_info : dict
        Debugging information
    """
    print("\n" + "="*70)
    print("STAGE 2: IDENTIFYING THE LEAF")
    print("="*70)

    height, width = image.shape[:2]

    # -------------------------------------------------------------------------
    # STEP 2.1: Find connected components WITH STATISTICS
    # -------------------------------------------------------------------------
    # cv2.connectedComponentsWithStats() not only labels components,
    # it also calculates useful statistics for each one:
    #   - Area (number of pixels)
    #   - Bounding box (x, y, width, height)
    #   - Centroid (x, y)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_no_background.astype(np.uint8),
        connectivity=8
    )
    print(f"✓ Components found (without background): {num_labels - 1}")

    # -------------------------------------------------------------------------
    # STEP 2.2: Find the LARGEST component
    # -------------------------------------------------------------------------
    # stats[i, cv2.CC_STAT_AREA] is the area of component i
    # We ignore component 0 (always the black background)

    if num_labels <= 1:
        print("⚠ WARNING: No components found after removing background")
        return np.zeros_like(mask_no_background), {}

    # Calculate areas of all components (except 0=background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_component_idx = np.argmax(areas) + 1  # +1 because we ignore index 0
    largest_area = areas[largest_component_idx - 1]

    print(f"\n  Largest component: #{largest_component_idx}")
    print(f"  Area: {largest_area:,} pixels")

    # -------------------------------------------------------------------------
    # STEP 2.3: Verify it touches both sides
    # -------------------------------------------------------------------------
    # James says: the leaf should touch the left side AND the right side
    # Extract bounding box of largest component
    x = stats[largest_component_idx, cv2.CC_STAT_LEFT]
    y = stats[largest_component_idx, cv2.CC_STAT_TOP]
    w = stats[largest_component_idx, cv2.CC_STAT_WIDTH]
    h = stats[largest_component_idx, cv2.CC_STAT_HEIGHT]

    print(f"  Bounding box: x={x}, y={y}, width={w}, height={h}")

    # Check if it touches the edges
    touches_left = (x <= 10)  # 10 pixel margin
    touches_right = (x + w >= width - 10)

    print(f"\n  Edge verification:")
    print(f"  - Touches left side? {'✓ YES' if touches_left else '✗ NO'}")
    print(f"  - Touches right side? {'✓ YES' if touches_right else '✗ NO'}")

    if not (touches_left and touches_right):
        print("  ⚠ WARNING: Largest component does NOT touch both sides")
        print("    This could indicate a problem with the segmentation")

    # -------------------------------------------------------------------------
    # STEP 2.4: Create final leaf mask
    # -------------------------------------------------------------------------
    leaf_mask = np.zeros_like(mask_no_background)
    leaf_mask[labels == largest_component_idx] = 255

    pixels_leaf = np.sum(leaf_mask > 0)
    percentage = (pixels_leaf / (width * height)) * 100
    print(f"\n✓ Leaf identified")
    print(f"  Leaf pixels: {pixels_leaf:,} ({percentage:.1f}% of image)")

    # -------------------------------------------------------------------------
    # Prepare debugging info
    # -------------------------------------------------------------------------
    debug_info = {
        'labels': labels,
        'num_labels': num_labels,
        'largest_component_idx': largest_component_idx,
        'largest_area': largest_area,
        'bbox': (x, y, w, h),
        'touches_left': touches_left,
        'touches_right': touches_right,
        'centroids': centroids
    }

    return leaf_mask, debug_info


# =============================================================================
# EXTRACT MEASUREMENTS WITH REGIONPROPS
# =============================================================================

def extract_measurements(leaf_mask, image):
    """
    Extract leaf measurements using regionprops from scikit-image.

    James specifies: use major/minor axes (NOT perpendicular to frame).

    CONCEPTS:
    ---------
    - MAJOR AXIS: The longest line that fits inside the leaf
    - MINOR AXIS: The shortest line perpendicular to the major axis
    - ORIENTATION: Angle of the major axis relative to the horizontal axis

    These axes adjust to the SHAPE of the leaf, not the image frame.

    PARAMETERS:
    ----------
    leaf_mask : numpy.ndarray
        Binary mask of the leaf
    image : numpy.ndarray
        Original BGR image

    RETURNS:
    -------
    measurements : dict
        Dictionary with all measurements
    """
    print("\n" + "="*70)
    print("EXTRACTING MEASUREMENTS")
    print("="*70)

    # -------------------------------------------------------------------------
    # Use regionprops from scikit-image
    # -------------------------------------------------------------------------
    # regionprops analyzes the geometry of regions in binary images
    # and calculates many useful properties
    label_image = measure.label(leaf_mask)
    props = measure.regionprops(label_image)

    if len(props) == 0:
        print("⚠ ERROR: No regions found in mask")
        return {}

    # Take the first (and should be only) region
    prop = props[0]

    # -------------------------------------------------------------------------
    # Extract geometric properties
    # -------------------------------------------------------------------------
    area = prop.area  # Number of pixels
    perimeter = prop.perimeter  # Perimeter in pixels

    # Major and minor axes (in pixels)
    major_axis_length = prop.major_axis_length
    minor_axis_length = prop.minor_axis_length

    # Orientation of major axis (in radians)
    orientation = prop.orientation  # Range: [-pi/2, pi/2]
    orientation_degrees = np.degrees(orientation)

    # Aspect ratio (length/width ratio)
    aspect_ratio = major_axis_length / minor_axis_length if minor_axis_length > 0 else 0

    # Centroid (center of mass)
    centroid_y, centroid_x = prop.centroid

    # Bounding box
    min_row, min_col, max_row, max_col = prop.bbox
    bbox_width = max_col - min_col
    bbox_height = max_row - min_row

    # -------------------------------------------------------------------------
    # Calculate average color of midrib (approximation)
    # -------------------------------------------------------------------------
    # To get the midrib, we would use the major axis
    # For now, we'll calculate the average color of the central region
    masked_image = cv2.bitwise_and(image, image, mask=leaf_mask)

    # Central region (midrib approximation)
    center_mask = np.zeros_like(leaf_mask)
    center_y = int(centroid_y)
    center_x = int(centroid_x)
    # Extract central vertical stripe (±5% of width)
    stripe_width = int(bbox_width * 0.05)
    x_start = max(0, center_x - stripe_width)
    x_end = min(leaf_mask.shape[1], center_x + stripe_width)
    center_mask[:, x_start:x_end] = leaf_mask[:, x_start:x_end]

    # Average color in central region
    center_pixels = masked_image[center_mask > 0]
    if len(center_pixels) > 0:
        mean_color_bgr = center_pixels.mean(axis=0)
        mean_color_rgb = mean_color_bgr[::-1]  # BGR to RGB
    else:
        mean_color_rgb = np.array([0, 0, 0])

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------
    print(f"\n📏 LEAF MEASUREMENTS:")
    print(f"  Area: {area:,} pixels")
    print(f"  Perimeter: {perimeter:.1f} pixels")
    print(f"  Major axis (length): {major_axis_length:.1f} pixels")
    print(f"  Minor axis (width): {minor_axis_length:.1f} pixels")
    print(f"  Aspect ratio (length/width): {aspect_ratio:.2f}")
    print(f"  Orientation: {orientation_degrees:.1f}°")
    print(f"  Centroid: ({centroid_x:.1f}, {centroid_y:.1f})")
    print(f"  Bounding box: {bbox_width} x {bbox_height} pixels")
    print(f"\n🎨 MIDRIB COLOR (approximation):")
    print(f"  RGB: ({mean_color_rgb[0]:.0f}, {mean_color_rgb[1]:.0f}, {mean_color_rgb[2]:.0f})")

    # -------------------------------------------------------------------------
    # Create measurements dictionary
    # -------------------------------------------------------------------------
    measurements = {
        'area_pixels': area,
        'perimeter_pixels': perimeter,
        'major_axis_length_pixels': major_axis_length,
        'minor_axis_length_pixels': minor_axis_length,
        'aspect_ratio': aspect_ratio,
        'orientation_degrees': orientation_degrees,
        'orientation_radians': orientation,
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'bbox_x': min_col,
        'bbox_y': min_row,
        'bbox_width': bbox_width,
        'bbox_height': bbox_height,
        'midrib_color_r': mean_color_rgb[0],
        'midrib_color_g': mean_color_rgb[1],
        'midrib_color_b': mean_color_rgb[2],
        'regionprops': prop  # Save complete object for additional analysis
    }

    return measurements


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_results(image, mask_no_background, leaf_mask, measurements,
                     stage1_debug, stage2_debug, output_path=None):
    """
    Create complete visualizations of the entire process.

    Shows:
    1. Original image
    2. STAGE 1: Mask without background
    3. STAGE 2: Final leaf mask
    4. Overlay with measurements
    """
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Segmentation Pipeline - James Method',
                 fontsize=16, fontweight='bold')

    # -------------------------------------------------------------------------
    # Row 1: Segmentation process
    # -------------------------------------------------------------------------

    # (1,1) Original image with seed points
    ax = axes[0, 0]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title('1. Original Image + Seed Points', fontweight='bold')
    # Mark seed points
    if 'seed_points' in stage1_debug:
        for i, (x, y) in enumerate(stage1_debug['seed_points']):
            color = 'red' if i == 0 else 'blue'
            label = 'Top' if i == 0 else 'Bottom'
            ax.plot(x, y, 'o', color=color, markersize=15,
                   markeredgewidth=2, markeredgecolor='white')
            ax.text(x, y-30, f'Seed {label}', color=color, fontsize=10,
                   ha='center', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.axis('off')

    # (1,2) Initial white mask (threshold)
    ax = axes[0, 1]
    ax.imshow(stage1_debug['white_mask'], cmap='gray')
    ax.set_title(f'2. STAGE 1a: White Threshold (>{stage1_debug["threshold_value"]:.0f})',
                fontweight='bold')
    ax.axis('off')

    # (1,3) Mask without background
    ax = axes[0, 2]
    ax.imshow(mask_no_background, cmap='gray')
    ax.set_title('3. STAGE 1b: Background Removed', fontweight='bold')
    ax.axis('off')

    # -------------------------------------------------------------------------
    # Row 2: Final result and measurements
    # -------------------------------------------------------------------------

    # (2,1) Final leaf mask
    ax = axes[1, 0]
    ax.imshow(leaf_mask, cmap='gray')
    ax.set_title('4. STAGE 2: Leaf Identified', fontweight='bold')
    ax.axis('off')

    # (2,2) Leaf overlay on original image
    ax = axes[1, 1]
    overlay = image.copy()
    overlay[leaf_mask > 0] = overlay[leaf_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    ax.imshow(cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax.set_title('5. Overlay: Leaf in Green', fontweight='bold')
    ax.axis('off')

    # (2,3) Visualized measurements
    ax = axes[1, 2]
    vis_image = image.copy()

    # Draw bounding box
    if measurements:
        x = int(measurements['bbox_x'])
        y = int(measurements['bbox_y'])
        w = int(measurements['bbox_width'])
        h = int(measurements['bbox_height'])
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), (255, 0, 0), 3)

        # Draw centroid
        cx = int(measurements['centroid_x'])
        cy = int(measurements['centroid_y'])
        cv2.circle(vis_image, (cx, cy), 10, (0, 0, 255), -1)

        # Draw major and minor axes
        prop = measurements['regionprops']

        # Major axis
        orientation = measurements['orientation_radians']
        major_length = measurements['major_axis_length_pixels'] / 2
        x1 = int(cx + major_length * np.cos(orientation))
        y1 = int(cy + major_length * np.sin(orientation))
        x2 = int(cx - major_length * np.cos(orientation))
        y2 = int(cy - major_length * np.sin(orientation))
        cv2.line(vis_image, (x1, y1), (x2, y2), (255, 255, 0), 3)

        # Minor axis
        minor_length = measurements['minor_axis_length_pixels'] / 2
        x1 = int(cx + minor_length * np.cos(orientation + np.pi/2))
        y1 = int(cy + minor_length * np.sin(orientation + np.pi/2))
        x2 = int(cx - minor_length * np.cos(orientation + np.pi/2))
        y2 = int(cy - minor_length * np.sin(orientation + np.pi/2))
        cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 3)

    ax.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    title_text = '6. Measurements\n'
    if measurements:
        title_text += f'Length: {measurements["major_axis_length_pixels"]:.0f}px, '
        title_text += f'Width: {measurements["minor_axis_length_pixels"]:.0f}px'
    ax.set_title(title_text, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved: {output_path}")

    plt.show()
    return fig


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def process_single_image(image_path, output_dir=None, visualize=True):
    """
    Process a single image with James's method.

    PARAMETERS:
    ----------
    image_path : str
        Path to input image
    output_dir : str, optional
        Directory to save results
    visualize : bool
        If True, shows visualizations

    RETURNS:
    -------
    results : dict
        Dictionary with masks, measurements and debugging info
    """
    print("\n" + "="*70)
    print("PROCESSING IMAGE")
    print("="*70)
    print(f"File: {image_path}")

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"⚠ ERROR: Could not read image: {image_path}")
        return None

    print(f"✓ Image loaded: {image.shape[1]} x {image.shape[0]} pixels")

    # STAGE 1: Remove white background
    mask_no_background, stage1_debug = remove_white_background(image, visualize)

    # STAGE 2: Identify the leaf
    leaf_mask, stage2_debug = identify_leaf(mask_no_background, image, visualize)

    # Extract measurements
    measurements = extract_measurements(leaf_mask, image)

    # Visualize
    fig = None
    if visualize:
        output_path = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            image_name = Path(image_path).stem
            output_path = output_dir / f"{image_name}_visualization.png"

        fig = visualize_results(
            image, mask_no_background, leaf_mask, measurements,
            stage1_debug, stage2_debug, output_path
        )

    # Save masks and data if output_dir is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_name = Path(image_path).stem

        # Save masks
        cv2.imwrite(str(output_dir / f"{image_name}_mask_no_background.png"),
                   mask_no_background)
        cv2.imwrite(str(output_dir / f"{image_name}_leaf_mask.png"),
                   leaf_mask)

        # Save measurements as CSV
        if measurements:
            df = pd.DataFrame([{k: v for k, v in measurements.items()
                              if k != 'regionprops'}])
            csv_path = output_dir / f"{image_name}_measurements.csv"
            df.to_csv(csv_path, index=False)
            print(f"✓ Measurements saved: {csv_path}")

    # Compile results
    results = {
        'image': image,
        'mask_no_background': mask_no_background,
        'leaf_mask': leaf_mask,
        'measurements': measurements,
        'stage1_debug': stage1_debug,
        'stage2_debug': stage2_debug,
        'figure': fig
    }

    return results


# =============================================================================
# MAIN SCRIPT
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LEAF SEGMENTATION PIPELINE - JAMES METHOD")
    print("="*70)
    print("Author: Karla")
    print("Date: 2025-11-20")
    print("="*70)

    # -------------------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------------------

    # Test image
    test_image_path = (
        "/media/preethi/Expansion/Extreme_SSD_phenotyping/data_fieldLeafImaging/"
        "data/FVSU_SAP_BAP_Images/AAMUImages/"
        "2025-10-27-09-40-04_2025 Sorghum_Update_073125-XK_Index_block_1/"
        "2025 Sorghum_Update_073125-XK_Index/LeafPhotoA/"
        "10107_LeafPhotoA_2025-10-15 15_50_53.408-05_00.jpg"
    )

    # Output directory
    output_dir = "/home/preethi/leaf_segmentation_project/05_output/james_method_test"

    # -------------------------------------------------------------------------
    # PROCESS
    # -------------------------------------------------------------------------

    results = process_single_image(
        image_path=test_image_path,
        output_dir=output_dir,
        visualize=True
    )

    if results:
        print("\n" + "="*70)
        print("✓ PROCESSING COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nResults saved in: {output_dir}")
        print("\nGenerated files:")
        print("  - *_mask_no_background.png: Mask without background (STAGE 1)")
        print("  - *_leaf_mask.png: Final leaf mask (STAGE 2)")
        print("  - *_measurements.csv: Leaf measurements")
        print("  - *_visualization.png: Complete process visualization")
    else:
        print("\n⚠ ERROR: Processing failed")
        sys.exit(1)
