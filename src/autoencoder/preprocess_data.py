"""
Preprocess dataset for leaf autoencoder.
"""
import os
import argparse
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import sys
import csv
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import src.autoencoder.segment_leaf as segment_leaf
from src.sam3.segment_leaves import LeafSegmenter
from src.sam3.test_sam3 import combine_masks
from sklearn.decomposition import PCA


"""
Preprocess images for leaf autoencoder
1. Create directory for processed image dataset if non-existent with tags for specified parameters
2. Get mask of the leaf (CV)
3. Get major axis of leaf
4. Crop image and mask to specified dims, with midrib major axis centered and perpendicular
5. Convert and normalize pixel values [0, 1] according to colorspace value
6. If use_mask = True,.multiply 
"""
        
def remove_pink_from_mask(image, mask, perimeter_width=500,
                          color_ranges=None):
    """
    Remove light pink areas from the mask, focusing on the perimeter of the image.

    Args:
        image: Original RGB image (BGR format from cv2.imread)
        mask: Binary mask to clean
        perimeter_width: Width of perimeter region to check for pink pixels (default: 500)
        color_ranges: List of tuples of (lower_bound, upper_bound) in HSV format.
                     If None, uses default pink foam ranges based on NE2025 device5 data.
                     Each bound should be a numpy array or list of [H, S, V] values.

    Returns:
        Cleaned mask with pink areas removed
    """
    # Convert image to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define default pink color ranges in HSV based on actual foam color from
    # data/ne2025/device5/1078_LeafPhotoA_2025-09-08 13_10_59.348-05_00.jpg
    # Analysis showed: Hue 0-14 (median: 12), Saturation 21-166 (median: 51), Value 151-255 (median: 198)
    if color_ranges is None:
        color_ranges = [
            # Primary range: Light pinkish foam (reddish-pink)
            (np.array([0, 20, 150]), np.array([20, 170, 255])),
            # Secondary range: Magenta-ish pink (fallback for other foam colors)
            (np.array([140, 30, 100]), np.array([170, 255, 255]))
        ]

    # Create masks for each color range and combine them
    pink_pixels = None
    for lower_bound, upper_bound in color_ranges:
        # Ensure bounds are numpy arrays
        lower = np.array(lower_bound) if not isinstance(lower_bound, np.ndarray) else lower_bound
        upper = np.array(upper_bound) if not isinstance(upper_bound, np.ndarray) else upper_bound

        range_mask = cv2.inRange(hsv, lower, upper)

        if pink_pixels is None:
            pink_pixels = range_mask
        else:
            pink_pixels = cv2.bitwise_or(pink_pixels, range_mask)

    # Create a perimeter mask (focus on edges of image)
    h, w = mask.shape
    perimeter_mask = np.zeros((h, w), dtype=np.uint8)

    # Mark perimeter regions
    perimeter_mask[:perimeter_width, :] = 255  # Top
    perimeter_mask[-perimeter_width:, :] = 255  # Bottom
    perimeter_mask[:, :perimeter_width] = 255  # Left
    perimeter_mask[:, -perimeter_width:] = 255  # Right

    # Combine: find pink pixels in perimeter area
    pink_in_perimeter = cv2.bitwise_and(pink_pixels, perimeter_mask)

    # Remove pink pixels from the original mask
    cleaned_mask = mask.copy()
    cleaned_mask[pink_in_perimeter > 0] = 0

    return cleaned_mask


def remove_reference_cards_from_mask(image, mask, min_card_area=5000, max_card_area=500000,
                                     min_rectangularity=0.7, perimeter_region_width=800):
    """
    Remove color reference cards and other rectangular objects from the mask.

    Reference cards are typically:
    - Rectangular/square shaped
    - Located near image edges
    - Have distinct, non-green colors
    - Relatively small compared to the leaf

    Args:
        image: Original RGB image (BGR format from cv2.imread)
        mask: Binary mask to clean
        min_card_area: Minimum area (in pixels) for a contour to be considered a card
        max_card_area: Maximum area (in pixels) for a contour to be considered a card
        min_rectangularity: Minimum ratio of contour area to bounding box area (0-1)
        perimeter_region_width: Width of perimeter region where cards are likely located

    Returns:
        Cleaned mask with reference cards removed
    """
    cleaned_mask = mask.copy()

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return cleaned_mask

    # Find the largest contour (assumed to be the leaf)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_contour)

    h, w = mask.shape

    # Create perimeter region mask
    perimeter_mask = np.zeros((h, w), dtype=np.uint8)
    perimeter_mask[:perimeter_region_width, :] = 255  # Top
    perimeter_mask[-perimeter_region_width:, :] = 255  # Bottom
    perimeter_mask[:, :perimeter_region_width] = 255  # Left
    perimeter_mask[:, -perimeter_region_width:] = 255  # Right

    # Check each contour to see if it's likely a reference card
    for contour in contours:
        area = cv2.contourArea(contour)

        # Skip if it's the largest contour (leaf) or too small/large
        if contour is largest_contour or area < min_card_area or area > max_card_area:
            continue

        # Skip if it's too large relative to the leaf (likely part of leaf)
        if area > largest_area * 0.3:
            continue

        # Get bounding box and calculate rectangularity
        x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
        bbox_area = bbox_w * bbox_h
        rectangularity = area / bbox_area if bbox_area > 0 else 0

        # Check if contour is rectangular
        if rectangularity < min_rectangularity:
            continue

        # Check if contour is in perimeter region
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        in_perimeter = cv2.bitwise_and(contour_mask, perimeter_mask)
        perimeter_overlap = np.sum(in_perimeter > 0) / area if area > 0 else 0

        # If mostly in perimeter and rectangular, likely a reference card
        if perimeter_overlap > 0.3:
            # Check color - reference cards are typically not green
            # Extract pixels from this contour in the original image
            contour_pixels = image[contour_mask > 0]
            if len(contour_pixels) > 0:
                # Convert to HSV to check if it's green
                hsv_pixels = cv2.cvtColor(contour_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
                hsv_pixels = hsv_pixels.reshape(-1, 3)

                # Green hue range in HSV (approximately 35-85 degrees)
                green_hue_mask = (hsv_pixels[:, 0] >= 35) & (hsv_pixels[:, 0] <= 85)
                green_sat_mask = hsv_pixels[:, 1] >= 30  # Minimum saturation for green
                green_pixels = np.sum(green_hue_mask & green_sat_mask)
                green_ratio = green_pixels / len(hsv_pixels)

                # If less than 50% green, likely a reference card
                if green_ratio < 0.5:
                    # Remove this contour from the mask
                    cv2.drawContours(cleaned_mask, [contour], -1, 0, -1)

    return cleaned_mask

def align_and_crop(image_path, mask_path, crop_dir, mask_crop_dir, step, x_dim, y_dim,
                   perimeter_width=500, color_ranges=None):
    """
    Saves cropped images and cropped masks to crop_dir and mask_crop_dir using a sliding window along the principal axis of a mask.

    Args:
        image_path: Path to image
        mask_path: Path to the binary mask image
        crop_dir: directory to save image crops to
        mask_crop_dir: directory to save mask crops to
        step: Step size in pixels for sliding window movement
        x_dim: Width of the bounding box (along the major axis)
        y_dim: Height of the bounding box (perpendicular to major axis)
        perimeter_width: Width of perimeter region to check for foam pixels (default: 500)
        color_ranges: List of tuples of (lower_bound, upper_bound) in HSV format for foam removal.
                     If None, uses default pink foam ranges.
    """
    # Load the mask and image
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")

    # Remove pink foam areas from the mask
    mask = remove_pink_from_mask(image, mask, perimeter_width=perimeter_width,
                                 color_ranges=color_ranges)

    # Remove color reference cards and other rectangular objects from the mask
    mask = remove_reference_cards_from_mask(image, mask)

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Find non-zero pixels in the mask
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        raise ValueError("No non-zero pixels found in mask")

    # Stack coordinates for PCA (use ALL mask pixels, no filtering)
    points = np.column_stack((x_coords, y_coords))

    # Perform PCA to find principal direction
    pca = PCA(n_components=2)
    pca.fit(points)

    # Get the principal axis (first component)
    principal_axis = pca.components_[0]  # [dx, dy] unit vector

    # Get the perpendicular axis (second component)
    perpendicular_axis = pca.components_[1]

    # Get the center point of the mask (use all coordinates)
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)

    # Validate axis orientation: principal axis should have greater extent than perpendicular
    # This prevents 90-degree rotation errors at leaf tips
    temp_principal_proj = np.dot(points - [center_x, center_y], principal_axis)
    temp_perpendicular_proj = np.dot(points - [center_x, center_y], perpendicular_axis)
    principal_extent = np.max(temp_principal_proj) - np.min(temp_principal_proj)
    perpendicular_extent = np.max(temp_perpendicular_proj) - np.min(temp_perpendicular_proj)

    if perpendicular_extent > principal_extent:
        # Swap axes if perpendicular has more extent (likely wrong orientation)
        principal_axis, perpendicular_axis = perpendicular_axis, principal_axis

    # Project all points onto the principal axis to find extent
    projections = np.dot(points - [center_x, center_y], principal_axis)
    min_proj = np.min(projections)
    max_proj = np.max(projections)

    # Calculate the starting and ending points along the principal axis
    start_point = np.array([center_x, center_y]) + min_proj * principal_axis
    end_point = np.array([center_x, center_y]) + max_proj * principal_axis

    # Calculate total distance along principal axis
    total_distance = max_proj - min_proj

    current_distance = 0
    i = 0

    while current_distance + x_dim <= total_distance:
        # Calculate center of current window along principal axis
        window_center_on_axis = start_point + (current_distance + x_dim/2) * principal_axis

        # Find the perpendicular center for this window region
        # Project all mask points onto the principal axis
        all_points = np.column_stack((x_coords, y_coords))
        all_projections = np.dot(all_points - [center_x, center_y], principal_axis)

        # Find points within current window region along principal axis
        window_min_proj = min_proj + current_distance
        window_max_proj = min_proj + current_distance + x_dim
        in_window = (all_projections >= window_min_proj) & (all_projections <= window_max_proj)

        # Calculate perpendicular offset to center on leaf content in this region
        # FIX 3: Use local window center instead of global center reference
        if np.sum(in_window) > 0:
            window_points = all_points[in_window]
            # Calculate the geometric center of points in this window
            window_points_center = np.mean(window_points, axis=0)
            # Calculate perpendicular offset from window_center_on_axis to this center
            offset_vector = window_points_center - window_center_on_axis
            mean_perp_offset = np.dot(offset_vector, perpendicular_axis)
        else:
            # Fallback to no offset if no points in window
            mean_perp_offset = 0

        # Apply perpendicular offset to center the window on the leaf
        window_center = window_center_on_axis + mean_perp_offset * perpendicular_axis

        # Half dimensions
        half_x = x_dim / 2
        half_y = y_dim / 2

        # Four corners in the rotated coordinate system
        local_corners = [
            [-half_x, -half_y],
            [half_x, -half_y],
            [half_x, half_y],
            [-half_x, half_y]
        ]

        # Function to calculate corners given a window center
        def calculate_corners(center):
            corners = []
            for lc in local_corners:
                corner = center + lc[0] * principal_axis + lc[1] * perpendicular_axis
                corners.append(corner)
            return np.array(corners, dtype=np.float32)

        # Function to check if corners are within bounds
        def corners_within_bounds(corners):
            for corner in corners:
                if corner[0] < 0 or corner[0] >= img_width or corner[1] < 0 or corner[1] >= img_height:
                    return False
            return True

        # Calculate initial corners
        corners = calculate_corners(window_center)

        # If corners are out of bounds, search for valid perpendicular offset
        # FIX 2: Choose offset that maximizes leaf content, not just closest to zero
        best_offset = 0.0
        corners_in_bounds = corners_within_bounds(corners)

        if not corners_in_bounds:
            # Try adjusting along perpendicular axis to find valid position
            # Search in both positive and negative directions
            search_range = max(img_width, img_height)  # Maximum possible offset needed

            # Find all valid offsets and score them by leaf content coverage
            valid_offsets_with_scores = []
            for offset in np.linspace(-search_range, search_range, 200):
                test_center = window_center + offset * perpendicular_axis
                test_corners = calculate_corners(test_center)

                if corners_within_bounds(test_corners):
                    # Score this offset by how many mask pixels would be in the window
                    # Transform mask to this window position to count pixels
                    dst_points = np.array([
                        [0, 0],
                        [x_dim - 1, 0],
                        [x_dim - 1, y_dim - 1],
                        [0, y_dim - 1]
                    ], dtype=np.float32)
                    transform_matrix = cv2.getPerspectiveTransform(test_corners, dst_points)
                    test_mask = cv2.warpPerspective(mask, transform_matrix, (x_dim, y_dim))
                    mask_pixel_count = np.sum(test_mask > 0)

                    valid_offsets_with_scores.append((offset, mask_pixel_count))

            # Choose the offset with maximum leaf content
            if valid_offsets_with_scores:
                best_offset, best_score = max(valid_offsets_with_scores, key=lambda x: x[1])
                corners = calculate_corners(window_center + best_offset * perpendicular_axis)
                corners_in_bounds = True

        # Only save if we found a valid position
        if corners_in_bounds:
            window_center = window_center + best_offset * perpendicular_axis

            # Create destination points for the transformed rectangle
            dst_points = np.array([
                [0, 0],
                [x_dim - 1, 0],
                [x_dim - 1, y_dim - 1],
                [0, y_dim - 1]
                ], dtype=np.float32)

            # Get perspective transform matrix
            transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)

            # Apply the transformation to get the cropped and aligned image
            cropped = cv2.warpPerspective(image, transform_matrix, (x_dim, y_dim))
            mask_cropped = cv2.warpPerspective(mask, transform_matrix, (x_dim, y_dim))

            # Calculate percentage of mask pixels that are true
            mask_pixel_count = np.sum(mask_cropped > 0)
            total_pixels = x_dim * y_dim
            mask_percentage = mask_pixel_count / total_pixels

            # Only save if at least 20% of pixels are masked
            if mask_percentage >= 0.20:
                # Save image and mask to appropriate directories
                image_basename = os.path.basename(image_path).replace('.jpg', '_' + str(i) + '.png')
                cv2.imwrite(crop_dir + '/' + image_basename, cropped)
                cv2.imwrite(mask_crop_dir + '/' + image_basename, mask_cropped)

                i += 1

        current_distance += step
        
def normalize_pixel_values(image_path, out_dir, colorspace='RGB'):
    """
    Normalize pixel values to [0, 1] range and save as NPZ files.

    Args:
        image_path: Path to input image
        out_dir: Output directory for normalized images
        colorspace: Colorspace to normalize to ('RGB', 'HSV', 'LAB')
    """
    image = cv2.imread(image_path)

    if colorspace=='RGB':
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        normalized_image = image_rgb.astype(np.float32) / 255.0

    if colorspace=='HSV':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Split the HSV image into H, S, and V channels
        h, s, v = cv2.split(hsv)

        # Normalize the H channel to the range [0, 1]
        # Note: OpenCV uses H in range [0, 179]
        h_normalized = h.astype(np.float32) / 179.0
        # Normalize S and V channels to the range [0, 1]
        s_normalized = s.astype(np.float32) / 255.0
        v_normalized = v.astype(np.float32) / 255.0
        normalized_image = cv2.merge([h_normalized, s_normalized, v_normalized])

    if colorspace=='LAB':
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # LAB values: L in [0, 100], a and b in [-128, 127]
        # Normalize to [0, 1] by dividing by max values
        l_channel, a_channel, b_channel = cv2.split(lab.astype(np.float32))
        l_normalized = l_channel / 255.0  # OpenCV stores L as [0, 255]
        a_normalized = a_channel / 255.0
        b_normalized = b_channel / 255.0
        normalized_image = cv2.merge([l_normalized, a_normalized, b_normalized])

    # Save as NPZ file to preserve float32 [0, 1] values
    basename = os.path.basename(image_path)
    basename_npz = os.path.splitext(basename)[0] + '.npz'
    np.savez_compressed(out_dir + '/' + basename_npz, image=normalized_image)
        
def main():
    """
    Pipeline steps:
    1. Segment: Create leaf masks using SAM3/CV2
    2. Crop: Align and crop images using masks
    3. Normalize: Convert and normalize pixel values [0, 1] according to colorspace
    4. Apply masks: Multiply normalized images by masks (if use_mask = True)
    """

    parser = argparse.ArgumentParser(
        description="Preprocess images for leaf autoencoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  1. segment      - Segment leaves and create binary masks
  2. crop         - Align and crop images using masks
  3. normalize    - Normalize pixel values to [0, 1] range
  4. apply_masks  - Apply masks to normalized images

Examples:
  # Run entire pipeline:
  python preprocess_data.py -i data/raw -o data/processed

  # Start from cropping step (if segmentation already done):
  python preprocess_data.py -i data/raw -o data/processed --start_step crop

  # Start from normalization step (if segmentation/cropping already done):
  python preprocess_data.py -i data/raw -o data/processed --start_step normalize

  # Only apply masks (if segmentation and normalization already done):
  python preprocess_data.py -i data/raw -o data/processed --start_step apply_masks
        """
    )
    parser.add_argument('--input_dir', "-i", required=True, help="Input directory")
    parser.add_argument('--output_dir', '-o', required=True, help='Output directory')
    parser.add_argument('--use_mask', '-m', default=True, help='Use masked images in training (default: True)')
    parser.add_argument('--colorspace', '-c', default='RGB', help="Colorspace ('RGB', 'HSV', 'LAB') to use")
    parser.add_argument('--normalize_to_reference', default=False, help='Currently not supported')
    parser.add_argument('--pattern', default='.jpg', help='File pattern to search for (default: .jpg)')
    parser.add_argument(
        '--start_step',
        type=str,
        default='segment',
        choices=['segment', 'crop', 'normalize', 'apply_masks'],
        help='Pipeline step to start from (default: segment for full pipeline)'
    )

    args = parser.parse_args()

    # Setup directories
    input_path = args.input_dir
    output_path = args.output_dir
    mask_dir = output_path + '/masks'
    crop_dir = output_path + '/cropped'
    cropped_mask_dir = mask_dir + '_cropped'
    cropped_normalized_dir = crop_dir + '_' + args.colorspace + '_normalized'
    cropped_normalized_masked_dir = cropped_normalized_dir + '_masked'

    # Create directories as needed
    os.makedirs(output_path, exist_ok=True)

    # Determine step numbers for control flow
    step_order = ['segment', 'crop', 'normalize', 'apply_masks']
    start_step_idx = step_order.index(args.start_step)

    print(f"\n{'='*80}")
    print(f"Starting pipeline from step: {args.start_step}")
    print(f"{'='*80}\n")

    # Track images with no detections (only relevant for segmentation)
    no_detection_images = []

    # Track segmentation methods used for each image
    segmentation_methods = []

    # ========================================================================
    # STEP 1: SEGMENT
    # ========================================================================
    if start_step_idx <= 0:
        print("\n" + "="*80)
        print("STEP 1: SEGMENT LEAVES")
        print("="*80)

        os.makedirs(mask_dir, exist_ok=True)

        # Initialize SAM3
        segmenter = LeafSegmenter(model_path='src/sam3')
        print('Initialized SAM3')

        # Get list of images to process
        raw_images = glob.glob(input_path + '/*' + args.pattern)
        print(f'Found {len(raw_images)} images to segment')

        for idx, image in enumerate(raw_images):
            print(f"\nProcessing image {idx+1}/{len(raw_images)}: {os.path.basename(image)}")

            print(f"  Segmenting with CV2...")
            mask = segment_leaf.process_single(image)
            segmentation_method = 'CV2'

            if mask is None or np.sum(mask[int(mask.shape[0]/2):mask.shape[0], 0:(mask.shape[1] - 1)]) == 0:
                sam3_results = segmenter.segment_image(image)
                masks = sam3_results['masks']
                print(f"  Found {len(masks)} SAM3 masks, combining...")
                mask = combine_masks(masks)
                segmentation_method = 'SAM3'

                if mask is None or mask.sum() == 0:
                    print(f"  Segmentation failed, skipping...")
                    no_detection_images.append(image)
                    segmentation_methods.append({'image_path': image, 'segmentation_method': 'Failed'})
                    continue

            basename = os.path.splitext(os.path.basename(image))[0]
            mask_path = mask_dir + '/' + basename + '.png'

            # Save mask (multiply by 255 to get proper white pixels)
            cv2.imwrite(mask_path, mask * 255)

            # Check if mask was saved successfully
            if not os.path.exists(mask_path):
                print(f"  Failed to save mask, skipping...")
                continue

            print(f"  Mask saved")

            # Record which segmentation method was used
            segmentation_methods.append({'image_path': image, 'segmentation_method': segmentation_method})

        # Get list of created masks
        masks_created = glob.glob(mask_dir + '/*.png')
        print(f'\nSegmentation complete. Created {len(masks_created)} masks.')

        # Save list of images with no detections to CSV
        if no_detection_images:
            no_detection_csv = output_path + '/no_detections.csv'
            with open(no_detection_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_path'])
                for img in no_detection_images:
                    writer.writerow([img])
            print(f'Saved {len(no_detection_images)} images with no detections to: {no_detection_csv}')

        # Save segmentation methods to CSV
        if segmentation_methods:
            segmentation_methods_csv = output_path + '/segmentation_methods.csv'
            with open(segmentation_methods_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_path', 'segmentation_method'])
                for record in segmentation_methods:
                    writer.writerow([record['image_path'], record['segmentation_method']])
            print(f'Saved segmentation methods for {len(segmentation_methods)} images to: {segmentation_methods_csv}')

        print(f"\nSTEP 1 COMPLETE: Segmentation finished")
    else:
        print(f"STEP 1: SKIPPED (starting from {args.start_step})")

    # ========================================================================
    # STEP 2: CROP
    # ========================================================================
    if start_step_idx <= 1:
        print("\n" + "="*80)
        print("STEP 2: ALIGN AND CROP IMAGES")
        print("="*80)

        os.makedirs(crop_dir, exist_ok=True)
        os.makedirs(cropped_mask_dir, exist_ok=True)

        # Get list of raw images and their corresponding masks
        raw_images = glob.glob(input_path + '/*' + args.pattern)
        print(f'Found {len(raw_images)} raw images')

        # Check for corresponding masks
        masks_available = glob.glob(mask_dir + '/*.png')
        print(f'Found {len(masks_available)} masks')

        if len(masks_available) == 0:
            print(f"ERROR: No masks found in {mask_dir}")
            print("Please ensure Step 1 (segment) has been completed first.")
            return

        cropping_errors = []
        cropped_count = 0

        for idx, image in enumerate(raw_images):
            basename = os.path.splitext(os.path.basename(image))[0]
            mask_path = mask_dir + '/' + basename + '.png'

            # Check if mask exists for this image
            if not os.path.exists(mask_path):
                if (idx + 1) % 100 == 0 or idx < 5:
                    print(f"  [{idx+1}/{len(raw_images)}] Skipping {os.path.basename(image)} - no mask found")
                continue

            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"  Cropping image {idx+1}/{len(raw_images)}: {os.path.basename(image)}")

            # Align and crop
            try:
                align_and_crop(image, mask_path, crop_dir, cropped_mask_dir, 500, 1000, 2000)
                cropped_count += 1
            except Exception as e:
                cropping_errors.append((image, str(e)))
                if len(cropping_errors) <= 5:
                    print(f"  Error cropping {os.path.basename(image)}: {e}")

        # Get list of cropped images
        cropped_images = glob.glob(crop_dir + '/*')
        print(f'\nCropping complete. Created {len(cropped_images)} cropped images.')

        if cropping_errors:
            print(f'Encountered {len(cropping_errors)} cropping errors')
            if len(cropping_errors) <= 10:
                for img, err in cropping_errors:
                    print(f"  - {os.path.basename(img)}: {err}")

        print(f"\nSTEP 2 COMPLETE: Aligned and cropped {cropped_count} images")
    else:
        print(f"STEP 2: SKIPPED (starting from {args.start_step})")

    # ========================================================================
    # STEP 3: NORMALIZE
    # ========================================================================
    if start_step_idx <= 2:
        print("\n" + "="*80)
        print("STEP 3: NORMALIZE PIXEL VALUES")
        print("="*80)

        os.makedirs(cropped_normalized_dir, exist_ok=True)

        # Get list of cropped images
        cropped_images = glob.glob(crop_dir + '/*')
        print(f'Found {len(cropped_images)} cropped images to normalize')

        if len(cropped_images) == 0:
            print(f"ERROR: No cropped images found in {crop_dir}")
            print("Please ensure Step 2 (crop) has been completed first.")
            return

        for idx, image in enumerate(cropped_images):
            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"  Normalizing image {idx+1}/{len(cropped_images)}")
            normalize_pixel_values(image, cropped_normalized_dir, args.colorspace)

        print(f'Normalized pixel values [0, 1] based on theoretical maximums in the {args.colorspace} colorspace')
        print(f"\nSTEP 3 COMPLETE: Normalized {len(cropped_images)} images")
    else:
        print(f"STEP 3: SKIPPED (starting from {args.start_step})")

    # ========================================================================
    # STEP 4: APPLY MASKS
    # ========================================================================
    if start_step_idx <= 3 and args.use_mask:
        print("\n" + "="*80)
        print("STEP 4: APPLY MASKS TO NORMALIZED IMAGES")
        print("="*80)

        os.makedirs(cropped_normalized_masked_dir, exist_ok=True)

        # Get list of normalized crops (now .npz files)
        normalized_crops = glob.glob(cropped_normalized_dir + '/*.npz')
        print(f'Found {len(normalized_crops)} normalized images to mask')

        if len(normalized_crops) == 0:
            print(f"ERROR: No normalized images found in {cropped_normalized_dir}")
            print("Please ensure Step 3 (normalize) has been completed first.")
            return

        masked_count = 0
        for idx, image_path in enumerate(normalized_crops):
            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"  Masking image {idx+1}/{len(normalized_crops)}")

            basename = os.path.basename(image_path)
            # Load normalized image from NPZ file
            image = np.load(image_path)['image']

            # Load mask (PNG) and normalize to [0, 1]
            mask_basename = os.path.splitext(basename)[0] + '.png'
            mask = cv2.imread(cropped_mask_dir + '/' + mask_basename)
            if mask is None:
                print(f"  Warning: Could not load mask for {basename}, skipping...")
                continue

            # Convert mask to RGB and normalize to [0, 1]
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Apply mask
            masked_image = image * mask_rgb

            # Save as NPZ to preserve float32 [0, 1] values
            np.savez_compressed(cropped_normalized_masked_dir + '/' + basename, image=masked_image)
            masked_count += 1

        print(f'\nSTEP 4 COMPLETE: Masked {masked_count}/{len(normalized_crops)} images')
    elif start_step_idx <= 3 and not args.use_mask:
        print(f"STEP 4: SKIPPED (use_mask=False)")
    else:
        print(f"STEP 4: SKIPPED (starting from {args.start_step})")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Output directory: {output_path}")

    if start_step_idx <= 0 and no_detection_images:
        raw_images = glob.glob(input_path + '/*' + args.pattern)
        print(f"Successfully processed: {len(raw_images) - len(no_detection_images)}/{len(raw_images)} images")

    print("\nGenerated outputs:")
    if start_step_idx <= 0:
        print(f"  - Masks: {mask_dir}")
    if start_step_idx <= 1:
        print(f"  - Cropped images: {crop_dir}")
        print(f"  - Cropped masks: {cropped_mask_dir}")
    if start_step_idx <= 2:
        print(f"  - Normalized images: {cropped_normalized_dir}")
    if start_step_idx <= 3 and args.use_mask:
        print(f"  - Masked normalized images: {cropped_normalized_masked_dir}")
    print()


if __name__ == "__main__":
    main()
