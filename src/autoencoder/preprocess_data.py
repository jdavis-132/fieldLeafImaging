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
2. Get SAM mask of the leaf
3. Get major axis of leaf
4. Crop image and mask to specified dims, with midrib major axis centered and perpendicular
5. Convert and normalize pixel values [0, 1] according to colorspace value
6. If use_mask = True,.multiply 
"""
        
def align_and_crop(image_path, mask_path, crop_dir, mask_crop_dir, step, x_dim, y_dim):
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
    """
    # Load the mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")
    
    # Find non-zero pixels in the mask
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        raise ValueError("No non-zero pixels found in mask")
    
    # Stack coordinates for PCA
    points = np.column_stack((x_coords, y_coords))
    
    # Perform PCA to find principal direction
    pca = PCA(n_components=2)
    pca.fit(points)
    
    # Get the principal axis (first component)
    principal_axis = pca.components_[0]  # [dx, dy] unit vector
    
    # Get the perpendicular axis (second component)
    perpendicular_axis = pca.components_[1]
    
    # Get the center point of the mask
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    
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
        window_center = start_point + (current_distance + x_dim/2) * principal_axis
        
        # Calculate the four corners of the bounding box
        # The box is aligned with principal axis as x-direction and perpendicular as y-direction
        corners = []
        
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
        
        # Transform to image coordinates
        for lc in local_corners:
            corner = window_center + lc[0] * principal_axis + lc[1] * perpendicular_axis
            corners.append(corner)
        
        corners = np.array(corners, dtype=np.float32)
        
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
        
        # Save image and mask to appropriate directories
        image_basename = os.path.basename(image_path).replace('.jpg', '_' + str(i) + '.png')
        cv2.imwrite(crop_dir + '/' + image_basename, cropped)
        cv2.imwrite(mask_crop_dir + '/' + image_basename, mask_cropped)

        current_distance += step
        i += 1
        
def normalize_pixel_values(image_path, out_dir, colorspace='RGB'):
    image = cv2.imread(image_path)
        
    if colorspace=='RGB':
        normalized_image = image.astype(np.float32) / 255.0
            
    if colorspace=='HSV':
        hsv = cv2.cvtColor(image, 'cv2.COLOR_BGR2HSV')
        # Split the HSV image into H, S, and V channels
        h, s, v = cv2.split(hsv)
            
        # Normalize the H channel to the range [0, 1]
        # Note: OpenCV uses H in range [0, 179]
        h_normalized = h.astype(np.float32) / 179.0
        # Normalize S and V channels to the range [0, 1]
        s_normalized = s.astype(np.float32) / 255.0
        v_normalized = v.astype(np.float32) / 255.0
        normalized_image =  cv2.merge([h, s, v])
            
    if colorspace=='LAB':
        lab = cv2.cvtColor(image, 'cv2.COLOR_BGR2LAB')
        normalized_image = lab.astype(np.float32) / 255.0
            
    basename = os.path.basename(image_path)
    cv2.imwrite(out_dir + '/' + basename, normalized_image)
        
def main():
    """
    
    1. Create directory for processed image dataset if non-existent with tags for specified parameters
    2. Get SAM mask of the leaf
    3. Get major axis of leaf
    4. Crop image and mask to specified dims, with midrib major axis centered and perpendicular
    5. Convert and normalize pixel values [0, 1] according to colorspace value
    6. If use_mask = True, multiply by mask and save image
    """
    
    parser = argparse.ArgumentParser(description="Preprocess images for leaf autoencoder")
    parser.add_argument('--input_dir', "-i", required=True, help="Input directory")
    parser.add_argument('--output_dir', '-o', required=True, help='Output directory')
    parser.add_argument('--use_mask', '-m', default=True, help='Use masked images in training (default: False)')
    parser.add_argument('--colorspace', '-c', default='RGB', help="Colorspace ('RGB', 'HSV', 'LAB') to use")
    parser.add_argument('--normalize_to_reference', default=False, help='Currently not supported')
    parser.add_argument('--pattern', default = '.jpg', help='File pattern to search for (default: jpg)')
    
    args = parser.parse_args()
    # Create directories to save to
    input_path = args.input_dir
    output_path = args.output_dir
    mask_dir = output_path + '/masks'
    crop_dir = output_path + '/cropped'
    cropped_mask_dir = mask_dir + '_cropped'
    cropped_normalized_dir = crop_dir + '_' + args.colorspace + '_normalized'
    cropped_normalized_masked_dir = cropped_normalized_dir + '_masked'
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(cropped_mask_dir, exist_ok=True)
    os.makedirs(cropped_normalized_dir, exist_ok=True)

    # Use CV2 segmentation (SAM3 disabled)
    # print('Using CV2 segmentation')
    # initialize SAM3
    segmenter = LeafSegmenter(model_path='src/sam3')
    print('Initialized SAM3')

    # get list of images to process
    raw_images = glob.glob(input_path + '/*' + args.pattern)
    print('Found ' + str(len(raw_images)) + ' images')

    # Track images with no detections
    no_detection_images = []

    for idx, image in enumerate(raw_images):
        print(f"\nProcessing image {idx+1}/{len(raw_images)}: {os.path.basename(image)}")
        sam3_results = segmenter.segment_image(image)
        masks = sam3_results['masks']

        if len(masks) == 0:
            # Use CV2 segmentation
            print(f"  Segmenting with CV2...")
            mask = segment_leaf.process_single(image)
        else:
            # Combine SAM3 masks
            print(f"  Found {len(masks)} SAM3 masks, combining...")
            mask = combine_masks(masks)

        if mask is None or mask.sum() == 0:
            print(f"  Segmentation failed, skipping...")
            no_detection_images.append(image)
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

        # align and crop
        try:
            align_and_crop(image, mask_path, crop_dir, cropped_mask_dir, 500, 1000, 2000)
            print(f"  Cropping complete")
        except Exception as e:
            print(f"  Error during cropping: {e}")
            continue
        
    # get list of cropped images
    cropped_images = glob.glob(crop_dir + '/*')
    print('Raw images segmented, aligned, and cropped. Found ' + str(len(cropped_images)) + ' image crops.')
    
    for image in cropped_images:
        # normalize pixel values
        normalize_pixel_values(image, cropped_normalized_dir, args.colorspace)
    
    print('Normalized pixel values [0, 1] based on theoretical maximums in the ' + args.colorspace + ' colorspace')
    if args.use_mask:
        # get list of normalized crops
        normalized_crops = glob.glob(cropped_normalized_dir + '/*')
        os.makedirs(cropped_normalized_masked_dir, exist_ok=True)
        
        for image_path in normalized_crops:
            # multiply by mask and save
            basename = os.path.basename(image_path)
            image = cv2.imread(image_path)
            mask = cv2.imread(cropped_mask_dir + '/' + basename) / 255.0
            masked_image = image * mask
            cv2.imwrite(cropped_normalized_masked_dir + '/' + basename, masked_image)
            
        print('Masked ' + str(len(normalized_crops)) + ' images')

    # Save list of images with no detections to CSV
    if no_detection_images:
        no_detection_csv = output_path + '/no_detections.csv'
        with open(no_detection_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path'])
            for img in no_detection_images:
                writer.writerow([img])
        print(f'\nSaved {len(no_detection_images)} images with no detections to: {no_detection_csv}')
    else:
        print('\nAll images had detections!')

    print(f'\nImage preprocessing complete for {args.input_dir}')
    print(f'Successfully processed: {len(raw_images) - len(no_detection_images)}/{len(raw_images)} images')


if __name__ == "__main__":
    main()
