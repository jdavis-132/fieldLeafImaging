"""
Preprocess dataset for leaf autoencoder.
"""
import os
os.chdir('/home/schnable/Documents/fieldLeafImaging/')
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
# from src.sam3.model_builder import build_sam3_image_model
# from src.sam3.model.sam3_image_processor import Sam3Processor
from sklearn.decomposition import PCA


"""
Preprocess images for leaf autoencoder
1. Create directory for processed image dataset if non-existent with tags for specified parameters
2. Identify leaf area using green color thresholds to generate a bounding box -- skip? if sam3 prompting with leaf works well
3. Get SAM mask of the leaf from bounding box
5. Get major axis of leaf
6. Crop image and mask to specified dims, with midrib major axis centered and perpendicular
8. Convert and normalize pixel values [0, 1] according to colorspace value
9. Perform data augmentation on training set if requested
10. Save datset to directory.
"""
def get_leaf_bbox(image):
    """
    Detects position of bounding box around leaf
    
    Args:
        image: BGR image
        
    Returns:
        list with ymin, ymax bounding box coordinates
    """

    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define range for green color
    # Adjust these values based on your specific green
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
        
    # distance to bottom of color reference card, measured from top of image
    bottom_color_ref_pixel = 1290 

    # Create mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
        
    # get row sums of mask
    mask_rowsums = np.sum(mask, axis = 1)
    nonzero_rowsum = np.where(mask_rowsums > 0)[0]
    nonzero_idx_clip = np.where(nonzero_rowsum > 1290)[0][0]
    # get min, max indexes of row mask_rowsums > 0 and idx > bottom_color_ref_pixel
    ymin = nonzero_rowsum[nonzero_idx_clip]
    ymax = nonzero_rowsum[nonzero_rowsum.shape[0] - 1]
        
    return [ymin, ymax]
    
def get_leaf_mask(image_path, y_coords, out, model):
    """
    Prompts SAM with a bounding box around the leaf and returns path to saved mask
        
    Args: 
        image_path: path to image to segment
        y_coords: list [ymin, ymax] coordinates for a bounding box assumed to span full x distance of image
        out: path to save mask to
    """
    x1 = 0
    x2 = cv2.imread(image_path).shape[1] - 1
    y1 = y_coords[0]
    y2 = y_coords[1]
    mask = model.predict_with_box(image_path, [x1, y1, x2, y2])['masks'][0]
    cv2.imwrite(out, (mask * 255).astype(np.uint8))
        
def align_and_crop(image, mask_path, step, x_dim, y_dim):
    """
    Creates a list of cropped images using a sliding window along the principal axis of a mask.
    
    Args:
        image: Input image as numpy array (H, W, C) or (H, W)
        mask_path: Path to the binary mask image
        step: Step size in pixels for sliding window movement
        x_dim: Width of the bounding box (along the major axis)
        y_dim: Height of the bounding box (perpendicular to major axis)

    Returns:
        List of cropped images aligned with the principal axis
    """
    # Load the mask
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
    
    # Generate sliding window positions
    cropped_images = []
    current_distance = 0
    
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
        
        cropped_images.append(cropped)
                    # Move to next position
        current_distance += step
    
    return cropped_images