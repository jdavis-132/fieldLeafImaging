import numpy as np
import cv2
from sklearn.decomposition import PCA
from typing import List, Tuple
import math

def sliding_window_along_axis(image: np.ndarray, 
                             mask_path: str, 
                             step: int, 
                             x_dim: int, 
                             y_dim: int) -> List[np.ndarray]:
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


def visualize_sliding_windows(image: np.ndarray, 
                              mask_path: str, 
                              step: int, 
                              x_dim: int, 
                              y_dim: int) -> np.ndarray:
    """
    Visualizes the sliding window positions on the original image.
    
    This is a helper function to visualize where the bounding boxes will be placed.
    """
    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")
    
    # Create a copy of the image for visualization
    vis_image = image.copy()
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    
    # Find non-zero pixels in the mask
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        return vis_image
    
    # Stack coordinates for PCA
    points = np.column_stack((x_coords, y_coords))
    
    # Perform PCA to find principal direction
    pca = PCA(n_components=2)
    pca.fit(points)
    
    # Get the principal and perpendicular axes
    principal_axis = pca.components_[0]
    perpendicular_axis = pca.components_[1]
    
    # Get the center point of the mask
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    
    # Project all points onto the principal axis
    projections = np.dot(points - [center_x, center_y], principal_axis)
    min_proj = np.min(projections)
    max_proj = np.max(projections)
    
    # Calculate the starting point along the principal axis
    start_point = np.array([center_x, center_y]) + min_proj * principal_axis
    
    # Calculate total distance along principal axis
    total_distance = max_proj - min_proj
    
    # Draw the principal axis
    axis_start = (start_point + min_proj * principal_axis).astype(int)
    axis_end = (start_point + max_proj * principal_axis).astype(int)
    cv2.line(vis_image, tuple(axis_start), tuple(axis_end), (0, 255, 0), 2)
    
    # Draw sliding window positions
    current_distance = 0
    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    color_idx = 0
    
    while current_distance + x_dim <= total_distance:
        # Calculate center of current window
        window_center = start_point + (current_distance + x_dim/2) * principal_axis
        
        # Calculate the four corners
        half_x = x_dim / 2
        half_y = y_dim / 2
        
        corners = []
        local_corners = [
            [-half_x, -half_y],
            [half_x, -half_y],
            [half_x, half_y],
            [-half_x, half_y]
        ]
        
        for lc in local_corners:
            corner = window_center + lc[0] * principal_axis + lc[1] * perpendicular_axis
            corners.append(corner.astype(int))
        
        # Draw the rectangle
        corners = np.array(corners)
        cv2.polylines(vis_image, [corners], True, colors[color_idx % len(colors)], 2)
        
        color_idx += 1
        current_distance += step
    
    return vis_image


# Example usage
if __name__ == "__main__":
    # Example of how to use the function
    import matplotlib.pyplot as plt
    
    # Create a sample image and mask for testing
    # This would normally be your actual image and mask
    sample_image = np.zeros((500, 500, 3), dtype=np.uint8)
    sample_image[:] = 255  # White background
    
    # Create a diagonal stripe mask
    mask = np.zeros((500, 500), dtype=np.uint8)
    for i in range(100, 400):
        for j in range(max(0, i-50), min(500, i+50)):
            mask[i, j] = 255
    
    # Save the mask temporarily
    cv2.imwrite('/tmp/test_mask.png', mask)
    
    # Apply the sliding window function
    crops = sliding_window_along_axis(
        image=sample_image,
        mask_path='/tmp/test_mask.png',
        step=30,
        x_dim=60,
        y_dim=40
    )
    
    print(f"Generated {len(crops)} crops")
    
    # Visualize the windows
    vis = visualize_sliding_windows(
        image=sample_image,
        mask_path='/tmp/test_mask.png',
        step=30,
        x_dim=60,
        y_dim=40
    )
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(vis)
    plt.title('Sliding Windows Visualization')
    plt.axis('off')
    
    if len(crops) > 0:
        plt.subplot(1, 3, 3)
        plt.imshow(crops[0])
        plt.title('First Crop')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
