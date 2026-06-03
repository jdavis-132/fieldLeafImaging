#!/usr/bin/env python3
"""
Demo script for using SAM2Tiny in jupyterlab-debugger environment
"""

import sys
sys.path.insert(0, 'src/sam2')
from sam2_tiny import SAM2TinyPredictor
import cv2
import os
import glob

def main():
    # Initialize predictor
    print('Initializing SAM2Tiny...')
    predictor = SAM2TinyPredictor(
        model_path='src/sam2/models/sam2.1_hiera_tiny.pt',
        device='cpu'  # Change to 'cuda' if you have GPU
    )

    # Find all test images
    test_images = glob.glob('data/test_input/*.jpg')

    if not test_images:
        print('No test images found in data/test_input/')
        return

    print(f'\nFound {len(test_images)} test images')

    # Process first image as demo
    test_image = test_images[0]
    print(f'\nProcessing: {test_image}')

    img = cv2.imread(test_image)
    h, w = img.shape[:2]
    print(f'Image size: {w}x{h}')

    # Example 1: Box prompt (center region)
    print('\n--- Example 1: Box prompt ---')
    box = [w//4, h//4, 3*w//4, 3*h//4]
    print(f'Box: {box}')

    result = predictor.predict_with_box(test_image, box)
    print(f'✅ Segmentation successful!')
    print(f'   Masks: {len(result["masks"])}')
    print(f'   Mask shape: {result["masks"][0].shape}')
    print(f'   Score: {result["scores"][0]:.4f}')

    # Save visualization
    output_path = 'test_sam2_box_output.png'
    predictor.visualize_prediction(result, output_path)
    print(f'   Saved to: {output_path}')

    # Example 2: Point prompt (center point)
    print('\n--- Example 2: Point prompt ---')
    center_point = [[w//2, h//2]]
    point_labels = [1]  # 1 = foreground, 0 = background
    print(f'Point: {center_point}, Label: {point_labels}')

    result = predictor.predict_with_points(test_image, center_point, point_labels)
    print(f'✅ Segmentation successful!')
    print(f'   Masks: {len(result["masks"])}')
    print(f'   Best mask score: {result["scores"][0]:.4f}')

    # Save visualization
    output_path = 'test_sam2_point_output.png'
    predictor.visualize_prediction(result, output_path)
    print(f'   Saved to: {output_path}')

    print('\n✅ Demo complete!')

if __name__ == '__main__':
    main()
