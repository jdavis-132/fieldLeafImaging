#!/usr/bin/env python3
"""
Convert images to HSV and save visualizations of H, S, and V channels.
"""
import cv2
import numpy as np
import glob
import os
from pathlib import Path

def visualize_hsv_channels(image_path, output_dir):
    """
    Convert image to HSV and save separate visualizations for H, S, V channels.

    Args:
        image_path: Path to input image
        output_dir: Directory to save output images
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read {image_path}")
        return

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Get base filename
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Visualize H channel (Hue)
    # Create an HSV image with full saturation and value to show pure hues
    h_vis = np.zeros_like(hsv)
    h_vis[:, :, 0] = h  # Hue values
    h_vis[:, :, 1] = 255  # Full saturation
    h_vis[:, :, 2] = 255  # Full value
    h_rgb = cv2.cvtColor(h_vis, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{basename}_H.jpg"), h_rgb)

    # Visualize S channel (Saturation) as grayscale
    cv2.imwrite(os.path.join(output_dir, f"{basename}_S.jpg"), s)

    # Visualize V channel (Value/Brightness) as grayscale
    cv2.imwrite(os.path.join(output_dir, f"{basename}_V.jpg"), v)

    print(f"Processed: {basename}")

def main():
    input_dir = "data/test_2"
    output_dir = "data/test_2/hsv_vals"

    # Get all jpg images
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))

    # Filter out images already in the hsv_vals subdirectory
    image_paths = [p for p in image_paths if "hsv_vals" not in p]

    print(f"Found {len(image_paths)} images to process")
    print(f"Output directory: {output_dir}\n")

    # Process each image
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] ", end="")
        visualize_hsv_channels(image_path, output_dir)

    print(f"\nDone! Processed {len(image_paths)} images.")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    main()
