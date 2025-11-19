#!/usr/bin/env python3
"""
Diagnostic script to analyze color properties of midrib vs lamina using actual green mask
"""
import cv2
import numpy as np

# Load a test image
image_path = "/home/schnable/Documents/fieldLeafImaging/data/ne2025/device3/2749_LeafPhotoA_2025-09-09 11_20_58.643-05_00.jpg"
image = cv2.imread(image_path)

# Convert to different color spaces
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Create green mask (same as in the detection algorithm)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
green_mask = cv2.inRange(hsv, lower_green, upper_green)

# Get green lamina pixels (from mask)
lamina_pixels_h = hsv[:,:,0][green_mask > 0]
lamina_pixels_s = hsv[:,:,1][green_mask > 0]
lamina_pixels_v = hsv[:,:,2][green_mask > 0]
lamina_pixels_l = lab[:,:,0][green_mask > 0]
lamina_pixels_a = lab[:,:,1][green_mask > 0]
lamina_pixels_b = lab[:,:,2][green_mask > 0]

print("=== GREEN LAMINA (from green mask) ===")
print(f"HSV: H={np.mean(lamina_pixels_h):.1f}±{np.std(lamina_pixels_h):.1f}, " +
      f"S={np.mean(lamina_pixels_s):.1f}±{np.std(lamina_pixels_s):.1f}, " +
      f"V={np.mean(lamina_pixels_v):.1f}±{np.std(lamina_pixels_v):.1f}")
print(f"LAB: L={np.mean(lamina_pixels_l):.1f}±{np.std(lamina_pixels_l):.1f}, " +
      f"A={np.mean(lamina_pixels_a):.1f}±{np.std(lamina_pixels_a):.1f}, " +
      f"B={np.mean(lamina_pixels_b):.1f}±{np.std(lamina_pixels_b):.1f}")

# Sample the ROI region between strips (from detection output: y=[2364, 2400])
roi_y1, roi_y2 = 2364, 2400
roi_x1, roi_x2 = 140, 436

roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
roi_hsv = hsv[roi_y1:roi_y2, roi_x1:roi_x2]
roi_lab = lab[roi_y1:roi_y2, roi_x1:roi_x2]
roi_green_mask = green_mask[roi_y1:roi_y2, roi_x1:roi_x2]

# Get ALL pixels in ROI (not just green ones)
print(f"\n=== ROI BETWEEN STRIPS (y={roi_y1}-{roi_y2}, x={roi_x1}-{roi_x2}) ===")
print("All pixels in ROI:")
print(f"HSV: H={np.mean(roi_hsv[:,:,0]):.1f}±{np.std(roi_hsv[:,:,0]):.1f}, " +
      f"S={np.mean(roi_hsv[:,:,1]):.1f}±{np.std(roi_hsv[:,:,1]):.1f}, " +
      f"V={np.mean(roi_hsv[:,:,2]):.1f}±{np.std(roi_hsv[:,:,2]):.1f}")
print(f"LAB: L={np.mean(roi_lab[:,:,0]):.1f}±{np.std(roi_lab[:,:,0]):.1f}, " +
      f"A={np.mean(roi_lab[:,:,1]):.1f}±{np.std(roi_lab[:,:,1]):.1f}, " +
      f"B={np.mean(roi_lab[:,:,2]):.1f}±{np.std(roi_lab[:,:,2]):.1f}")

# Get bright pixels in ROI
bright_mask = (roi_lab[:,:,0] > 70) & (roi_hsv[:,:,2] > 70)
if np.sum(bright_mask) > 0:
    print(f"\nBright pixels in ROI (L>70, V>70): {np.sum(bright_mask)} pixels")
    print(f"HSV: H={np.mean(roi_hsv[:,:,0][bright_mask]):.1f}, " +
          f"S={np.mean(roi_hsv[:,:,1][bright_mask]):.1f}, " +
          f"V={np.mean(roi_hsv[:,:,2][bright_mask]):.1f}")
    print(f"LAB: L={np.mean(roi_lab[:,:,0][bright_mask]):.1f}, " +
          f"A={np.mean(roi_lab[:,:,1][bright_mask]):.1f}, " +
          f"B={np.mean(roi_lab[:,:,2][bright_mask]):.1f}")

# Get very bright pixels in ROI
very_bright_mask = (roi_lab[:,:,0] > 100) & (roi_hsv[:,:,2] > 100)
if np.sum(very_bright_mask) > 0:
    print(f"\nVery bright pixels in ROI (L>100, V>100): {np.sum(very_bright_mask)} pixels")
    print(f"HSV: H={np.mean(roi_hsv[:,:,0][very_bright_mask]):.1f}, " +
          f"S={np.mean(roi_hsv[:,:,1][very_bright_mask]):.1f}, " +
          f"V={np.mean(roi_hsv[:,:,2][very_bright_mask]):.1f}")
    print(f"LAB: L={np.mean(roi_lab[:,:,0][very_bright_mask]):.1f}, " +
          f"A={np.mean(roi_lab[:,:,1][very_bright_mask]):.1f}, " +
          f"B={np.mean(roi_lab[:,:,2][very_bright_mask]):.1f}")

# Check hue distribution in bright pixels
print("\n=== HUE DISTRIBUTION IN BRIGHT ROI PIXELS ===")
if np.sum(bright_mask) > 0:
    hue_values = roi_hsv[:,:,0][bright_mask]
    print(f"Hue range: {np.min(hue_values):.1f} - {np.max(hue_values):.1f}")
    print(f"Hue percentiles: 25%={np.percentile(hue_values, 25):.1f}, " +
          f"50%={np.percentile(hue_values, 50):.1f}, 75%={np.percentile(hue_values, 75):.1f}")

# Show what passes each filter
print("\n=== FILTER ANALYSIS ===")
print(f"Brightness filter (L>73.6, V>70.3): {np.sum((roi_lab[:,:,0] > 73.6) & (roi_hsv[:,:,2] > 70.3))} pixels")
print(f"Color filter (B>143.8): {np.sum(roi_lab[:,:,2] > 143.8)} pixels")
print(f"Color filter (S<86.5): {np.sum(roi_hsv[:,:,1] < 86.5)} pixels")
print(f"Color filter (H 30-75): {np.sum((roi_hsv[:,:,0] >= 30) & (roi_hsv[:,:,0] <= 75))} pixels")
print(f"Combined color filter: {np.sum((roi_lab[:,:,2] > 143.8) & (roi_hsv[:,:,1] < 86.5) & (roi_hsv[:,:,1] > 10) & (roi_hsv[:,:,0] >= 30) & (roi_hsv[:,:,0] <= 75))} pixels")
