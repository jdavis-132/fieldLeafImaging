#!/usr/bin/env python3
"""
ColorChecker Classic Nano Color Normalization Script

This script normalizes colors across multiple images using a Calibrite ColorChecker
Classic Nano card visible in each frame. It detects the card, extracts patch colors,
computes a color correction transform, and applies it to the entire image.

Usage:
    python colorchecker_normalize.py --input data/ne2025 --output data/ne2025_normalized
    python colorchecker_normalize.py --input data/ne2025/device1 --output output/
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List
import json


# ColorChecker Classic reference values (sRGB D65, 0-255 range)
# Order: Row-wise from top-left, 4 rows x 6 columns = 24 patches
# Source: Based on post-2014 ColorChecker Classic specifications
COLORCHECKER_REFERENCE_RGB = np.array([
    # Row 1
    [115, 82, 68],    # Dark Skin
    [194, 150, 130],  # Light Skin
    [98, 122, 157],   # Blue Sky
    [87, 108, 67],    # Foliage
    [133, 128, 177],  # Blue Flower
    [103, 189, 170],  # Bluish Green

    # Row 2
    [214, 126, 44],   # Orange
    [80, 91, 166],    # Purplish Blue
    [193, 90, 99],    # Moderate Red
    [94, 60, 108],    # Purple
    [157, 188, 64],   # Yellow Green
    [224, 163, 46],   # Orange Yellow

    # Row 3
    [56, 61, 150],    # Blue
    [70, 148, 73],    # Green
    [175, 54, 60],    # Red
    [231, 199, 31],   # Yellow
    [187, 86, 149],   # Magenta
    [8, 133, 161],    # Cyan

    # Row 4 (Grayscale)
    [243, 243, 242],  # White
    [200, 200, 200],  # Neutral 8
    [160, 160, 160],  # Neutral 6.5
    [122, 122, 121],  # Neutral 5
    [85, 85, 85],     # Neutral 3.5
    [52, 52, 52],     # Black
], dtype=np.float32)


class ColorCheckerDetector:
    """Detects and extracts ColorChecker Classic Nano card from images."""

    def __init__(self, search_region='top-left', min_area_ratio=0.00005, max_area_ratio=0.2,
                 use_fixed_position=True, card_width=768, card_height=493,
                 offset_from_right=516, offset_from_top=291):
        """
        Initialize detector.

        Args:
            search_region: Region to search ('top-left', 'top-right', 'full')
            min_area_ratio: Minimum card area as ratio of image area
            max_area_ratio: Maximum card area as ratio of image area
            use_fixed_position: Use fixed position instead of automatic detection
            card_width: Width of ColorChecker card in pixels (after rotation: 768)
            card_height: Height of ColorChecker card in pixels (after rotation: 493)
            offset_from_right: Distance from right edge to right edge of card (516)
            offset_from_top: Distance from bottom edge to bottom edge of card (291)
        """
        self.search_region = search_region
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.use_fixed_position = use_fixed_position
        self.card_width = card_width
        self.card_height = card_height
        self.offset_from_right = offset_from_right
        self.offset_from_top = offset_from_top

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect ColorChecker card in image.

        Args:
            image: Input image (BGR)

        Returns:
            4 corner points of the card [top-left, top-right, bottom-right, bottom-left]
            or None if not found
        """
        h, w = image.shape[:2]

        # Use fixed position if enabled
        if self.use_fixed_position:
            return self._detect_fixed_position(image)

        # Define search region
        if self.search_region == 'top-left':
            # Search in top-left quadrant with some margin
            roi = image[0:h//2, 0:w//2].copy()
            offset = (0, 0)
        elif self.search_region == 'top-right':
            roi = image[0:h//2, w//2:].copy()
            offset = (w//2, 0)
        else:  # full
            roi = image.copy()
            offset = (0, 0)

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny edge detection with adjusted parameters
        edges = cv2.Canny(blurred, 30, 100)

        # Close gaps with morphology (less aggressive)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter for rectangular contours
        candidates = []
        total_area = h * w

        for contour in contours:
            area = cv2.contourArea(contour)

            # Check area constraints
            if area < total_area * self.min_area_ratio or area > total_area * self.max_area_ratio:
                continue

            # Approximate polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)

            # Look for quadrilateral (4-6 vertices, then reduce to 4)
            if 4 <= len(approx) <= 6:
                # Check aspect ratio (ColorChecker is roughly 2:3 ratio)
                x, y, rw, rh = cv2.boundingRect(approx)
                aspect_ratio = float(rw) / rh if rh > 0 else 0

                # ColorChecker aspect ratio is roughly 1.5 (6 cols / 4 rows)
                if 1.2 < aspect_ratio < 1.8:
                    candidates.append((area, approx, offset))

        # Return largest candidate
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, approx, offset = candidates[0]

            # Order points: top-left, top-right, bottom-right, bottom-left
            pts = approx.reshape(4, 2).astype(np.float32)
            pts[:, 0] += offset[0]
            pts[:, 1] += offset[1]

            return self._order_points(pts)

        return None

    def _detect_fixed_position(self, image: np.ndarray) -> np.ndarray:
        """
        Detect card using fixed position in bottom-right corner.

        NOTE: This method expects the image to already be rotated.

        Args:
            image: Input image (BGR) - should already be rotated

        Returns:
            4 corner points of the card
        """
        h, w = image.shape[:2]

        # Detect card in bottom-right corner
        x_right = w - self.offset_from_right
        x_left = x_right - self.card_width
        y_bottom = h - self.offset_from_top  # From bottom
        y_top = y_bottom - self.card_height

        # Create corner points: TL, TR, BR, BL
        corners = np.array([
            [x_left, y_top],
            [x_right, y_top],
            [x_right, y_bottom],
            [x_left, y_bottom]
        ], dtype=np.float32)

        return corners

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in consistent order: TL, TR, BR, BL."""
        rect = np.zeros((4, 2), dtype=np.float32)

        # Top-left has smallest sum, bottom-right has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Top-right has smallest diff, bottom-left has largest diff
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect


class ColorCheckerExtractor:
    """Extracts color values from ColorChecker patches."""

    def __init__(self, rows=4, cols=6):
        """
        Initialize extractor.

        Args:
            rows: Number of patch rows (4 for Classic)
            cols: Number of patch columns (6 for Classic)
        """
        self.rows = rows
        self.cols = cols

    def extract_patches(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Extract RGB values from all 24 patches.

        Args:
            image: Input image (BGR)
            corners: 4 corner points of the card

        Returns:
            Array of shape (24, 3) with RGB values for each patch
        """
        # Get perspective-corrected card (BGR)
        card_bgr = self._get_card_roi(image, corners)

        # Convert to RGB for proper color extraction
        card_rgb = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        h, w = card_rgb.shape[:2]

        # Calculate patch dimensions with margins
        # ColorChecker has white borders, so we add margins
        margin_h = h * 0.08
        margin_w = w * 0.08

        patch_h = (h - 2 * margin_h) / self.rows
        patch_w = (w - 2 * margin_w) / self.cols

        patches_rgb = []

        for row in range(self.rows):
            for col in range(self.cols):
                # Calculate patch center region (avoid borders)
                y_start = int(margin_h + row * patch_h + patch_h * 0.25)
                y_end = int(margin_h + row * patch_h + patch_h * 0.75)
                x_start = int(margin_w + col * patch_w + patch_w * 0.25)
                x_end = int(margin_w + col * patch_w + patch_w * 0.75)

                # Extract patch from RGB image
                patch = card_rgb[y_start:y_end, x_start:x_end]

                # Calculate mean RGB
                mean_rgb = patch.mean(axis=(0, 1))

                patches_rgb.append(mean_rgb)

        return np.array(patches_rgb, dtype=np.float32)

    def _get_card_roi(self, image: np.ndarray, corners: np.ndarray,
                      output_width=600, output_height=400) -> np.ndarray:
        """Apply perspective transform to get top-down view of card."""
        dst = np.array([
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(image, M, (output_width, output_height))

        return warped


class ColorCorrector:
    """Computes and applies color correction transforms."""

    def __init__(self, method='polynomial', degree=2):
        """
        Initialize corrector.

        Args:
            method: Correction method ('polynomial', 'linear')
            degree: Polynomial degree for polynomial method
        """
        self.method = method
        self.degree = degree
        self.transform = None

    def fit(self, measured: np.ndarray, reference: np.ndarray):
        """
        Compute color correction transform.

        Args:
            measured: Measured RGB values (N, 3)
            reference: Reference RGB values (N, 3)
        """
        if self.method == 'polynomial':
            self.transform = self._fit_polynomial(measured, reference)
        elif self.method == 'linear':
            self.transform = self._fit_linear(measured, reference)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fit_polynomial(self, measured: np.ndarray, reference: np.ndarray):
        """Fit polynomial transform for each channel."""
        transforms = []

        for channel in range(3):
            # Create polynomial features
            X = self._create_polynomial_features(measured, self.degree)
            y = reference[:, channel]

            # Solve least squares
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            transforms.append(coeffs)

        return np.array(transforms)

    def _fit_linear(self, measured: np.ndarray, reference: np.ndarray):
        """Fit linear transform (3x4 matrix)."""
        # Add homogeneous coordinate
        measured_h = np.hstack([measured, np.ones((measured.shape[0], 1))])

        # Solve for each channel
        transform = np.linalg.lstsq(measured_h, reference, rcond=None)[0]
        return transform.T

    def _create_polynomial_features(self, rgb: np.ndarray, degree: int) -> np.ndarray:
        """Create polynomial features from RGB values."""
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        features = [np.ones(len(rgb))]  # Bias term

        # Degree 1: R, G, B
        features.extend([r, g, b])

        if degree >= 2:
            # Degree 2: R^2, G^2, B^2, RG, RB, GB
            features.extend([r*r, g*g, b*b, r*g, r*b, g*b])

        if degree >= 3:
            # Degree 3: R^3, G^3, B^3, R^2G, R^2B, G^2R, G^2B, B^2R, B^2G, RGB
            features.extend([r*r*r, g*g*g, b*b*b,
                           r*r*g, r*r*b, g*g*r, g*g*b, b*b*r, b*b*g,
                           r*g*b])

        return np.column_stack(features)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color correction to entire image.

        Args:
            image: Input image (BGR)

        Returns:
            Corrected image (BGR)
        """
        if self.transform is None:
            raise ValueError("Must call fit() before apply()")

        # Convert to RGB and float
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        h, w, c = img_rgb.shape

        # Reshape to (N, 3)
        pixels = img_rgb.reshape(-1, 3)

        # Apply transform
        if self.method == 'polynomial':
            X = self._create_polynomial_features(pixels, self.degree)
            corrected = X @ self.transform.T
        elif self.method == 'linear':
            pixels_h = np.hstack([pixels, np.ones((pixels.shape[0], 1))])
            corrected = pixels_h @ self.transform.T

        # Clip and reshape
        corrected = np.clip(corrected, 0, 255)
        corrected = corrected.reshape(h, w, c).astype(np.uint8)

        # Convert back to BGR
        return cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)


def process_image(image_path: str, output_path: str, detector: ColorCheckerDetector,
                 extractor: ColorCheckerExtractor, save_debug: bool = False,
                 correction_method: str = 'linear') -> bool:
    """
    Process a single image.

    Args:
        image_path: Path to input image
        output_path: Path to save normalized image
        detector: ColorChecker detector
        extractor: Color extractor
        save_debug: Whether to save debug visualization

    Returns:
        True if successful, False otherwise
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return False

    # Rotate image 90 degrees clockwise
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Detect card in bottom-right corner of rotated image
    corners = detector.detect(image)
    if corners is None:
        print(f"Warning: Could not detect ColorChecker in {image_path}")
        return False

    # Extract patches from rotated image
    measured_rgb = extractor.extract_patches(image, corners)

    # Compute correction
    corrector = ColorCorrector(method=correction_method, degree=2)
    corrector.fit(measured_rgb, COLORCHECKER_REFERENCE_RGB)

    # Apply correction to rotated image
    corrected = corrector.apply(image)

    # Save rotated, corrected output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, corrected)

    # Save debug visualization if requested (rotated image with box)
    if save_debug:
        debug_path = output_path.replace('.jpg', '_debug.jpg')
        debug_img = image.copy()  # Rotated image
        cv2.polylines(debug_img, [corners.astype(np.int32)], True, (0, 255, 0), 3)
        cv2.imwrite(debug_path, debug_img)

    return True


def process_directory(input_dir: str, output_dir: str, save_debug: bool = False,
                     search_region: str = 'top-right', correction_method: str = 'linear'):
    """
    Process all images in directory structure.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        save_debug: Whether to save debug visualizations
        search_region: Region to search for card
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'}
    image_files = [f for f in input_path.rglob('*') if f.suffix in image_extensions]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    # Initialize detector and extractor
    detector = ColorCheckerDetector(search_region=search_region)
    extractor = ColorCheckerExtractor()

    # Process images
    successful = 0
    failed = []

    for i, img_file in enumerate(image_files, 1):
        # Compute relative path to preserve directory structure
        rel_path = img_file.relative_to(input_path)
        out_file = output_path / rel_path

        print(f"[{i}/{len(image_files)}] Processing {rel_path}... ", end='', flush=True)

        if process_image(str(img_file), str(out_file), detector, extractor, save_debug, correction_method):
            successful += 1
            print("✓")
        else:
            failed.append(str(rel_path))
            print("✗")

    # Summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successful: {successful}/{len(image_files)}")
    print(f"Failed: {len(failed)}/{len(image_files)}")

    if failed:
        print(f"\nFailed images:")
        for f in failed[:10]:  # Show first 10
            print(f"  - {f}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

        # Save failed list
        failed_log = output_path / "failed_images.txt"
        with open(failed_log, 'w') as f:
            f.write('\n'.join(failed))
        print(f"\nFull list saved to: {failed_log}")


def main():
    parser = argparse.ArgumentParser(
        description='Normalize colors using ColorChecker Classic Nano card'
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input directory containing images')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for normalized images')
    parser.add_argument('--debug', action='store_true',
                       help='Save debug visualizations showing detected card')
    parser.add_argument('--region', default='top-right',
                       choices=['top-left', 'top-right', 'full'],
                       help='Region to search for ColorChecker card')
    parser.add_argument('--method', default='linear',
                       choices=['linear', 'polynomial'],
                       help='Color correction method (default: linear)')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input directory does not exist: {args.input}")
        sys.exit(1)

    # Process
    process_directory(args.input, args.output, args.debug, args.region, args.method)


if __name__ == '__main__':
    main()
