#!/usr/bin/env python3
"""
Color Correction Method Comparison Script

This script compares multiple color correction methods from the colour-science library
and the local implementation. It creates side-by-side visualizations showing:
- Original image
- Results from each correction method
- Quality metrics (Delta E) for each method

Usage:
    python compare_color_corrections.py --images image1.jpg image2.jpg ... --output comparison_output/
    python compare_color_corrections.py --auto-select 5 --output comparison_output/
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Use non-interactive backend for matplotlib (no display required)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Import local ColorChecker detection and extraction
from colorchecker_normalize import (
    ColorCheckerDetector,
    ColorCheckerExtractor,
    COLORCHECKER_REFERENCE_RGB
)

# Try to import colour-science library
try:
    import colour
    from colour.characterisation import (
        matrix_colour_correction_Cheung2004,
        matrix_colour_correction_Finlayson2015,
        matrix_colour_correction_Vandermonde,
        apply_matrix_colour_correction_Cheung2004,
        apply_matrix_colour_correction_Finlayson2015,
        apply_matrix_colour_correction_Vandermonde
    )
    COLOUR_AVAILABLE = True
except ImportError:
    COLOUR_AVAILABLE = False
    print("Warning: colour-science library not available. Install with: pip install colour-science")


def compute_delta_e(measured: np.ndarray, reference: np.ndarray) -> float:
    """
    Compute average Delta E (CIE76) between measured and reference colors.

    Args:
        measured: Measured RGB values (N, 3) in 0-255 range
        reference: Reference RGB values (N, 3) in 0-255 range

    Returns:
        Average Delta E across all patches
    """
    # Convert RGB to LAB color space (simple approximation)
    # For more accurate results, use proper color space conversion
    def rgb_to_lab_simple(rgb):
        """Simple RGB to LAB conversion using colour library if available."""
        if COLOUR_AVAILABLE:
            # Normalize to 0-1
            rgb_normalized = rgb / 255.0
            # Convert to XYZ then to LAB
            xyz = colour.sRGB_to_XYZ(rgb_normalized)
            lab = colour.XYZ_to_Lab(xyz)
            return lab
        else:
            # Fallback: simple Euclidean distance in RGB space
            return rgb

    measured_lab = rgb_to_lab_simple(measured)
    reference_lab = rgb_to_lab_simple(reference)

    # Compute Euclidean distance
    delta_e = np.sqrt(np.sum((measured_lab - reference_lab) ** 2, axis=1))
    return np.mean(delta_e)


class ColourScienceCorrector:
    """Wrapper for colour-science library correction methods."""

    def __init__(self, method='Cheung2004', degree=1, terms=3):
        """
        Initialize corrector.

        Args:
            method: Correction method name ('Cheung2004', 'Finlayson2015', 'Vandermonde')
            degree: Polynomial degree (for Finlayson2015 and Vandermonde)
            terms: Number of terms (for Cheung2004)
        """
        if not COLOUR_AVAILABLE:
            raise ImportError("colour-science library is required for this corrector")

        self.method = method
        self.degree = degree
        self.terms = terms
        self.correction_matrix = None

    def fit(self, measured: np.ndarray, reference: np.ndarray):
        """
        Compute color correction matrix.

        Args:
            measured: Measured RGB values (N, 3) in 0-255 range
            reference: Reference RGB values (N, 3) in 0-255 range
        """
        # Normalize to 0-1 range for colour-science library
        measured_norm = measured / 255.0
        reference_norm = reference / 255.0

        # Compute correction matrix using the appropriate method
        if self.method == 'Cheung2004':
            self.correction_matrix = matrix_colour_correction_Cheung2004(
                M_T=measured_norm,
                M_R=reference_norm,
                terms=self.terms
            )
        elif self.method == 'Finlayson2015':
            self.correction_matrix = matrix_colour_correction_Finlayson2015(
                M_T=measured_norm,
                M_R=reference_norm,
                degree=self.degree,
                root_polynomial_expansion=True
            )
        elif self.method == 'Vandermonde':
            self.correction_matrix = matrix_colour_correction_Vandermonde(
                M_T=measured_norm,
                M_R=reference_norm,
                degree=self.degree
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color correction to image.

        Args:
            image: Input image (BGR) in 0-255 range

        Returns:
            Corrected image (BGR) in 0-255 range
        """
        if self.correction_matrix is None:
            raise ValueError("Must call fit() before apply()")

        # Convert to RGB and normalize
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w, c = img_rgb.shape

        # Reshape to (N, 3)
        pixels = img_rgb.reshape(-1, 3)

        # Apply correction matrix using the appropriate apply function
        if self.method == 'Cheung2004':
            corrected = apply_matrix_colour_correction_Cheung2004(
                RGB=pixels,
                CCM=self.correction_matrix,
                terms=self.terms
            )
        elif self.method == 'Finlayson2015':
            corrected = apply_matrix_colour_correction_Finlayson2015(
                RGB=pixels,
                CCM=self.correction_matrix,
                degree=self.degree,
                root_polynomial_expansion=True
            )
        elif self.method == 'Vandermonde':
            corrected = apply_matrix_colour_correction_Vandermonde(
                RGB=pixels,
                CCM=self.correction_matrix,
                degree=self.degree
            )

        # Clip and denormalize
        corrected = np.clip(corrected, 0, 1)
        corrected = (corrected * 255).astype(np.uint8)
        corrected = corrected.reshape(h, w, c)

        # Convert back to BGR
        return cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)


class LocalPolynomialCorrector:
    """Local implementation of polynomial color correction."""

    def __init__(self, degree=2):
        """
        Initialize corrector.

        Args:
            degree: Polynomial degree
        """
        self.degree = degree
        self.transform = None

    def fit(self, measured: np.ndarray, reference: np.ndarray):
        """Compute polynomial transform for each channel."""
        transforms = []

        for channel in range(3):
            # Create polynomial features
            X = self._create_polynomial_features(measured, self.degree)
            y = reference[:, channel]

            # Solve least squares
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            transforms.append(coeffs)

        self.transform = np.array(transforms)

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
            # Degree 3
            features.extend([r*r*r, g*g*g, b*b*b,
                           r*r*g, r*r*b, g*g*r, g*g*b, b*b*r, b*b*g,
                           r*g*b])

        return np.column_stack(features)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply polynomial correction to image."""
        if self.transform is None:
            raise ValueError("Must call fit() before apply()")

        # Convert to RGB and float
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        h, w, c = img_rgb.shape

        # Reshape to (N, 3)
        pixels = img_rgb.reshape(-1, 3)

        # Apply transform
        X = self._create_polynomial_features(pixels, self.degree)
        corrected = X @ self.transform.T

        # Clip and reshape
        corrected = np.clip(corrected, 0, 255)
        corrected = corrected.reshape(h, w, c).astype(np.uint8)

        # Convert back to BGR
        return cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)


class LocalLinearCorrector:
    """Local implementation of linear color correction."""

    def __init__(self):
        self.transform = None

    def fit(self, measured: np.ndarray, reference: np.ndarray):
        """Fit linear transform (3x4 matrix)."""
        # Add homogeneous coordinate
        measured_h = np.hstack([measured, np.ones((measured.shape[0], 1))])

        # Solve for each channel
        transform = np.linalg.lstsq(measured_h, reference, rcond=None)[0]
        self.transform = transform.T

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply linear correction to image."""
        if self.transform is None:
            raise ValueError("Must call fit() before apply()")

        # Convert to RGB and float
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        h, w, c = img_rgb.shape

        # Reshape to (N, 3)
        pixels = img_rgb.reshape(-1, 3)
        pixels_h = np.hstack([pixels, np.ones((pixels.shape[0], 1))])
        corrected = pixels_h @ self.transform.T

        # Clip and reshape
        corrected = np.clip(corrected, 0, 255)
        corrected = corrected.reshape(h, w, c).astype(np.uint8)

        # Convert back to BGR
        return cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)


def create_comparison_visualization(
    original_image: np.ndarray,
    corrected_images: Dict[str, np.ndarray],
    delta_e_scores: Dict[str, float],
    output_path: str,
    image_name: str
):
    """
    Create comparison visualization showing original and all corrected versions.

    Args:
        original_image: Original image (BGR)
        corrected_images: Dict mapping method name to corrected image (BGR)
        delta_e_scores: Dict mapping method name to Delta E score
        output_path: Path to save visualization
        image_name: Name of the image being processed
    """
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    corrected_rgb = {name: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                     for name, img in corrected_images.items()}

    # Determine grid layout
    n_methods = len(corrected_images)
    n_cols = min(3, n_methods + 1)  # Max 3 columns
    n_rows = int(np.ceil((n_methods + 1) / n_cols))

    # Create figure
    fig = plt.figure(figsize=(n_cols * 5, n_rows * 5))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)

    # Plot original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(original_rgb)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Plot corrected versions
    for idx, (method_name, corrected_rgb_img) in enumerate(corrected_rgb.items(), start=1):
        row = idx // n_cols
        col = idx % n_cols

        ax = fig.add_subplot(gs[row, col])
        ax.imshow(corrected_rgb_img)

        # Add title with Delta E score
        delta_e = delta_e_scores.get(method_name, 0)
        title = f'{method_name}\n\u0394E = {delta_e:.2f}'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    # Overall title
    fig.suptitle(f'Color Correction Method Comparison\n{image_name}',
                 fontsize=14, fontweight='bold', y=0.98)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def process_image(
    image_path: str,
    output_dir: str,
    detector: ColorCheckerDetector,
    extractor: ColorCheckerExtractor
) -> bool:
    """
    Process a single image with all correction methods.

    Args:
        image_path: Path to input image
        output_dir: Directory to save comparison visualization
        detector: ColorChecker detector
        extractor: Color extractor

    Returns:
        True if successful, False otherwise
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return False

    # Rotate image 90 degrees clockwise (as per original script)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Detect ColorChecker
    corners = detector.detect(image)
    if corners is None:
        print(f"Warning: Could not detect ColorChecker in {image_path}")
        return False

    # Extract measured patch colors
    measured_rgb = extractor.extract_patches(image, corners)

    # Prepare correction methods
    methods = {}

    # Local methods (always available)
    methods['Linear'] = LocalLinearCorrector()
    methods['Polynomial (deg=2)'] = LocalPolynomialCorrector(degree=2)
    methods['Polynomial (deg=3)'] = LocalPolynomialCorrector(degree=3)

    # colour-science methods (if available)
    if COLOUR_AVAILABLE:
        methods['Cheung 2004'] = ColourScienceCorrector(method='Cheung2004', terms=3)
        methods['Finlayson 2015'] = ColourScienceCorrector(method='Finlayson2015', degree=1)
        methods['Vandermonde'] = ColourScienceCorrector(method='Vandermonde', degree=2)

    # Apply all methods
    corrected_images = {}
    delta_e_scores = {}

    for method_name, corrector in methods.items():
        try:
            # Fit correction model
            corrector.fit(measured_rgb, COLORCHECKER_REFERENCE_RGB)

            # Apply correction
            corrected = corrector.apply(image)
            corrected_images[method_name] = corrected

            # Extract corrected patch colors for quality metric
            corrected_patches = extractor.extract_patches(corrected, corners)

            # Compute Delta E
            delta_e = compute_delta_e(corrected_patches, COLORCHECKER_REFERENCE_RGB)
            delta_e_scores[method_name] = delta_e

        except Exception as e:
            print(f"Warning: Failed to apply {method_name}: {e}")
            continue

    # Create comparison visualization
    image_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f'{image_name}_comparison.png')

    create_comparison_visualization(
        image,
        corrected_images,
        delta_e_scores,
        output_path,
        image_name
    )

    print(f"✓ Saved comparison to {output_path}")
    return True


def auto_select_images(data_dir: str, n_images: int = 5) -> List[str]:
    """
    Automatically select diverse test images from data directory.

    Args:
        data_dir: Directory containing images
        n_images: Number of images to select

    Returns:
        List of image paths
    """
    data_path = Path(data_dir)

    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'}
    all_images = [str(f) for f in data_path.rglob('*') if f.suffix in image_extensions]

    if not all_images:
        raise ValueError(f"No images found in {data_dir}")

    # Select evenly spaced images for diversity
    step = max(1, len(all_images) // n_images)
    selected = all_images[::step][:n_images]

    return selected


def main():
    parser = argparse.ArgumentParser(
        description='Compare color correction methods on ColorChecker images'
    )
    parser.add_argument('--images', '-i', nargs='+',
                       help='Input image paths to process')
    parser.add_argument('--auto-select', type=int, metavar='N',
                       help='Automatically select N diverse images from data directory')
    parser.add_argument('--data-dir', default='data',
                       help='Data directory for auto-selection (default: data)')
    parser.add_argument('--output', '-o', default='color_correction_comparisons',
                       help='Output directory for comparison visualizations')
    parser.add_argument('--region', default='top-right',
                       choices=['top-left', 'top-right', 'full'],
                       help='Region to search for ColorChecker card')

    args = parser.parse_args()

    # Determine which images to process
    if args.auto_select:
        print(f"Auto-selecting {args.auto_select} diverse images from {args.data_dir}...")
        image_paths = auto_select_images(args.data_dir, args.auto_select)
    elif args.images:
        image_paths = args.images
    else:
        print("Error: Must specify either --images or --auto-select")
        sys.exit(1)

    print(f"Processing {len(image_paths)} images...")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize detector and extractor
    detector = ColorCheckerDetector(search_region=args.region)
    extractor = ColorCheckerExtractor()

    # Process each image
    successful = 0
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing {Path(image_path).name}...")

        if process_image(image_path, args.output, detector, extractor):
            successful += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successful: {successful}/{len(image_paths)}")
    print(f"Output directory: {args.output}")


if __name__ == '__main__':
    main()
