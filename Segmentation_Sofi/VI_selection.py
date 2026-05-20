#!/usr/bin/env python3
"""
Disease Threshold Analysis for Leaf Images (Simplified - No OpenCV required)

Calculates vegetation indices at pixel level, creates histograms,
and helps determine optimal thresholds for disease detection.
"""

import argparse
import numpy as np
from pathlib import Path
import pickle
import json
from PIL import Image

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def safe_divide(numerator, denominator, fill_value=0.0):
    """Safely divide arrays, handling division by zero."""
    result = np.full_like(numerator, fill_value, dtype=float)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    return result


def calculate_vegetation_indices(image_rgb):
    """
    Calculate vegetation indices at pixel level.

    Args:
        image_rgb: RGB image (H, W, 3) with values 0-255

    Returns:
        Dictionary of vegetation indices (all as 2D arrays)
    """
    # Extract RGB channels (0-255)
    R = image_rgb[:, :, 0].astype(float)
    G = image_rgb[:, :, 1].astype(float)
    B = image_rgb[:, :, 2].astype(float)

    # Calculate normalized RGB (r, g, b) - values between 0 and 1
    sum_rgb = R + G + B
    r = safe_divide(R, sum_rgb, fill_value=0)
    g = safe_divide(G, sum_rgb, fill_value=0)
    b = safe_divide(B, sum_rgb, fill_value=0)

    indices = {}

    # Key indices for disease detection
    # 1. NGRDI (Normalized Green-Red Difference Index) - Good for health status
    indices['NGRDI'] = safe_divide(G - R, G + R, fill_value=0)

    # 2. ExG (Excess Green) - Commonly used for vegetation segmentation
    indices['ExG'] = 2*g - r - b

    # 3. ExR (Excess Red) - Can highlight diseased/senescing tissue
    indices['ExR'] = 1.4*r - g

    # 4. ExGR (Excess Green minus Excess Red)
    indices['ExGR'] = indices['ExG'] - indices['ExR']

    # 5. GLI (Green Leaf Index)
    indices['GLI'] = safe_divide(2*G - R - B, 2*G + R + B, fill_value=0)

    # 6. VARI (Visible Atmospherically Resistant Index)
    indices['VARI'] = safe_divide(G - R, G + R - B, fill_value=0)

    # 7. MGRVI (Modified Green Red Vegetation Index)
    indices['MGRVI'] = safe_divide(G**2 - R**2, G**2 + R**2, fill_value=0)

    # 8. Redness Index (custom for disease)
    # Higher values indicate more red/brown coloration (disease symptom)
    indices['Redness'] = safe_divide(R, G + 1, fill_value=0)

    # 9. Brownness Index (custom for disease)
    # Brown = high R, moderate G, low B
    indices['Brownness'] = safe_divide(R * G, (B + 1) * (R + G + B + 1), fill_value=0)

    # 10. Green Chromatic Coordinate
    indices['GCC'] = g

    # 11. Red Chromatic Coordinate
    indices['RCC'] = r

    return indices


def create_leaf_mask(image_rgb, threshold=10):
    """
    Create a binary mask to separate leaf from background.

    Args:
        image_rgb: RGB image
        threshold: Intensity threshold for background separation

    Returns:
        Binary mask (leaf=True, background=False)
    """
    # Convert to grayscale
    gray = np.mean(image_rgb, axis=2)

    # Background is typically very dark (black)
    mask = gray > threshold

    return mask


def extract_valid_pixels(index_data, mask):
    """
    Extract valid (non-background, finite) pixel values from an index.

    Args:
        index_data: 2D array of index values
        mask: Binary mask (True=leaf, False=background)

    Returns:
        1D array of valid pixel values
    """
    # Only keep pixels that are:
    # 1. Part of the leaf (mask == True)
    # 2. Finite (not NaN or Inf)
    valid_mask = mask & np.isfinite(index_data)
    valid_pixels = index_data[valid_mask]

    return valid_pixels


def process_images_to_pixel_data(image_paths, output_dir):
    """
    Process all images and extract pixel-level vegetation index data.

    Args:
        image_paths: List of image file paths
        output_dir: Directory to save results

    Returns:
        Dictionary containing flattened pixel data for all indices and all images
    """
    print("\n" + "="*80)
    print("EXTRACTING PIXEL-LEVEL VEGETATION INDEX DATA")
    print("="*80)

    # Data structure: {index_name: {image_name: pixel_values_array}}
    pixel_data = {}
    image_info = []

    for img_path in image_paths:
        image_name = Path(img_path).stem
        print(f"\nProcessing: {image_name}")

        # Load image using PIL
        try:
            img = Image.open(str(img_path))
            image_rgb = np.array(img.convert('RGB'))
        except Exception as e:
            print(f"  Error: Could not load {img_path}: {e}")
            continue

        print(f"  Image size: {image_rgb.shape[1]}x{image_rgb.shape[0]}")

        # Create leaf mask
        mask = create_leaf_mask(image_rgb)
        leaf_pixels = np.sum(mask)
        print(f"  Leaf pixels: {leaf_pixels:,}")

        # Calculate indices
        indices = calculate_vegetation_indices(image_rgb)

        # Extract valid pixels for each index
        for index_name, index_data in indices.items():
            if index_name not in pixel_data:
                pixel_data[index_name] = {}

            valid_pixels = extract_valid_pixels(index_data, mask)
            pixel_data[index_name][image_name] = valid_pixels

            if len(valid_pixels) > 0:
                print(f"    {index_name}: {len(valid_pixels):,} valid pixels, "
                      f"mean={np.mean(valid_pixels):.4f}, std={np.std(valid_pixels):.4f}")

        # Store image info
        image_info.append({
            'name': image_name,
            'path': str(img_path),
            'width': image_rgb.shape[1],
            'height': image_rgb.shape[0],
            'leaf_pixels': int(leaf_pixels)
        })

    # Save pixel data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as pickle for easy reloading
    pickle_path = output_path / 'pixel_data.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(pixel_data, f)
    print(f"\n✓ Pixel data saved to: {pickle_path}")

    # Save image info as JSON
    json_path = output_path / 'image_info.json'
    with open(json_path, 'w') as f:
        json.dump(image_info, f, indent=2)
    print(f"✓ Image info saved to: {json_path}")

    return pixel_data, image_info


def create_combined_histograms(pixel_data, output_dir, bins=100):
    """
    Create histograms showing distribution of each index across all images.

    Args:
        pixel_data: Dictionary of pixel data from process_images_to_pixel_data
        output_dir: Directory to save histogram plots
        bins: Number of histogram bins
    """
    print("\n" + "="*80)
    print("CREATING COMBINED HISTOGRAMS")
    print("="*80)

    output_path = Path(output_dir) / 'histograms'
    output_path.mkdir(parents=True, exist_ok=True)

    index_names = list(pixel_data.keys())

    for index_name in index_names:
        print(f"\nCreating histogram for {index_name}...")

        # Collect all pixel values across all images
        all_pixels = []
        for image_name, pixels in pixel_data[index_name].items():
            all_pixels.extend(pixels)

        all_pixels = np.array(all_pixels)

        if len(all_pixels) == 0:
            print(f"  Warning: No valid pixels for {index_name}")
            continue

        # Calculate statistics
        mean_val = np.mean(all_pixels)
        median_val = np.median(all_pixels)
        std_val = np.std(all_pixels)
        q25 = np.percentile(all_pixels, 25)
        q75 = np.percentile(all_pixels, 75)

        # Create histogram
        fig, ax = plt.subplots(figsize=(12, 6))

        counts, bin_edges, patches = ax.hist(all_pixels, bins=bins, alpha=0.7,
                                            color='green', edgecolor='black', density=True)

        # Add vertical lines for statistics
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='blue', linestyle='--', linewidth=2,
                  label=f'Median: {median_val:.4f}')
        ax.axvline(q25, color='orange', linestyle=':', linewidth=1.5,
                  label=f'Q25: {q25:.4f}')
        ax.axvline(q75, color='orange', linestyle=':', linewidth=1.5,
                  label=f'Q75: {q75:.4f}')

        ax.set_xlabel(f'{index_name} Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{index_name} Distribution (All Images Combined)\n'
                    f'Total pixels: {len(all_pixels):,} | Std: {std_val:.4f}',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Save plot
        plot_path = output_path / f'{index_name}_combined_histogram.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {plot_path}")

    print(f"\n✓ All histograms saved to: {output_path}/")


def create_per_image_histograms(pixel_data, output_dir, bins=50):
    """
    Create overlaid histograms showing each image separately for comparison.

    Args:
        pixel_data: Dictionary of pixel data
        output_dir: Directory to save plots
        bins: Number of histogram bins
    """
    print("\n" + "="*80)
    print("CREATING PER-IMAGE COMPARISON HISTOGRAMS")
    print("="*80)

    output_path = Path(output_dir) / 'histograms_per_image'
    output_path.mkdir(parents=True, exist_ok=True)

    index_names = list(pixel_data.keys())

    for index_name in index_names:
        print(f"\nCreating per-image histogram for {index_name}...")

        fig, ax = plt.subplots(figsize=(14, 8))

        # Get all image names
        image_names = list(pixel_data[index_name].keys())

        # Use colormap for different images
        colors = plt.cm.tab20(np.linspace(0, 1, min(len(image_names), 20)))

        # Plot histogram for each image
        for idx, (image_name, pixels) in enumerate(pixel_data[index_name].items()):
            if len(pixels) > 0:
                color_idx = idx % len(colors)
                ax.hist(pixels, bins=bins, alpha=0.3,
                       color=colors[color_idx], edgecolor='black',
                       linewidth=0.5, density=True, label=image_name if idx < 15 else "")

        ax.set_xlabel(f'{index_name} Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{index_name} Distribution by Image',
                    fontsize=13, fontweight='bold')

        # Only show legend if not too many images
        if len(image_names) <= 15:
            ax.legend(fontsize=7, loc='best', ncol=2)
        else:
            ax.text(0.98, 0.98, f'{len(image_names)} images',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.grid(True, alpha=0.3)

        # Save plot
        plot_path = output_path / f'{index_name}_per_image_histogram.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {plot_path}")

    print(f"\n✓ All per-image histograms saved to: {output_path}/")


def otsu_threshold(data):
    """
    Calculate Otsu's threshold for 1D data.

    Args:
        data: 1D array of values

    Returns:
        Optimal threshold value
    """
    # Normalize data to 0-255 range
    data_min = np.min(data)
    data_max = np.max(data)

    if data_max - data_min < 1e-6:
        return np.mean(data)

    normalized = ((data - data_min) / (data_max - data_min) * 255).astype(int)

    # Calculate histogram
    hist, _ = np.histogram(normalized, bins=256, range=(0, 256))
    hist = hist.astype(float)

    # Total number of pixels
    total = len(normalized)

    # Calculate cumulative sums
    sum_total = np.sum(np.arange(256) * hist)

    weight_bg = 0
    sum_bg = 0
    max_variance = 0
    threshold = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue

        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]

        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        # Calculate between-class variance
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance > max_variance:
            max_variance = variance
            threshold = t

    # Convert back to original scale
    threshold_original = (threshold / 255.0) * (data_max - data_min) + data_min

    return threshold_original


def suggest_thresholds(pixel_data, output_dir):
    """
    Suggest threshold values for disease detection based on histogram analysis.

    Args:
        pixel_data: Dictionary of pixel data
        output_dir: Directory to save results

    Returns:
        Dictionary of suggested thresholds for each index
    """
    print("\n" + "="*80)
    print("CALCULATING SUGGESTED THRESHOLDS")
    print("="*80)

    thresholds = {}

    for index_name, image_data in pixel_data.items():
        print(f"\n{index_name}:")

        # Combine all pixels
        all_pixels = []
        for pixels in image_data.values():
            all_pixels.extend(pixels)
        all_pixels = np.array(all_pixels)

        if len(all_pixels) == 0:
            continue

        # Calculate multiple threshold options
        index_thresholds = {}

        # 1. Otsu's method
        try:
            otsu_thresh = otsu_threshold(all_pixels)
            index_thresholds['otsu'] = float(otsu_thresh)
            print(f"  Otsu threshold: {otsu_thresh:.4f}")
        except:
            print(f"  Could not calculate Otsu threshold")

        # 2. Mean threshold
        mean_thresh = np.mean(all_pixels)
        index_thresholds['mean'] = float(mean_thresh)
        print(f"  Mean threshold: {mean_thresh:.4f}")

        # 3. Median threshold
        median_thresh = np.median(all_pixels)
        index_thresholds['median'] = float(median_thresh)
        print(f"  Median threshold: {median_thresh:.4f}")

        # 4. Percentile thresholds (useful for disease = minority class)
        for pct in [10, 25, 75, 90]:
            pct_thresh = np.percentile(all_pixels, pct)
            index_thresholds[f'p{pct}'] = float(pct_thresh)
            print(f"  {pct}th percentile: {pct_thresh:.4f}")

        # 5. Mean - std and Mean + std
        std_val = np.std(all_pixels)
        index_thresholds['mean_minus_std'] = float(mean_thresh - std_val)
        index_thresholds['mean_plus_std'] = float(mean_thresh + std_val)
        print(f"  Mean - std: {mean_thresh - std_val:.4f}")
        print(f"  Mean + std: {mean_thresh + std_val:.4f}")

        thresholds[index_name] = index_thresholds

    # Save thresholds to JSON
    output_path = Path(output_dir)
    json_path = output_path / 'suggested_thresholds.json'
    with open(json_path, 'w') as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n✓ Thresholds saved to: {json_path}")

    return thresholds


def visualize_threshold_effects(image_paths, thresholds, output_dir, n_images=5):
    """
    Visualize the effect of different thresholds on sample images.

    Args:
        image_paths: List of image paths
        thresholds: Dictionary of threshold values
        output_dir: Directory to save visualizations
        n_images: Number of sample images to visualize
    """
    print("\n" + "="*80)
    print("VISUALIZING THRESHOLD EFFECTS ON SAMPLE IMAGES")
    print("="*80)

    output_path = Path(output_dir) / 'threshold_visualizations'
    output_path.mkdir(parents=True, exist_ok=True)

    # Select sample images
    sample_images = image_paths[:n_images]

    # Select key indices to visualize
    key_indices = ['NGRDI', 'ExG', 'ExGR', 'Redness', 'Brownness']

    for img_path in sample_images:
        image_name = Path(img_path).stem
        print(f"\nProcessing: {image_name}")

        # Load image
        try:
            img = Image.open(str(img_path))
            image_rgb = np.array(img.convert('RGB'))
        except Exception as e:
            print(f"  Error loading {img_path}: {e}")
            continue

        mask = create_leaf_mask(image_rgb)

        # Calculate indices
        indices = calculate_vegetation_indices(image_rgb)

        # Create visualization for each key index
        for index_name in key_indices:
            if index_name not in indices or index_name not in thresholds:
                continue

            index_data = indices[index_name]
            index_thresholds = thresholds[index_name]

            # Create figure with multiple threshold visualizations
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()

            # Original image
            axes[0].imshow(image_rgb)
            axes[0].set_title('Original Image', fontweight='bold')
            axes[0].axis('off')

            # Index heatmap
            im = axes[1].imshow(index_data, cmap='RdYlGn')
            axes[1].set_title(f'{index_name} Index', fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046)

            # Different threshold methods
            threshold_methods = ['otsu', 'mean', 'median', 'p25', 'mean_minus_std', 'mean_plus_std']

            for idx, method in enumerate(threshold_methods):
                if method not in index_thresholds:
                    continue

                thresh_val = index_thresholds[method]

                # Apply threshold (disease = values below threshold for most indices)
                # For Redness/Brownness, disease = values above threshold
                if index_name in ['Redness', 'Brownness', 'ExR', 'RCC']:
                    disease_mask = (index_data > thresh_val) & mask
                else:
                    disease_mask = (index_data < thresh_val) & mask

                # Create visualization
                vis = image_rgb.copy()
                vis[disease_mask] = [255, 0, 0]  # Mark disease pixels as red

                ax = axes[idx + 2]
                ax.imshow(vis)

                disease_pct = (np.sum(disease_mask) / np.sum(mask)) * 100 if np.sum(mask) > 0 else 0
                ax.set_title(f'{method.upper()}\nThresh={thresh_val:.3f} | Disease={disease_pct:.1f}%',
                           fontsize=9, fontweight='bold')
                ax.axis('off')

            plt.suptitle(f'{image_name} - {index_name} Threshold Comparison',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()

            # Save
            save_path = output_path / f'{image_name}_{index_name}_thresholds.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Saved: {save_path.name}")

    print(f"\n✓ Threshold visualizations saved to: {output_path}/")


def create_threshold_selection_report(thresholds, pixel_data, output_dir):
    """
    Create a detailed report to help select optimal thresholds.

    Args:
        thresholds: Dictionary of threshold values
        pixel_data: Dictionary of pixel data
        output_dir: Directory to save report
    """
    output_path = Path(output_dir) / 'threshold_selection_report.txt'

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DISEASE THRESHOLD SELECTION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("This report helps you select appropriate thresholds for disease detection.\n\n")

        f.write("RECOMMENDED INDICES FOR DISEASE DETECTION:\n")
        f.write("-"*80 + "\n\n")

        f.write("1. NGRDI (Normalized Green-Red Difference Index)\n")
        f.write("   - Healthy tissue: HIGH values (more green)\n")
        f.write("   - Diseased tissue: LOW values (less green, more red/brown)\n")
        f.write("   - Threshold: Disease = NGRDI < threshold\n\n")

        f.write("2. ExGR (Excess Green minus Excess Red)\n")
        f.write("   - Healthy tissue: HIGH values (green dominates)\n")
        f.write("   - Diseased tissue: LOW/NEGATIVE values (red dominates)\n")
        f.write("   - Threshold: Disease = ExGR < threshold\n\n")

        f.write("3. Redness Index (R/(G+1))\n")
        f.write("   - Healthy tissue: LOW values\n")
        f.write("   - Diseased tissue: HIGH values (brown/red spots)\n")
        f.write("   - Threshold: Disease = Redness > threshold\n\n")

        f.write("4. Brownness Index (R*G/((B+1)*(R+G+B+1)))\n")
        f.write("   - Healthy tissue: LOW values\n")
        f.write("   - Diseased tissue: HIGH values (brown necrotic spots)\n")
        f.write("   - Threshold: Disease = Brownness > threshold\n\n")

        f.write("\n" + "="*80 + "\n")
        f.write("CALCULATED THRESHOLD VALUES\n")
        f.write("="*80 + "\n\n")

        for index_name, thresh_dict in thresholds.items():
            f.write(f"\n{index_name}:\n")
            f.write("-"*40 + "\n")

            # Get pixel stats
            all_pixels = []
            for pixels in pixel_data[index_name].values():
                all_pixels.extend(pixels)
            all_pixels = np.array(all_pixels)

            f.write(f"  Data range: [{np.min(all_pixels):.4f}, {np.max(all_pixels):.4f}]\n")
            f.write(f"  Mean: {np.mean(all_pixels):.4f}\n")
            f.write(f"  Std: {np.std(all_pixels):.4f}\n\n")

            f.write("  Suggested thresholds:\n")
            for method, value in sorted(thresh_dict.items()):
                f.write(f"    {method:20s}: {value:8.4f}\n")
            f.write("\n")

        f.write("\n" + "="*80 + "\n")
        f.write("HOW TO USE THIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("1. Review the histogram plots in the 'histograms/' directory\n")
        f.write("   - Look for bimodal distributions (healthy vs diseased peaks)\n\n")

        f.write("2. Examine the 'threshold_visualizations/' directory\n")
        f.write("   - See how different thresholds separate healthy from diseased tissue\n\n")

        f.write("3. Start with these thresholds:\n")
        f.write("   - NGRDI: Try 'p25' (25th percentile) - captures lower 25% as disease\n")
        f.write("   - ExGR: Try 'mean' or 'p25'\n")
        f.write("   - Redness: Try 'p75' (75th percentile) - captures upper 25% as disease\n")
        f.write("   - Brownness: Try 'p75' or 'p90'\n\n")

        f.write("4. Combine multiple indices for robust detection:\n")
        f.write("   - Disease = (NGRDI < threshold_low) OR (Redness > threshold_high)\n")
        f.write("   - Disease = (ExGR < threshold) AND (Brownness > threshold)\n\n")

        f.write("5. Validate on a subset of images and adjust thresholds as needed\n\n")

    print(f"\n✓ Threshold selection report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Disease threshold analysis using vegetation indices"
    )
    parser.add_argument(
        '--input-dir', '-i', type=str, default='.',
        help='Input directory containing leaf images'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, default='disease_threshold_analysis',
        help='Output directory for results'
    )
    parser.add_argument(
        '--pattern', '-p', type=str, default='*_leaf.png',
        help='File pattern to match (default: *_leaf.png)'
    )
    parser.add_argument(
        '--max-images', '-n', type=int, default=None,
        help='Maximum number of images to process (default: all)'
    )

    args = parser.parse_args()

    # Find images
    input_dir = Path(args.input_dir)
    image_files = sorted(list(input_dir.glob(args.pattern)))

    if args.max_images:
        image_files = image_files[:args.max_images]

    if not image_files:
        print(f"Error: No images found matching '{args.pattern}' in '{input_dir}'")
        return

    print(f"\n{'='*80}")
    print(f"DISEASE THRESHOLD ANALYSIS")
    print(f"{'='*80}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Images found: {len(image_files)}")
    print(f"{'='*80}\n")

    # Step 1: Process images and extract pixel data
    pixel_data, image_info = process_images_to_pixel_data(image_files, args.output_dir)

    # Step 2: Create combined histograms
    create_combined_histograms(pixel_data, args.output_dir, bins=100)

    # Step 3: Create per-image histograms
    create_per_image_histograms(pixel_data, args.output_dir, bins=50)

    # Step 4: Suggest thresholds
    thresholds = suggest_thresholds(pixel_data, args.output_dir)

    # Step 5: Visualize threshold effects
    visualize_threshold_effects(image_files, thresholds, args.output_dir, n_images=min(5, len(image_files)))

    # Step 6: Create selection report
    create_threshold_selection_report(thresholds, pixel_data, args.output_dir)

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}/")
    print("\nGenerated outputs:")
    print(f"  1. pixel_data.pkl - Flattened pixel data for all indices")
    print(f"  2. histograms/ - Combined histograms for all indices")
    print(f"  3. histograms_per_image/ - Per-image comparison histograms")
    print(f"  4. suggested_thresholds.json - Calculated threshold values")
    print(f"  5. threshold_visualizations/ - Visual comparison of thresholds")
    print(f"  6. threshold_selection_report.txt - Detailed guidance")
    print("\nNext steps:")
    print("  1. Review histograms to understand data distribution")
    print("  2. Check threshold visualizations to see disease detection")
    print("  3. Read the threshold selection report for recommendations")
    print("  4. Adjust thresholds based on your specific disease symptoms")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
