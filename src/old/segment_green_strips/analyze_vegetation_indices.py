#!/usr/bin/env python3
"""
Vegetation Index Analysis for Sorghum Leaf Images

Implements vegetation indices from Table 1 of:
"A review of vegetation indices" Plant Methods (2019)
https://plantmethods.biomedcentral.com/articles/10.1186/s13007-019-0418-8/tables/1

Focus: Identify indices that provide best contrast for midrib detection.
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
import csv
import warnings

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

warnings.filterwarnings('ignore', category=RuntimeWarning)


def safe_divide(numerator, denominator, fill_value=0.0):
    """Safely divide arrays, handling division by zero."""
    return np.divide(numerator, denominator,
                    out=np.full_like(numerator, fill_value, dtype=float),
                    where=denominator!=0)


def calculate_vegetation_indices(image_rgb):
    """
    Calculate all vegetation indices from Table 1.

    Args:
        image_rgb: RGB image (H, W, 3) with values 0-255

    Returns:
        Dictionary of vegetation indices
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

    # 1. RGBVI (RGB Vegetation Index)
    # Formula: (G²-(R*B))/(G²+(R*B))
    numerator = G**2 - (R * B)
    denominator = G**2 + (R * B)
    indices['RGBVI'] = safe_divide(numerator, denominator, fill_value=0)

    # 2. GLI (Green Leaf Index)
    # Formula: (2*G-R-B)/(2*G+R+B)
    numerator = 2*G - R - B
    denominator = 2*G + R + B
    indices['GLI'] = safe_divide(numerator, denominator, fill_value=0)

    # 3. VARI (Visible Atmospherically Resistant Index)
    # Formula: (G-R)/(G+R-B)
    numerator = G - R
    denominator = G + R - B
    indices['VARI'] = safe_divide(numerator, denominator, fill_value=0)

    # 4. NGRDI (Normalized Green-Red Difference Index)
    # Formula: (G-R)/(G+R)
    numerator = G - R
    denominator = G + R
    indices['NGRDI'] = safe_divide(numerator, denominator, fill_value=0)

    # 5. NGBDI (Normalized Green-Blue Difference Index)
    # Formula: (G-B)/(G+B)
    numerator = G - B
    denominator = G + B
    indices['NGBDI'] = safe_divide(numerator, denominator, fill_value=0)

    # 6. ExG (Excess Green)
    # Formula: 2*g - r - b (using normalized RGB)
    indices['ExG'] = 2*g - r - b

    # 7. ExR (Excess Red)
    # Formula: 1.4*r - g
    indices['ExR'] = 1.4*r - g

    # 8. ExB (Excess Blue)
    # Formula: 1.4*b - g
    indices['ExB'] = 1.4*b - g

    # 9. ExGR (Excess Green minus Excess Red)
    # Formula: ExG - ExR
    indices['ExGR'] = indices['ExG'] - indices['ExR']

    # 10. CIVE (Color Index of Vegetation Extraction)
    # Formula: 0.441*r - 0.811*g + 0.385*b + 18.78745
    indices['CIVE'] = 0.441*r - 0.811*g + 0.385*b + 18.78745

    # 11. VEG (Vegetativeness)
    # Formula: g / (r^a * b^(1-a)) where a = 0.667
    a = 0.667
    denominator = (r**a) * (b**(1-a))
    indices['VEG'] = safe_divide(g, denominator, fill_value=0)

    # 12. COM (Combination)
    # Formula: 0.25*ExG + 0.30*ExGR + 0.33*CIVE + 0.12*VEG
    indices['COM'] = (0.25*indices['ExG'] +
                     0.30*indices['ExGR'] +
                     0.33*indices['CIVE'] +
                     0.12*indices['VEG'])

    # 13. TGI (Triangular Greenness Index)
    # Formula: g - 0.39*r - 0.61*b
    indices['TGI'] = g - 0.39*r - 0.61*b

    # 14. MGRVI (Modified Green Red Vegetation Index)
    # Formula: (G²-R²)/(G²+R²)
    numerator = G**2 - R**2
    denominator = G**2 + R**2
    indices['MGRVI'] = safe_divide(numerator, denominator, fill_value=0)

    # 15. RGVBI (Red Green Blue Vegetation Index)
    # Formula: (G-(B*R))/(G²+(B*R))
    numerator = G - (B * R)
    denominator = G**2 + (B * R)
    indices['RGVBI'] = safe_divide(numerator, denominator, fill_value=0)

    return indices


def get_index_info():
    """
    Get information about each vegetation index including expected ranges and colormaps.

    Returns:
        Dictionary with index metadata
    """
    return {
        'RGBVI': {
            'name': 'RGB Vegetation Index',
            'range': (-1, 1),
            'cmap': 'RdYlGn',
            'description': 'Sensitive to green vegetation'
        },
        'GLI': {
            'name': 'Green Leaf Index',
            'range': (-1, 1),
            'cmap': 'RdYlGn',
            'description': 'Emphasizes green reflectance'
        },
        'VARI': {
            'name': 'Visible Atmospherically Resistant Index',
            'range': (-2, 2),
            'cmap': 'RdYlGn',
            'description': 'Resistant to atmospheric effects'
        },
        'NGRDI': {
            'name': 'Normalized Green-Red Difference Index',
            'range': (-1, 1),
            'cmap': 'RdYlGn',
            'description': 'Green vs red contrast'
        },
        'NGBDI': {
            'name': 'Normalized Green-Blue Difference Index',
            'range': (-1, 1),
            'cmap': 'RdYlGn',
            'description': 'Green vs blue contrast'
        },
        'ExG': {
            'name': 'Excess Green',
            'range': (-1, 1),
            'cmap': 'RdYlGn',
            'description': 'Excess green component'
        },
        'ExR': {
            'name': 'Excess Red',
            'range': (-1, 1),
            'cmap': 'RdBu_r',
            'description': 'Excess red component'
        },
        'ExB': {
            'name': 'Excess Blue',
            'range': (-1, 1),
            'cmap': 'RdBu',
            'description': 'Excess blue component'
        },
        'ExGR': {
            'name': 'Excess Green minus Excess Red',
            'range': (-2, 2),
            'cmap': 'RdYlGn',
            'description': 'ExG - ExR'
        },
        'CIVE': {
            'name': 'Color Index of Vegetation Extraction',
            'range': (15, 22),
            'cmap': 'RdYlGn',
            'description': 'Color-based vegetation index'
        },
        'VEG': {
            'name': 'Vegetativeness',
            'range': (0, 5),
            'cmap': 'RdYlGn',
            'description': 'Ratio-based vegetation measure'
        },
        'COM': {
            'name': 'Combination Index',
            'range': (-2, 10),
            'cmap': 'RdYlGn',
            'description': 'Weighted combination of indices'
        },
        'TGI': {
            'name': 'Triangular Greenness Index',
            'range': (-0.5, 0.5),
            'cmap': 'RdYlGn',
            'description': 'Triangular geometry in RGB space'
        },
        'MGRVI': {
            'name': 'Modified Green Red Vegetation Index',
            'range': (-1, 1),
            'cmap': 'RdYlGn',
            'description': 'Modified version of GRVI'
        },
        'RGVBI': {
            'name': 'Red Green Blue Vegetation Index',
            'range': (-1, 1),
            'cmap': 'RdYlGn',
            'description': 'Three-band vegetation index'
        },
    }


def calculate_index_statistics(index_data, index_name):
    """
    Calculate statistics for a vegetation index.

    Args:
        index_data: 2D numpy array of index values
        index_name: Name of the index

    Returns:
        Dictionary of statistics
    """
    # Remove NaN and Inf values for statistics
    valid_data = index_data[np.isfinite(index_data)]

    if len(valid_data) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan,
            'q25': np.nan,
            'q75': np.nan,
            'valid_pixels': 0,
            'total_pixels': index_data.size
        }

    stats = {
        'mean': np.mean(valid_data),
        'std': np.std(valid_data),
        'min': np.min(valid_data),
        'max': np.max(valid_data),
        'median': np.median(valid_data),
        'q25': np.percentile(valid_data, 25),
        'q75': np.percentile(valid_data, 75),
        'valid_pixels': len(valid_data),
        'total_pixels': index_data.size
    }

    return stats


def create_index_visualization(index_data, index_name, info, output_path, image_name):
    """
    Create visualization for a single vegetation index.

    Args:
        index_data: 2D array of index values
        index_name: Name of the index
        info: Metadata about the index
        output_path: Path to save visualization
        image_name: Name of the image
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap
    vmin, vmax = info['range']
    im = axes[0].imshow(index_data, cmap=info['cmap'], vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{info['name']}\n{image_name}", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Histogram
    valid_data = index_data[np.isfinite(index_data)]
    if len(valid_data) > 0:
        axes[1].hist(valid_data, bins=100, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(np.mean(valid_data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(valid_data):.3f}')
        axes[1].axvline(np.median(valid_data), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(valid_data):.3f}')
        axes[1].set_xlabel('Index Value', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title(f'Distribution ({index_name})', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_panel(image_rgb, indices, output_path, image_name):
    """
    Create multi-panel comparison showing all vegetation indices.

    Args:
        image_rgb: Original RGB image
        indices: Dictionary of calculated indices
        output_path: Path to save visualization
        image_name: Name of the image
    """
    info_dict = get_index_info()

    # Create 4x4 grid (16 panels: 1 original + 15 indices)
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)

    # Original image
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image_rgb)
    ax.set_title('Original Image', fontsize=10, fontweight='bold')
    ax.axis('off')

    # Plot each index
    index_names = ['RGBVI', 'GLI', 'VARI', 'NGRDI', 'NGBDI', 'ExG', 'ExR', 'ExB',
                   'ExGR', 'CIVE', 'VEG', 'COM', 'TGI', 'MGRVI', 'RGVBI']

    for idx, index_name in enumerate(index_names):
        row = (idx + 1) // 4
        col = (idx + 1) % 4

        ax = fig.add_subplot(gs[row, col])

        info = info_dict[index_name]
        data = indices[index_name]

        vmin, vmax = info['range']
        im = ax.imshow(data, cmap=info['cmap'], vmin=vmin, vmax=vmax)
        ax.set_title(index_name, fontsize=10, fontweight='bold')
        ax.axis('off')

        # Add small colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f'Vegetation Indices Comparison: {image_name}',
                fontsize=14, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def apply_otsu_thresholding(indices):
    """
    Apply Otsu thresholding to each index to create binary masks.

    Args:
        indices: Dictionary of vegetation indices

    Returns:
        Dictionary of binary masks and thresholds
    """
    masks = {}
    thresholds = {}

    for index_name, data in indices.items():
        # Normalize to 0-255 for Otsu
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            masks[index_name] = np.zeros_like(data, dtype=np.uint8)
            thresholds[index_name] = None
            continue

        data_min = np.min(valid_data)
        data_max = np.max(valid_data)

        if data_max - data_min < 1e-6:
            masks[index_name] = np.zeros_like(data, dtype=np.uint8)
            thresholds[index_name] = None
            continue

        # Normalize to 0-255
        normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)

        # Replace NaN/Inf with 0
        normalized[~np.isfinite(data)] = 0

        # Apply Otsu thresholding
        threshold, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert threshold back to original scale
        original_threshold = (threshold / 255.0) * (data_max - data_min) + data_min

        masks[index_name] = binary
        thresholds[index_name] = original_threshold

    return masks, thresholds


def create_thresholding_comparison(indices, masks, thresholds, output_path, image_name):
    """
    Create comparison of Otsu thresholding results for all indices.

    Args:
        indices: Dictionary of vegetation indices
        masks: Dictionary of binary masks
        thresholds: Dictionary of threshold values
        output_path: Path to save visualization
        image_name: Name of the image
    """
    index_names = ['RGBVI', 'GLI', 'VARI', 'NGRDI', 'NGBDI', 'ExG', 'ExR',
                   'ExGR', 'CIVE', 'VEG', 'COM', 'TGI', 'MGRVI', 'RGVBI']

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(5, 3, figure=fig, hspace=0.3, wspace=0.25)

    for idx, index_name in enumerate(index_names):
        row = idx // 3
        col = idx % 3

        ax = fig.add_subplot(gs[row, col])

        mask = masks[index_name]
        threshold = thresholds[index_name]

        ax.imshow(mask, cmap='gray')

        if threshold is not None:
            title = f"{index_name}\n(threshold={threshold:.3f})"
        else:
            title = f"{index_name}\n(no threshold)"

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

        # Add pixel count
        white_pixels = np.sum(mask == 255)
        total_pixels = mask.size
        percentage = (white_pixels / total_pixels) * 100
        ax.text(0.02, 0.98, f'{percentage:.1f}% white',
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(f'Otsu Thresholding Results: {image_name}',
                fontsize=14, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_correlation_matrix(indices):
    """
    Calculate correlation matrix between all vegetation indices.

    Args:
        indices: Dictionary of vegetation indices

    Returns:
        Correlation matrix, index names
    """
    index_names = list(indices.keys())
    n_indices = len(index_names)

    # Flatten each index and stack
    data_matrix = []
    for name in index_names:
        flat_data = indices[name].ravel()
        # Replace NaN/Inf with 0
        flat_data[~np.isfinite(flat_data)] = 0
        data_matrix.append(flat_data)

    data_matrix = np.array(data_matrix)

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(data_matrix)

    return corr_matrix, index_names


def create_correlation_heatmap(corr_matrix, index_names, output_path, image_name):
    """
    Create correlation heatmap for vegetation indices.

    Args:
        corr_matrix: Correlation matrix
        index_names: List of index names
        output_path: Path to save visualization
        image_name: Name of the image
    """
    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(index_names)))
    ax.set_yticks(np.arange(len(index_names)))
    ax.set_xticklabels(index_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(index_names, fontsize=10)

    # Add correlation values
    for i in range(len(index_names)):
        for j in range(len(index_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)

    ax.set_title(f'Vegetation Indices Correlation Matrix: {image_name}',
                fontsize=14, fontweight='bold', pad=20)

    plt.colorbar(im, ax=ax, label='Correlation Coefficient')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_image(image_path, output_base_dir):
    """
    Process a single image: calculate all vegetation indices and create visualizations.

    Args:
        image_path: Path to input image
        output_base_dir: Base output directory

    Returns:
        Dictionary with statistics for this image
    """
    image_name = Path(image_path).stem
    print(f"\nProcessing: {image_name}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Error: Could not load image {image_path}")
        return None

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"  Image size: {image_rgb.shape[1]}x{image_rgb.shape[0]}")

    # Calculate all vegetation indices
    print("  Calculating vegetation indices...")
    indices = calculate_vegetation_indices(image_rgb)

    # Get index information
    info_dict = get_index_info()

    # Calculate statistics for each index
    print("  Calculating statistics...")
    all_stats = {'image_name': image_name}

    for index_name, data in indices.items():
        stats = calculate_index_statistics(data, index_name)

        for stat_name, value in stats.items():
            all_stats[f'{index_name}_{stat_name}'] = value

        print(f"    {index_name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")

    # Save individual index visualizations
    print("  Creating individual visualizations...")
    for index_name, data in indices.items():
        index_dir = Path(output_base_dir) / 'vegetation_indices' / index_name.lower()
        index_dir.mkdir(parents=True, exist_ok=True)

        vis_path = index_dir / f'{image_name}_{index_name}.png'
        create_index_visualization(data, index_name, info_dict[index_name],
                                  str(vis_path), image_name)

    # Create comparison panel
    print("  Creating comparison panel...")
    comparison_path = Path(output_base_dir) / 'comparison_panels' / f'{image_name}_all_indices.png'
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    create_comparison_panel(image_rgb, indices, str(comparison_path), image_name)

    # Apply Otsu thresholding
    print("  Applying Otsu thresholding...")
    masks, thresholds = apply_otsu_thresholding(indices)

    # Save threshold comparison
    threshold_path = Path(output_base_dir) / 'comparison_panels' / f'{image_name}_thresholding.png'
    create_thresholding_comparison(indices, masks, thresholds, str(threshold_path), image_name)

    # Calculate correlation matrix
    print("  Calculating correlations...")
    corr_matrix, index_names = calculate_correlation_matrix(indices)

    # Save correlation heatmap
    corr_path = Path(output_base_dir) / 'correlation_analysis' / f'{image_name}_correlation.png'
    corr_path.parent.mkdir(parents=True, exist_ok=True)
    create_correlation_heatmap(corr_matrix, index_names, str(corr_path), image_name)

    print(f"  ✓ Completed {image_name}")

    return all_stats


def generate_summary_report(all_stats, output_path):
    """
    Generate summary report analyzing vegetation indices across all images.

    Args:
        all_stats: List of statistics dicts from all images
        output_path: Path to save summary report
    """
    if not all_stats:
        print("No statistics to summarize")
        return

    index_names = ['RGBVI', 'GLI', 'VARI', 'NGRDI', 'NGBDI', 'ExG', 'ExR', 'ExB',
                   'ExGR', 'CIVE', 'VEG', 'COM', 'TGI', 'MGRVI', 'RGVBI']

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VEGETATION INDEX ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total images processed: {len(all_stats)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("AVERAGE STATISTICS ACROSS ALL IMAGES\n")
        f.write("-" * 80 + "\n\n")

        for index_name in index_names:
            f.write(f"\n{index_name}:\n")

            mean_vals = [s[f'{index_name}_mean'] for s in all_stats if not np.isnan(s[f'{index_name}_mean'])]
            std_vals = [s[f'{index_name}_std'] for s in all_stats if not np.isnan(s[f'{index_name}_std'])]
            min_vals = [s[f'{index_name}_min'] for s in all_stats if not np.isnan(s[f'{index_name}_min'])]
            max_vals = [s[f'{index_name}_max'] for s in all_stats if not np.isnan(s[f'{index_name}_max'])]

            if mean_vals:
                f.write(f"  Average mean: {np.mean(mean_vals):.4f} ± {np.std(mean_vals):.4f}\n")
                f.write(f"  Average std: {np.mean(std_vals):.4f}\n")
                f.write(f"  Range: [{np.mean(min_vals):.4f}, {np.mean(max_vals):.4f}]\n")

        # Recommendations for midrib detection
        f.write("\n" + "-" * 80 + "\n")
        f.write("RECOMMENDATIONS FOR MIDRIB DETECTION\n")
        f.write("-" * 80 + "\n\n")

        f.write("Based on typical vegetation index behavior:\n\n")

        f.write("1. INDICES LIKELY TO SHOW MIDRIB AS DIFFERENT VALUES:\n")
        f.write("   - ExG (Excess Green): Midrib typically has lower green excess\n")
        f.write("   - NGRDI: Green-red balance differs in midrib\n")
        f.write("   - GLI: Green leaf index may show midrib as less 'leafy'\n")
        f.write("   - VARI: Atmospheric resistant - good for consistent detection\n\n")

        f.write("2. INDICES TO EXPLORE:\n")
        f.write("   - COM (Combination): Weighted sum may amplify differences\n")
        f.write("   - ExGR: Difference between green and red excess\n")
        f.write("   - MGRVI: Modified green-red index\n\n")

        f.write("3. USAGE STRATEGY:\n")
        f.write("   - Combine brightness (LAB L or YCrCb Y) with vegetation indices\n")
        f.write("   - Use ExG or GLI to exclude highly green regions (lamina)\n")
        f.write("   - Midrib often shows lower vegetation index values\n")
        f.write("   - Test Otsu thresholding on individual indices\n")
        f.write("   - Consider ensemble: brightness HIGH + vegetation index LOW = midrib\n\n")

        f.write("4. NEXT STEPS:\n")
        f.write("   - Review visualizations to identify which indices show clear midrib\n")
        f.write("   - Check correlation matrix for redundant indices\n")
        f.write("   - Combine complementary indices for robust detection\n")

    print(f"\nSummary report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate and visualize vegetation indices for sorghum leaf images"
    )
    parser.add_argument(
        '--input-dir', '-i', type=str, default='data/test_input',
        help='Input directory containing images'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, default='data/test_output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--pattern', '-p', type=str, default='*.jpg',
        help='File pattern to match (default: *.jpg)'
    )

    args = parser.parse_args()

    # Find all images
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return

    image_files = list(input_dir.glob(args.pattern))
    if not image_files:
        print(f"Error: No images found matching '{args.pattern}' in '{input_dir}'")
        return

    print(f"Found {len(image_files)} images to process")

    # Create output directories
    output_base = Path(args.output_dir)
    subdirs = ['vegetation_indices', 'comparison_panels', 'statistics', 'correlation_analysis']
    for subdir in subdirs:
        (output_base / subdir).mkdir(parents=True, exist_ok=True)

    # Process all images
    all_stats = []
    for image_path in image_files:
        stats = process_image(str(image_path), str(output_base))
        if stats:
            all_stats.append(stats)

    # Save statistics to CSV
    if all_stats:
        csv_path = output_base / 'statistics' / 'vegetation_indices_stats.csv'
        print(f"\nSaving statistics to {csv_path}")

        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = all_stats[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_stats)

        print(f"✓ CSV saved with {len(all_stats)} rows")

        # Generate summary report
        summary_path = output_base / 'statistics' / 'vegetation_indices_summary.txt'
        generate_summary_report(all_stats, str(summary_path))

    print("\n" + "="*80)
    print(f"✅ ANALYSIS COMPLETE")
    print(f"✅ Processed {len(all_stats)}/{len(image_files)} images successfully")
    print(f"✅ Results saved to: {output_base}/")
    print("="*80)

    # Print quick insights
    if all_stats:
        print("\nQUICK INSIGHTS:")
        print("  Check the following for midrib detection:")
        print("  1. ExG - Look for low values (less green)")
        print("  2. NGRDI - Check green-red balance")
        print("  3. Correlation matrix - Identify unique indices")
        print(f"\n  View comparison panels: {output_base}/comparison_panels/")
        print(f"  View correlations: {output_base}/correlation_analysis/")


if __name__ == "__main__":
    main()
