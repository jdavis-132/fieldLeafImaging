#!/usr/bin/env python3
"""
Brightness Channel Analysis for Sorghum Leaf Midrib Detection

Generates grayscale brightness maps from different color spaces (HSV, LAB, YCrCb)
to understand which provides the best separation between midrib and leaf lamina.
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
import csv

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


def extract_brightness_channels(image):
    """
    Extract brightness channels from different color spaces.

    Args:
        image: BGR image from cv2.imread

    Returns:
        dict with brightness channels and statistics
    """
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Extract brightness channels
    hsv_v = hsv[:, :, 2]  # Value channel (0-255)
    lab_l = lab[:, :, 0]  # Lightness channel (0-255 in OpenCV, represents 0-100)
    ycrcb_y = ycrcb[:, :, 0]  # Luminance channel (0-255)

    # Calculate statistics for each channel
    channels = {
        'hsv_v': {
            'data': hsv_v,
            'mean': np.mean(hsv_v),
            'std': np.std(hsv_v),
            'min': np.min(hsv_v),
            'max': np.max(hsv_v),
            'median': np.median(hsv_v),
        },
        'lab_l': {
            'data': lab_l,
            'mean': np.mean(lab_l),
            'std': np.std(lab_l),
            'min': np.min(lab_l),
            'max': np.max(lab_l),
            'median': np.median(lab_l),
        },
        'ycrcb_y': {
            'data': ycrcb_y,
            'mean': np.mean(ycrcb_y),
            'std': np.std(ycrcb_y),
            'min': np.min(ycrcb_y),
            'max': np.max(ycrcb_y),
            'median': np.median(ycrcb_y),
        }
    }

    return channels


def apply_thresholding(channel_data, method='otsu'):
    """
    Apply different thresholding methods to a brightness channel.

    Args:
        channel_data: 2D numpy array (grayscale image)
        method: 'otsu', 'adaptive_mean', or 'adaptive_gaussian'

    Returns:
        Binary mask and threshold value
    """
    if method == 'otsu':
        threshold, binary = cv2.threshold(channel_data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary, threshold

    elif method == 'adaptive_mean':
        binary = cv2.adaptiveThreshold(channel_data, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 21, 5)
        return binary, None

    elif method == 'adaptive_gaussian':
        binary = cv2.adaptiveThreshold(channel_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 5)
        return binary, None

    else:
        raise ValueError(f"Unknown thresholding method: {method}")


def calculate_contrast_ratio(channel_data, percentile=95):
    """
    Calculate contrast ratio between bright and dark regions.

    Args:
        channel_data: 2D numpy array
        percentile: Percentile to use for "bright" regions (default 95th)

    Returns:
        Contrast ratio
    """
    bright_threshold = np.percentile(channel_data, percentile)
    dark_threshold = np.percentile(channel_data, 100 - percentile)

    bright_mean = np.mean(channel_data[channel_data >= bright_threshold])
    dark_mean = np.mean(channel_data[channel_data <= dark_threshold])

    if dark_mean > 0:
        contrast_ratio = bright_mean / dark_mean
    else:
        contrast_ratio = np.inf

    return contrast_ratio, bright_mean, dark_mean


def create_brightness_comparison(image, channels, output_path, image_name):
    """
    Create comprehensive visualization comparing brightness channels.

    Args:
        image: Original BGR image
        channels: Dict of brightness channels from extract_brightness_channels
        output_path: Path to save visualization
        image_name: Name of the image for title
    """
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Row 1: Original image and brightness maps
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(channels['hsv_v']['data'], cmap='gray', vmin=0, vmax=255)
    ax2.set_title(f"HSV V-Channel\n(μ={channels['hsv_v']['mean']:.1f}, σ={channels['hsv_v']['std']:.1f})",
                 fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(channels['lab_l']['data'], cmap='gray', vmin=0, vmax=255)
    ax3.set_title(f"LAB L-Channel\n(μ={channels['lab_l']['mean']:.1f}, σ={channels['lab_l']['std']:.1f})",
                 fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Row 2: YCrCb and color-mapped versions
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(channels['ycrcb_y']['data'], cmap='gray', vmin=0, vmax=255)
    ax4.set_title(f"YCrCb Y-Channel\n(μ={channels['ycrcb_y']['mean']:.1f}, σ={channels['ycrcb_y']['std']:.1f})",
                 fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(channels['hsv_v']['data'], cmap='viridis')
    ax5.set_title('HSV V (Viridis colormap)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(channels['lab_l']['data'], cmap='viridis')
    ax6.set_title('LAB L (Viridis colormap)', fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)

    # Row 3: Histograms
    ax7 = fig.add_subplot(gs[2, :])
    ax7.hist(channels['hsv_v']['data'].ravel(), bins=256, range=(0, 256),
            alpha=0.5, label='HSV V', color='blue', density=True)
    ax7.hist(channels['lab_l']['data'].ravel(), bins=256, range=(0, 256),
            alpha=0.5, label='LAB L', color='red', density=True)
    ax7.hist(channels['ycrcb_y']['data'].ravel(), bins=256, range=(0, 256),
            alpha=0.5, label='YCrCb Y', color='green', density=True)
    ax7.set_xlabel('Brightness Value', fontsize=11)
    ax7.set_ylabel('Density', fontsize=11)
    ax7.set_title('Brightness Distribution Comparison', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)

    # Row 4: Thresholded versions (Otsu)
    ax8 = fig.add_subplot(gs[3, 0])
    otsu_v, thresh_v = apply_thresholding(channels['hsv_v']['data'], 'otsu')
    ax8.imshow(otsu_v, cmap='gray')
    ax8.set_title(f'HSV V (Otsu threshold={thresh_v:.1f})', fontsize=12, fontweight='bold')
    ax8.axis('off')

    ax9 = fig.add_subplot(gs[3, 1])
    otsu_l, thresh_l = apply_thresholding(channels['lab_l']['data'], 'otsu')
    ax9.imshow(otsu_l, cmap='gray')
    ax9.set_title(f'LAB L (Otsu threshold={thresh_l:.1f})', fontsize=12, fontweight='bold')
    ax9.axis('off')

    ax10 = fig.add_subplot(gs[3, 2])
    otsu_y, thresh_y = apply_thresholding(channels['ycrcb_y']['data'], 'otsu')
    ax10.imshow(otsu_y, cmap='gray')
    ax10.set_title(f'YCrCb Y (Otsu threshold={thresh_y:.1f})', fontsize=12, fontweight='bold')
    ax10.axis('off')

    # Add overall title
    fig.suptitle(f'Brightness Channel Analysis: {image_name}', fontsize=14, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_line_profile(image, channels, output_path, image_name):
    """
    Create line profile across the leaf width showing brightness transitions.

    Args:
        image: Original BGR image
        channels: Dict of brightness channels
        output_path: Path to save visualization
        image_name: Name of the image for title
    """
    h, w = image.shape[:2]

    # Sample horizontal line profiles at different y-positions
    y_positions = [h // 4, h // 2, 3 * h // 4]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Horizontal Brightness Profiles: {image_name}', fontsize=14, fontweight='bold')

    for idx, y in enumerate(y_positions):
        ax = axes[idx]

        # Extract line profiles
        hsv_profile = channels['hsv_v']['data'][y, :]
        lab_profile = channels['lab_l']['data'][y, :]
        ycrcb_profile = channels['ycrcb_y']['data'][y, :]

        # Plot
        x = np.arange(w)
        ax.plot(x, hsv_profile, label='HSV V', alpha=0.7, linewidth=2)
        ax.plot(x, lab_profile, label='LAB L', alpha=0.7, linewidth=2)
        ax.plot(x, ycrcb_profile, label='YCrCb Y', alpha=0.7, linewidth=2)

        ax.set_xlabel('X Position (pixels)', fontsize=10)
        ax.set_ylabel('Brightness Value', fontsize=10)
        ax.set_title(f'Profile at y={y} (row {idx+1}/3)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 255)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_edge_detection(channels, output_path, image_name):
    """
    Apply edge detection to each brightness channel to see which highlights midrib best.

    Args:
        channels: Dict of brightness channels
        output_path: Path to save visualization
        image_name: Name of the image for title
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Edge Detection Comparison: {image_name}', fontsize=14, fontweight='bold')

    channel_names = ['hsv_v', 'lab_l', 'ycrcb_y']
    titles = ['HSV V', 'LAB L', 'YCrCb Y']

    for idx, (name, title) in enumerate(zip(channel_names, titles)):
        data = channels[name]['data']

        # Sobel edge detection
        sobelx = cv2.Sobel(data, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(data, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.clip(sobel, 0, 255).astype(np.uint8)

        # Canny edge detection
        canny = cv2.Canny(data, 50, 150)

        # Plot
        axes[0, idx].imshow(sobel, cmap='gray')
        axes[0, idx].set_title(f'{title} - Sobel Edges', fontsize=11, fontweight='bold')
        axes[0, idx].axis('off')

        axes[1, idx].imshow(canny, cmap='gray')
        axes[1, idx].set_title(f'{title} - Canny Edges', fontsize=11, fontweight='bold')
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_image(image_path, output_base_dir):
    """
    Process a single image: extract brightness channels, create visualizations, return statistics.

    Args:
        image_path: Path to input image
        output_base_dir: Base output directory

    Returns:
        Dict with statistics for this image
    """
    image_name = Path(image_path).stem
    print(f"\nProcessing: {image_name}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Error: Could not load image {image_path}")
        return None

    print(f"  Image size: {image.shape[1]}x{image.shape[0]}")

    # Extract brightness channels
    print("  Extracting brightness channels...")
    channels = extract_brightness_channels(image)

    # Save individual grayscale images
    print("  Saving individual channel images...")
    cv2.imwrite(os.path.join(output_base_dir, 'brightness_maps', 'hsv_v', f'{image_name}_hsv_v.png'),
                channels['hsv_v']['data'])
    cv2.imwrite(os.path.join(output_base_dir, 'brightness_maps', 'lab_l', f'{image_name}_lab_l.png'),
                channels['lab_l']['data'])
    cv2.imwrite(os.path.join(output_base_dir, 'brightness_maps', 'ycrcb_y', f'{image_name}_ycrcb_y.png'),
                channels['ycrcb_y']['data'])

    # Calculate contrast ratios
    print("  Calculating contrast ratios...")
    for channel_name in ['hsv_v', 'lab_l', 'ycrcb_y']:
        contrast, bright, dark = calculate_contrast_ratio(channels[channel_name]['data'])
        channels[channel_name]['contrast_ratio'] = contrast
        channels[channel_name]['bright_95th'] = bright
        channels[channel_name]['dark_5th'] = dark
        print(f"    {channel_name}: contrast={contrast:.2f}, bright={bright:.1f}, dark={dark:.1f}")

    # Create comparison visualization
    print("  Creating brightness comparison...")
    comparison_path = os.path.join(output_base_dir, 'comparisons', f'{image_name}_brightness_comparison.png')
    create_brightness_comparison(image, channels, comparison_path, image_name)

    # Create line profiles
    print("  Creating line profiles...")
    profile_path = os.path.join(output_base_dir, 'histograms', f'{image_name}_line_profiles.png')
    create_line_profile(image, channels, profile_path, image_name)

    # Create edge detection comparison
    print("  Analyzing edge detection...")
    edge_path = os.path.join(output_base_dir, 'comparisons', f'{image_name}_edge_detection.png')
    analyze_edge_detection(channels, edge_path, image_name)

    # Compile statistics for CSV
    stats = {
        'image_name': image_name,
        'image_width': image.shape[1],
        'image_height': image.shape[0],
    }

    for channel_name in ['hsv_v', 'lab_l', 'ycrcb_y']:
        ch = channels[channel_name]
        stats[f'{channel_name}_mean'] = ch['mean']
        stats[f'{channel_name}_std'] = ch['std']
        stats[f'{channel_name}_min'] = ch['min']
        stats[f'{channel_name}_max'] = ch['max']
        stats[f'{channel_name}_median'] = ch['median']
        stats[f'{channel_name}_contrast_ratio'] = ch['contrast_ratio']
        stats[f'{channel_name}_bright_95th'] = ch['bright_95th']
        stats[f'{channel_name}_dark_5th'] = ch['dark_5th']

    print(f"  ✓ Completed {image_name}")

    return stats


def generate_summary_report(all_stats, output_path):
    """
    Generate a summary report analyzing all processed images.

    Args:
        all_stats: List of statistics dicts from all images
        output_path: Path to save summary report
    """
    if not all_stats:
        print("No statistics to summarize")
        return

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BRIGHTNESS CHANNEL ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total images processed: {len(all_stats)}\n\n")

        # Calculate averages across all images
        f.write("-" * 80 + "\n")
        f.write("AVERAGE STATISTICS ACROSS ALL IMAGES\n")
        f.write("-" * 80 + "\n\n")

        for channel in ['hsv_v', 'lab_l', 'ycrcb_y']:
            f.write(f"\n{channel.upper()} Channel:\n")

            mean_vals = [s[f'{channel}_mean'] for s in all_stats]
            std_vals = [s[f'{channel}_std'] for s in all_stats]
            contrast_vals = [s[f'{channel}_contrast_ratio'] for s in all_stats]

            f.write(f"  Average mean brightness: {np.mean(mean_vals):.2f} ± {np.std(mean_vals):.2f}\n")
            f.write(f"  Average std deviation: {np.mean(std_vals):.2f} ± {np.std(std_vals):.2f}\n")
            f.write(f"  Average contrast ratio: {np.mean(contrast_vals):.2f} ± {np.std(contrast_vals):.2f}\n")
            f.write(f"  Range: [{np.mean([s[f'{channel}_min'] for s in all_stats]):.1f}, "
                   f"{np.mean([s[f'{channel}_max'] for s in all_stats]):.1f}]\n")

        # Recommendations
        f.write("\n" + "-" * 80 + "\n")
        f.write("RECOMMENDATIONS FOR MIDRIB DETECTION\n")
        f.write("-" * 80 + "\n\n")

        # Find best channel based on contrast ratio
        avg_contrasts = {
            'HSV V': np.mean([s['hsv_v_contrast_ratio'] for s in all_stats]),
            'LAB L': np.mean([s['lab_l_contrast_ratio'] for s in all_stats]),
            'YCrCb Y': np.mean([s['ycrcb_y_contrast_ratio'] for s in all_stats]),
        }

        best_channel = max(avg_contrasts, key=avg_contrasts.get)

        f.write(f"1. BEST COLOR SPACE: {best_channel}\n")
        f.write(f"   - Highest average contrast ratio: {avg_contrasts[best_channel]:.2f}\n\n")

        # Threshold recommendations
        channel_key = best_channel.lower().replace(' ', '_')
        avg_bright = np.mean([s[f'{channel_key}_bright_95th'] for s in all_stats])
        avg_mean = np.mean([s[f'{channel_key}_mean'] for s in all_stats])
        avg_std = np.mean([s[f'{channel_key}_std'] for s in all_stats])

        f.write(f"2. SUGGESTED THRESHOLDS FOR {best_channel}:\n")
        f.write(f"   - Conservative (high precision): > {avg_bright:.1f}\n")
        f.write(f"   - Balanced: > {avg_mean + 1.5*avg_std:.1f} (mean + 1.5*std)\n")
        f.write(f"   - Sensitive (high recall): > {avg_mean + avg_std:.1f} (mean + 1.0*std)\n\n")

        f.write("3. COLOR SPACE COMPARISON:\n")
        for name, contrast in sorted(avg_contrasts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"   - {name}: contrast ratio = {contrast:.2f}\n")

        f.write("\n4. PROCESSING RECOMMENDATIONS:\n")
        f.write(f"   - Use {best_channel} as primary brightness channel\n")
        f.write("   - Apply morphological closing to connect midrib fragments\n")
        f.write("   - Use shape constraints (high aspect ratio, centered position)\n")
        f.write("   - Consider combining with color-based filtering (low saturation)\n")

    print(f"\nSummary report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze brightness channels in sorghum leaf images for midrib detection"
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
    for subdir in ['brightness_maps/hsv_v', 'brightness_maps/lab_l', 'brightness_maps/ycrcb_y',
                   'comparisons', 'histograms']:
        (output_base / subdir).mkdir(parents=True, exist_ok=True)

    # Process all images
    all_stats = []
    for image_path in image_files:
        stats = process_image(str(image_path), str(output_base))
        if stats:
            all_stats.append(stats)

    # Save statistics to CSV
    if all_stats:
        csv_path = output_base / 'brightness_statistics.csv'
        print(f"\nSaving statistics to {csv_path}")

        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = all_stats[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_stats)

        print(f"✓ CSV saved with {len(all_stats)} rows")

        # Generate summary report
        summary_path = output_base / 'brightness_analysis_summary.txt'
        generate_summary_report(all_stats, str(summary_path))

    print("\n" + "="*80)
    print(f"✅ ANALYSIS COMPLETE")
    print(f"✅ Processed {len(all_stats)}/{len(image_files)} images successfully")
    print(f"✅ Results saved to: {output_base}/")
    print("="*80)

    # Print quick summary
    if all_stats:
        print("\nQUICK SUMMARY:")
        avg_contrasts = {
            'HSV V': np.mean([s['hsv_v_contrast_ratio'] for s in all_stats]),
            'LAB L': np.mean([s['lab_l_contrast_ratio'] for s in all_stats]),
            'YCrCb Y': np.mean([s['ycrcb_y_contrast_ratio'] for s in all_stats]),
        }
        best = max(avg_contrasts, key=avg_contrasts.get)
        print(f"  Best color space: {best} (contrast ratio: {avg_contrasts[best]:.2f})")
        print(f"  All contrast ratios: {avg_contrasts}")


if __name__ == "__main__":
    main()
