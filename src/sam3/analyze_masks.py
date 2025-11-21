"""
Analyze and visualize saved mask files
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_mask_data(npz_path):
    """Load mask data from npz file"""
    data = np.load(npz_path)
    return {
        'masks': data['masks'],
        'boxes': data['boxes'],
        'scores': data['scores']
    }


def analyze_masks(npz_path, show_details=True):
    """
    Analyze mask file and print statistics

    Args:
        npz_path: Path to .npz mask file
        show_details: Print detailed info for each mask
    """
    data = load_mask_data(npz_path)
    masks = data['masks']
    boxes = data['boxes']
    scores = data['scores']

    print(f"\n{'='*60}")
    print(f"Analysis: {Path(npz_path).name}")
    print(f"{'='*60}")

    print(f"\nNumber of objects detected: {len(masks)}")

    if len(masks) == 0:
        print("No objects to analyze")
        return

    print(f"\nScores:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Min:  {scores.min():.4f}")
    print(f"  Max:  {scores.max():.4f}")

    # Calculate mask areas
    areas = [(mask > 0).sum() for mask in masks]
    total_pixels = masks[0].size

    print(f"\nMask Areas (pixels):")
    print(f"  Mean: {np.mean(areas):.0f}")
    print(f"  Min:  {np.min(areas):.0f}")
    print(f"  Max:  {np.max(areas):.0f}")

    print(f"\nMask Areas (% of image):")
    percentages = [100 * area / total_pixels for area in areas]
    print(f"  Mean: {np.mean(percentages):.2f}%")
    print(f"  Min:  {np.min(percentages):.2f}%")
    print(f"  Max:  {np.max(percentages):.2f}%")

    if show_details:
        print(f"\nDetailed breakdown:")
        print(f"{'ID':<4} {'Score':<8} {'Area (px)':<12} {'Area (%)':<10} {'Box (x1,y1,x2,y2)'}")
        print("-" * 70)
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            area = (mask > 0).sum()
            pct = 100 * area / total_pixels
            box_str = f"({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})"
            print(f"{i:<4} {score:<8.4f} {area:<12} {pct:<10.2f} {box_str}")


def visualize_masks(npz_path, output_path=None):
    """
    Create visualization of individual masks

    Args:
        npz_path: Path to .npz mask file
        output_path: Where to save visualization (optional)
    """
    data = load_mask_data(npz_path)
    masks = data['masks']
    scores = data['scores']

    n_masks = len(masks)
    if n_masks == 0:
        print("No masks to visualize")
        return

    # Create grid layout
    cols = min(3, n_masks)
    rows = (n_masks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if n_masks == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(n_masks):
        axes[i].imshow(masks[i], cmap='gray')
        axes[i].set_title(f"Mask {i}\nScore: {scores[i]:.3f}")
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(n_masks, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()


def compare_masks(npz_paths, output_path=None):
    """
    Compare mask counts across multiple files

    Args:
        npz_paths: List of .npz mask file paths
        output_path: Where to save comparison plot (optional)
    """
    names = []
    counts = []
    mean_scores = []

    for path in npz_paths:
        data = load_mask_data(path)
        name = Path(path).stem.replace('masks_', '')
        names.append(name[:30])  # Truncate long names
        counts.append(len(data['masks']))
        if len(data['masks']) > 0:
            mean_scores.append(data['scores'].mean())
        else:
            mean_scores.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot counts
    ax1.bar(range(len(names)), counts, color='steelblue')
    ax1.set_xlabel('Image')
    ax1.set_ylabel('Number of Objects Detected')
    ax1.set_title('Object Detection Counts')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Plot scores
    ax2.bar(range(len(names)), mean_scores, color='coral')
    ax2.set_xlabel('Image')
    ax2.set_ylabel('Mean Confidence Score')
    ax2.set_title('Mean Detection Scores')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze SAM3 mask files")
    parser.add_argument("masks", nargs="+", help="Path(s) to .npz mask files or directory")
    parser.add_argument("--visualize", "-v", action="store_true",
                       help="Create visualization of individual masks")
    parser.add_argument("--compare", "-c", action="store_true",
                       help="Compare multiple mask files")
    parser.add_argument("--output", "-o", help="Output path for visualizations")
    parser.add_argument("--no-details", action="store_true",
                       help="Don't print detailed per-mask info")

    args = parser.parse_args()

    # Collect all mask files
    mask_files = []
    for path_str in args.masks:
        path = Path(path_str)
        if path.is_file() and path.suffix == '.npz':
            mask_files.append(path)
        elif path.is_dir():
            mask_files.extend(path.glob("masks_*.npz"))

    if not mask_files:
        print("No mask files found")
        return 1

    print(f"Found {len(mask_files)} mask file(s)")

    # Analyze each file
    for mask_file in sorted(mask_files):
        analyze_masks(mask_file, show_details=not args.no_details)

    # Visualize if requested
    if args.visualize:
        for mask_file in mask_files:
            output = None
            if args.output:
                output = Path(args.output) / f"viz_{mask_file.stem}.png"
            visualize_masks(mask_file, output)

    # Compare if requested and multiple files
    if args.compare and len(mask_files) > 1:
        compare_masks(mask_files, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
