"""
Prepare train/validation/test splits based on genotype.

CRITICAL: All images with the same genotype MUST be in the same split
to prevent data leakage.
"""

import os
import re
import json
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from autoencoder.config import Config


def find_all_images(config):
    """
    Find all usable images (with raw image + both masks + genotype).

    Returns:
        list: List of dicts with keys: device, plot, filename, genotype, image_path, mask_dir
    """
    # Load CSV
    df = pd.read_csv(config.csv_path)
    df.columns = df.columns.str.strip()
    plot_to_genotype = dict(zip(df['plot'], df['genotype']))

    usable_images = []

    for device in range(1, 9):
        # Use raw data directory (data/ne2025/device*)
        device_dir = config.data_dir / f'device{device}'
        if not device_dir.exists():
            continue

        for img_file in device_dir.iterdir():
            if not img_file.suffix in ['.jpg', '.JPG', '.jpeg', '.JPEG']:
                continue

            # Extract plot number from filename
            match = re.match(r'(\d+)_', img_file.name)
            if not match:
                continue

            plot = int(match.group(1))

            # Build expected mask directory name
            base_name = img_file.stem  # Filename without extension
            mask_dir = config.output_dir / 'stripSegmentation' / 'ne2025' / f'device{device}_{base_name}'

            # Check if both masks exist
            mask0_path = mask_dir / 'strip_0000_mask.png'
            mask1_path = mask_dir / 'strip_0001_mask.png'

            if not (mask0_path.exists() and mask1_path.exists()):
                continue

            # Check if genotype is available
            if plot not in plot_to_genotype:
                continue

            genotype = plot_to_genotype[plot]

            usable_images.append({
                'device': device,
                'plot': plot,
                'filename': img_file.name,
                'genotype': genotype,
                'image_path': str(img_file),
                'mask_dir': str(mask_dir),
                'mask0_path': str(mask0_path),
                'mask1_path': str(mask1_path)
            })

    return usable_images


def split_by_genotype(images, config):
    """
    Split images by genotype to prevent data leakage.

    Args:
        images: List of image metadata dicts
        config: Config object with split ratios

    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    # Set random seed for reproducibility
    random.seed(config.random_seed)

    # Group images by genotype
    genotype_to_images = defaultdict(list)
    for img in images:
        genotype_to_images[img['genotype']].append(img)

    print(f"Total genotypes: {len(genotype_to_images)}")

    # Count genotypes by number of images
    genotype_counts = Counter({g: len(imgs) for g, imgs in genotype_to_images.items()})
    print(f"\nGenotype distribution:")
    print(f"  Genotypes with 1 image: {sum(1 for c in genotype_counts.values() if c == 1)}")
    print(f"  Genotypes with 2 images: {sum(1 for c in genotype_counts.values() if c == 2)}")
    print(f"  Genotypes with 3-10 images: {sum(1 for c in genotype_counts.values() if 3 <= c <= 10)}")
    print(f"  Genotypes with >10 images: {sum(1 for c in genotype_counts.values() if c > 10)}")

    # Separate genotypes by size
    small_genotypes = []  # <3 images - assign all to train
    splittable_genotypes = []  # >=3 images - can split

    for genotype, imgs in genotype_to_images.items():
        if len(imgs) < 3:
            small_genotypes.append(genotype)
        else:
            splittable_genotypes.append(genotype)

    print(f"\nSplitting strategy:")
    print(f"  Small genotypes (<3 images): {len(small_genotypes)} -> assign to train")
    print(f"  Splittable genotypes (≥3 images): {len(splittable_genotypes)} -> split train/val/test")

    # Shuffle splittable genotypes
    random.shuffle(splittable_genotypes)

    # Calculate split indices
    n_splittable = len(splittable_genotypes)
    n_train = int(n_splittable * config.train_ratio)
    n_val = int(n_splittable * config.val_ratio)
    # n_test = remaining

    train_genotypes = set(splittable_genotypes[:n_train] + small_genotypes)
    val_genotypes = set(splittable_genotypes[n_train:n_train + n_val])
    test_genotypes = set(splittable_genotypes[n_train + n_val:])

    # Assign images to splits based on their genotype
    splits = {'train': [], 'val': [], 'test': []}

    for img in images:
        genotype = img['genotype']
        if genotype in train_genotypes:
            img['split'] = 'train'
            splits['train'].append(img)
        elif genotype in val_genotypes:
            img['split'] = 'val'
            splits['val'].append(img)
        elif genotype in test_genotypes:
            img['split'] = 'test'
            splits['test'].append(img)

    # Print split summary
    print(f"\n{'='*60}")
    print("Split Summary:")
    print(f"{'='*60}")
    for split_name in ['train', 'val', 'test']:
        imgs = splits[split_name]
        genotypes = set(img['genotype'] for img in imgs)
        print(f"\n{split_name.upper()}:")
        print(f"  Images: {len(imgs)}")
        print(f"  Genotypes: {len(genotypes)}")
        print(f"  Images per genotype: {len(imgs) / len(genotypes):.2f}")

        # Show genotype distribution
        split_genotype_counts = Counter(img['genotype'] for img in imgs)
        print(f"  Top 10 genotypes:")
        for genotype, count in split_genotype_counts.most_common(10):
            print(f"    {genotype:20} {count:4d} images")

    return splits, {
        'train_genotypes': sorted(list(train_genotypes)),
        'val_genotypes': sorted(list(val_genotypes)),
        'test_genotypes': sorted(list(test_genotypes))
    }


def save_splits(splits, genotype_splits, config):
    """
    Save split information to JSON files.

    Args:
        splits: Dict with train/val/test image lists
        genotype_splits: Dict with train/val/test genotype lists
        config: Config object
    """
    # Save image splits
    splits_path = config.logs_dir / 'image_splits.json'
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"\nImage splits saved to {splits_path}")

    # Save genotype splits
    genotype_splits_path = config.logs_dir / 'genotype_splits.json'
    with open(genotype_splits_path, 'w') as f:
        json.dump(genotype_splits, f, indent=2)
    print(f"Genotype splits saved to {genotype_splits_path}")

    # Save summary statistics
    summary_path = config.logs_dir / 'dataset_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Dataset Summary\n")
        f.write("=" * 60 + "\n\n")

        total_images = sum(len(splits[s]) for s in ['train', 'val', 'test'])
        f.write(f"Total images: {total_images}\n\n")

        for split_name in ['train', 'val', 'test']:
            imgs = splits[split_name]
            genotypes = set(img['genotype'] for img in imgs)
            f.write(f"{split_name.upper()}:\n")
            f.write(f"  Images: {len(imgs)} ({len(imgs)/total_images*100:.1f}%)\n")
            f.write(f"  Genotypes: {len(genotypes)}\n")
            f.write(f"  Avg images per genotype: {len(imgs) / len(genotypes):.2f}\n\n")

        f.write(f"\nGenotype assignment ensures no data leakage:\n")
        f.write(f"  - All images from the same genotype are in the same split\n")
        f.write(f"  - Small genotypes (<3 images) are assigned to training\n")
        f.write(f"  - Random seed: {config.random_seed}\n")

    print(f"Dataset summary saved to {summary_path}")


def main():
    """Main function to prepare splits."""
    config = Config()

    print("Preparing train/validation/test splits...")
    print("=" * 60)

    # Find all usable images
    print("\nFinding all usable images...")
    images = find_all_images(config)
    print(f"Found {len(images)} usable images")

    # Split by genotype
    print("\nSplitting by genotype...")
    splits, genotype_splits = split_by_genotype(images, config)

    # Save splits
    print("\nSaving splits...")
    save_splits(splits, genotype_splits, config)

    print("\n" + "=" * 60)
    print("Split preparation complete!")
    print("=" * 60)

    return splits, genotype_splits


if __name__ == '__main__':
    main()
