#!/usr/bin/env python3
"""
Compare DINOv2 and SAM2.1 embeddings.

This script compares the feature spaces of both models to help you
understand their different characteristics and choose the right one
for your task.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os


def load_embeddings(dinov2_csv='dinov2_features.csv', sam2_csv='src/sam2/sam2_image_embeddings.csv'):
    """Load both DINOv2 and SAM2 embeddings."""
    print("📂 Loading embeddings...")

    results = {}

    # Load DINOv2
    if os.path.exists(dinov2_csv):
        df_dinov2 = pd.read_csv(dinov2_csv)
        results['dinov2_paths'] = df_dinov2['image_path'].values
        results['dinov2_features'] = df_dinov2.iloc[:, 1:].values
        print(f"   ✅ DINOv2: {results['dinov2_features'].shape}")
    else:
        print(f"   ⚠️  DINOv2 not found: {dinov2_csv}")
        results['dinov2_features'] = None

    # Load SAM2
    if os.path.exists(sam2_csv):
        df_sam2 = pd.read_csv(sam2_csv)
        results['sam2_paths'] = df_sam2['image_path'].values
        results['sam2_features'] = df_sam2.iloc[:, 1:].values
        print(f"   ✅ SAM2: {results['sam2_features'].shape}")
    else:
        print(f"   ⚠️  SAM2 not found: {sam2_csv}")
        results['sam2_features'] = None

    return results


def compare_statistics(embeddings):
    """Compare statistical properties of both embeddings."""
    print("\n" + "="*70)
    print("📊 FEATURE STATISTICS COMPARISON")
    print("="*70)

    stats = []

    for name in ['dinov2', 'sam2']:
        features = embeddings.get(f'{name}_features')
        if features is None:
            continue

        stat = {
            'Model': name.upper(),
            'Dimensions': features.shape[1],
            'Images': features.shape[0],
            'Mean': features.mean(),
            'Std': features.std(),
            'Min': features.min(),
            'Max': features.max(),
            'Median': np.median(features),
            'Sparsity': (features == 0).sum() / features.size * 100
        }
        stats.append(stat)

    # Print comparison table
    if len(stats) == 2:
        print(f"\n{'Metric':<15} {'DINOv2':<15} {'SAM2':<15} {'Difference':<15}")
        print("-" * 65)

        for key in ['Dimensions', 'Images', 'Mean', 'Std', 'Min', 'Max', 'Median', 'Sparsity']:
            dinov2_val = stats[0][key] if key in stats[0] else 0
            sam2_val = stats[1][key] if key in stats[1] else 0

            if isinstance(dinov2_val, (int, np.integer)):
                print(f"{key:<15} {dinov2_val:<15} {sam2_val:<15} {dinov2_val - sam2_val:<15}")
            else:
                diff = dinov2_val - sam2_val
                print(f"{key:<15} {dinov2_val:<15.4f} {sam2_val:<15.4f} {diff:<15.4f}")
    else:
        # Print available stats
        for stat in stats:
            print(f"\n{stat['Model']}:")
            for key, val in stat.items():
                if key != 'Model':
                    if isinstance(val, (int, np.integer)):
                        print(f"  {key}: {val}")
                    else:
                        print(f"  {key}: {val:.4f}")


def compare_similarity_distributions(embeddings):
    """Compare similarity distributions between models."""
    print("\n" + "="*70)
    print("🔍 SIMILARITY DISTRIBUTION COMPARISON")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Pairwise Cosine Similarity Distributions', fontsize=14, fontweight='bold')

    for idx, name in enumerate(['dinov2', 'sam2']):
        features = embeddings.get(f'{name}_features')
        if features is None:
            continue

        # Sample 100 random images for efficiency
        n_samples = min(100, features.shape[0])
        sample_indices = np.random.choice(features.shape[0], n_samples, replace=False)
        sample_features = features[sample_indices]

        # Compute pairwise similarities
        similarities = cosine_similarity(sample_features)

        # Get upper triangle (exclude diagonal)
        triu_indices = np.triu_indices_from(similarities, k=1)
        similarity_values = similarities[triu_indices]

        # Plot histogram
        axes[idx].hist(similarity_values, bins=50, edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel('Cosine Similarity')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{name.upper()} (n={n_samples} images)')
        axes[idx].grid(True, alpha=0.3)

        # Print statistics
        print(f"\n{name.upper()} Pairwise Similarities (sample of {n_samples} images):")
        print(f"  Mean: {similarity_values.mean():.4f}")
        print(f"  Std: {similarity_values.std():.4f}")
        print(f"  Min: {similarity_values.min():.4f}")
        print(f"  Max: {similarity_values.max():.4f}")
        print(f"  Median: {np.median(similarity_values):.4f}")

    plt.tight_layout()
    plt.savefig('embedding_similarity_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved similarity comparison to: embedding_similarity_comparison.png")
    plt.close()


def compare_feature_distributions(embeddings):
    """Compare feature value distributions."""
    print("\n" + "="*70)
    print("📈 FEATURE VALUE DISTRIBUTION COMPARISON")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Feature Value Distributions', fontsize=14, fontweight='bold')

    for idx, name in enumerate(['dinov2', 'sam2']):
        features = embeddings.get(f'{name}_features')
        if features is None:
            continue

        # Flatten all features
        flat_features = features.flatten()

        # Plot histogram
        axes[idx].hist(flat_features, bins=100, edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel('Feature Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{name.upper()}')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_yscale('log')  # Log scale for better visualization

    plt.tight_layout()
    plt.savefig('embedding_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved distribution comparison to: embedding_distribution_comparison.png")
    plt.close()


def compare_alignment(embeddings):
    """Check if images align between the two embedding sets."""
    print("\n" + "="*70)
    print("🔗 IMAGE ALIGNMENT CHECK")
    print("="*70)

    dinov2_paths = embeddings.get('dinov2_paths')
    sam2_paths = embeddings.get('sam2_paths')

    if dinov2_paths is None or sam2_paths is None:
        print("⚠️  Cannot compare - one or both embedding sets missing")
        return

    # Find common images
    dinov2_set = set(dinov2_paths)
    sam2_set = set(sam2_paths)

    common = dinov2_set & sam2_set
    only_dinov2 = dinov2_set - sam2_set
    only_sam2 = sam2_set - dinov2_set

    print(f"DINOv2 images: {len(dinov2_paths)}")
    print(f"SAM2 images: {len(sam2_paths)}")
    print(f"Common images: {len(common)}")
    print(f"Only in DINOv2: {len(only_dinov2)}")
    print(f"Only in SAM2: {len(only_sam2)}")

    if len(common) > 0:
        overlap_pct = len(common) / max(len(dinov2_paths), len(sam2_paths)) * 100
        print(f"Overlap: {overlap_pct:.2f}%")

        if overlap_pct < 100:
            print("\n⚠️  Warning: Image sets don't fully overlap!")
            print("   This may affect direct comparisons.")
    else:
        print("\n❌ Error: No common images found!")
        print("   The two embedding sets are for completely different images.")


def main():
    """Main comparison."""
    print("="*70)
    print("🔬 DINOv2 vs SAM2.1 Embedding Comparison")
    print("="*70)

    # Load embeddings
    embeddings = load_embeddings()

    if embeddings['dinov2_features'] is None and embeddings['sam2_features'] is None:
        print("\n❌ Error: No embeddings found!")
        print("\nPlease generate embeddings first:")
        print("  1. DINOv2: python extract_dinov2_features.py")
        print("  2. SAM2: python src/sam2/extract_embeddings.py")
        return

    # Perform comparisons
    compare_alignment(embeddings)
    compare_statistics(embeddings)

    if embeddings['dinov2_features'] is not None and embeddings['sam2_features'] is not None:
        compare_similarity_distributions(embeddings)
        compare_feature_distributions(embeddings)

    print("\n" + "="*70)
    print("✅ Comparison complete!")
    print("="*70)

    print("\nGenerated files:")
    if os.path.exists('embedding_similarity_comparison.png'):
        print("  - embedding_similarity_comparison.png")
    if os.path.exists('embedding_distribution_comparison.png'):
        print("  - embedding_distribution_comparison.png")

    print("\n💡 Interpretation:")
    print("  - Higher similarity values = more discriminative features")
    print("  - More spread in distribution = richer feature representation")
    print("  - DINOv2: Optimized for classification/retrieval tasks")
    print("  - SAM2: Optimized for segmentation tasks")


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    main()
