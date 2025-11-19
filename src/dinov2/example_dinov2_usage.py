#!/usr/bin/env python3
"""
Example usage of DINOv2 features for downstream tasks.

This script demonstrates:
1. Loading extracted features
2. Computing image similarities
3. Finding nearest neighbors
4. Clustering images
5. Dimensionality reduction visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import os


def load_features(csv_path='dinov2_features.csv'):
    """Load DINOv2 features from CSV."""
    print(f"📂 Loading features from {csv_path}")
    df = pd.read_csv(csv_path)

    image_paths = df['image_path'].values
    features = df.iloc[:, 1:].values

    print(f"✅ Loaded {len(features)} features")
    print(f"   Feature shape: {features.shape}")
    print(f"   Feature dimension: {features.shape[1]}")

    return image_paths, features


def compute_similarities(features, query_idx=0, top_k=5):
    """Find most similar images to a query image."""
    print(f"\n🔍 Finding {top_k} most similar images to image {query_idx}")

    # Compute cosine similarities
    query_feature = features[query_idx:query_idx+1]
    similarities = cosine_similarity(query_feature, features)[0]

    # Get top-k indices (excluding query itself)
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]

    print(f"   Query image similarity (self): {similarities[query_idx]:.4f}")
    print(f"\n   Top {top_k} similar images:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"   {rank}. Image {idx}: similarity = {similarities[idx]:.4f}")

    return similarities, top_indices


def cluster_images(features, n_clusters=10, random_state=42):
    """Cluster images using K-Means."""
    print(f"\n📊 Clustering images into {n_clusters} groups")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(features)

    # Print cluster statistics
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"   Cluster sizes:")
    for cluster_id, count in zip(unique, counts):
        print(f"   Cluster {cluster_id}: {count} images")

    return cluster_labels, kmeans


def visualize_clusters(cluster_labels, n_clusters=10, save_path='cluster_distribution.png'):
    """Visualize cluster distribution."""
    print(f"\n📈 Creating cluster distribution plot")

    plt.figure(figsize=(10, 6))
    plt.hist(cluster_labels, bins=n_clusters, edgecolor='black', alpha=0.7)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Images')
    plt.title(f'Image Cluster Distribution ({n_clusters} clusters)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved to {save_path}")
    plt.close()


def compute_feature_statistics(features):
    """Compute and print feature statistics."""
    print(f"\n📊 Feature Statistics")
    print(f"   Shape: {features.shape}")
    print(f"   Mean: {features.mean():.4f}")
    print(f"   Std: {features.std():.4f}")
    print(f"   Min: {features.min():.4f}")
    print(f"   Max: {features.max():.4f}")
    print(f"   Median: {np.median(features):.4f}")

    # Per-dimension statistics
    dim_means = features.mean(axis=0)
    dim_stds = features.std(axis=0)
    print(f"\n   Per-dimension statistics:")
    print(f"   Mean of dimension means: {dim_means.mean():.4f}")
    print(f"   Mean of dimension stds: {dim_stds.mean():.4f}")
    print(f"   Most variable dimension: {np.argmax(dim_stds)} (std={dim_stds.max():.4f})")
    print(f"   Least variable dimension: {np.argmin(dim_stds)} (std={dim_stds.min():.4f})")


def visualize_umap_2d(features, save_path='dinov2_umap_2d.png', n_neighbors=15):
    """Reduce features to 2D using UMAP and visualize."""
    try:
        from umap import UMAP

        print(f"\n🗺️  Creating UMAP 2D visualization")
        print(f"   Computing UMAP projection (this may take a minute)...")

        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        features_2d = reducer.fit_transform(features)

        print(f"   Creating plot...")
        plt.figure(figsize=(12, 10))
        plt.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            alpha=0.5,
            s=1,
            c='blue',
            edgecolors='none'
        )
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title('DINOv2 Feature Space (UMAP 2D Projection)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved to {save_path}")
        plt.close()

        return features_2d

    except ImportError:
        print("   ⚠️  UMAP not installed. Install with: pip install umap-learn")
        return None


def save_cluster_assignments(image_paths, cluster_labels, output_csv='image_clusters.csv'):
    """Save cluster assignments to CSV."""
    print(f"\n💾 Saving cluster assignments to {output_csv}")

    df = pd.DataFrame({
        'image_path': image_paths,
        'cluster_id': cluster_labels
    })
    df.to_csv(output_csv, index=False)

    print(f"   ✅ Saved {len(df)} image-cluster assignments")


def main():
    """Main execution."""
    print("="*70)
    print("🎯 DINOv2 Feature Usage Examples")
    print("="*70)

    # Check if features file exists
    features_csv = 'dinov2_features.csv'
    if not os.path.exists(features_csv):
        features_csv = 'test_dinov2_features.csv'
        if not os.path.exists(features_csv):
            print(f"❌ Error: No features file found!")
            print(f"   Please run: python extract_dinov2_features.py")
            return

    # Load features
    image_paths, features = load_features(features_csv)

    # 1. Feature statistics
    compute_feature_statistics(features)

    # 2. Compute similarities for first image
    similarities, top_indices = compute_similarities(features, query_idx=0, top_k=5)

    # 3. Cluster images
    n_clusters = min(10, len(features) // 10)  # Adaptive number of clusters
    if n_clusters >= 2:
        cluster_labels, kmeans = cluster_images(features, n_clusters=n_clusters)

        # 4. Visualize clusters
        visualize_clusters(cluster_labels, n_clusters=n_clusters)

        # 5. Save cluster assignments
        save_cluster_assignments(image_paths, cluster_labels)
    else:
        print(f"\n⚠️  Too few images ({len(features)}) for clustering")

    # 6. UMAP visualization (if available and enough images)
    if len(features) >= 10:
        features_2d = visualize_umap_2d(features, n_neighbors=min(15, len(features)-1))

    print("\n" + "="*70)
    print("✅ Analysis complete!")
    print("="*70)

    print("\nGenerated files:")
    if os.path.exists('cluster_distribution.png'):
        print(f"  - cluster_distribution.png (cluster histogram)")
    if os.path.exists('image_clusters.csv'):
        print(f"  - image_clusters.csv (cluster assignments)")
    if os.path.exists('dinov2_umap_2d.png'):
        print(f"  - dinov2_umap_2d.png (2D visualization)")


if __name__ == "__main__":
    main()
