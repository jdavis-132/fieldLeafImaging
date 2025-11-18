"""
Helper script for clustering analysis using extracted embeddings.

Example usage:
    python -m disease_autoencoder.cluster_analysis --n_clusters 5 --method kmeans
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

# Clustering algorithms
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


def load_embeddings(embeddings_dir):
    """Load embeddings and metadata."""
    embeddings_path = embeddings_dir / 'latent_embeddings.npy'
    metadata_path = embeddings_dir / 'embeddings_metadata.json'
    umap_path = embeddings_dir / 'umap_coordinates.npy'

    embeddings = np.load(embeddings_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    umap_coords = None
    if umap_path.exists():
        umap_coords = np.load(umap_path)

    return embeddings, metadata, umap_coords


def cluster_kmeans(embeddings, n_clusters=5, random_state=42):
    """K-means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    return clusters, kmeans


def cluster_dbscan(embeddings, eps=0.5, min_samples=5):
    """DBSCAN clustering."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(embeddings)
    return clusters, dbscan


def cluster_hierarchical(embeddings, n_clusters=5):
    """Hierarchical clustering."""
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = hierarchical.fit_predict(embeddings)
    return clusters, hierarchical


def cluster_gmm(embeddings, n_components=5, random_state=42):
    """Gaussian Mixture Model clustering."""
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    clusters = gmm.fit_predict(embeddings)
    return clusters, gmm


def analyze_clusters(clusters, metadata):
    """Analyze cluster composition."""
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

    print(f"\n{'='*60}")
    print(f"Cluster Analysis")
    print(f"{'='*60}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of samples: {len(clusters)}")

    if -1 in clusters:
        n_noise = sum(1 for c in clusters if c == -1)
        print(f"Noise points: {n_noise} ({n_noise/len(clusters)*100:.1f}%)")

    print(f"\n{'='*60}")

    for cluster_id in sorted(set(clusters)):
        if cluster_id == -1:
            continue

        cluster_mask = clusters == cluster_id
        cluster_size = sum(cluster_mask)

        # Get metadata for this cluster
        cluster_genotypes = [metadata[i]['genotype']
                            for i in range(len(metadata)) if cluster_mask[i]]

        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {cluster_size} images ({cluster_size/len(clusters)*100:.1f}%)")
        print(f"  Unique genotypes: {len(set(cluster_genotypes))}")

        # Top genotypes
        genotype_counts = Counter(cluster_genotypes)
        print(f"  Top 5 genotypes:")
        for genotype, count in genotype_counts.most_common(5):
            pct = count / cluster_size * 100
            print(f"    - {genotype[:30]:30} {count:4d} ({pct:5.1f}%)")


def visualize_clusters(umap_coords, clusters, metadata, output_path):
    """Visualize clusters in UMAP space."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Clusters
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    scatter = axes[0].scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=clusters,
        cmap='tab10' if n_clusters <= 10 else 'tab20',
        alpha=0.6,
        s=20
    )
    axes[0].set_title(f'Clusters in UMAP Space (n={n_clusters})', fontsize=14)
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=axes[0], label='Cluster')

    # Plot 2: Genotype diversity per cluster
    cluster_diversity = []
    cluster_ids = sorted(set(clusters))
    if -1 in cluster_ids:
        cluster_ids.remove(-1)

    for cluster_id in cluster_ids:
        cluster_mask = clusters == cluster_id
        cluster_genotypes = [metadata[i]['genotype']
                            for i in range(len(metadata)) if cluster_mask[i]]
        n_unique = len(set(cluster_genotypes))
        cluster_diversity.append(n_unique)

    axes[1].bar(cluster_ids, cluster_diversity, color='steelblue', alpha=0.7)
    axes[1].set_xlabel('Cluster ID')
    axes[1].set_ylabel('Number of Unique Genotypes')
    axes[1].set_title('Genotype Diversity per Cluster', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCluster visualization saved to {output_path}")
    plt.close()


def create_confusion_matrix(clusters, metadata, output_path):
    """Create confusion matrix: genotype vs cluster."""
    # Get top genotypes
    all_genotypes = [m['genotype'] for m in metadata]
    genotype_counts = Counter(all_genotypes)
    top_genotypes = [g for g, _ in genotype_counts.most_common(15)]

    # Filter to top genotypes
    filtered_indices = [i for i, m in enumerate(metadata)
                       if m['genotype'] in top_genotypes]

    if len(filtered_indices) == 0:
        print("Not enough data for confusion matrix")
        return

    filtered_clusters = clusters[filtered_indices]
    filtered_genotypes = [metadata[i]['genotype'] for i in filtered_indices]

    # Create matrix
    cluster_ids = sorted(set(filtered_clusters))
    if -1 in cluster_ids:
        cluster_ids.remove(-1)

    matrix = np.zeros((len(top_genotypes), len(cluster_ids)))

    for i, genotype in enumerate(top_genotypes):
        for j, cluster_id in enumerate(cluster_ids):
            count = sum(1 for g, c in zip(filtered_genotypes, filtered_clusters)
                       if g == genotype and c == cluster_id)
            matrix[i, j] = count

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        xticklabels=[f'C{c}' for c in cluster_ids],
        yticklabels=[g[:20] for g in top_genotypes],
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Number of Images'}
    )
    plt.title('Genotype vs Cluster Distribution (Top 15 Genotypes)', fontsize=14)
    plt.xlabel('Cluster ID')
    plt.ylabel('Genotype')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Cluster analysis of embeddings')
    parser.add_argument('--method', type=str, default='kmeans',
                       choices=['kmeans', 'dbscan', 'hierarchical', 'gmm'],
                       help='Clustering method')
    parser.add_argument('--n_clusters', type=int, default=5,
                       help='Number of clusters (for kmeans, hierarchical, gmm)')
    parser.add_argument('--eps', type=float, default=0.5,
                       help='DBSCAN epsilon parameter')
    parser.add_argument('--min_samples', type=int, default=5,
                       help='DBSCAN min_samples parameter')

    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).parent
    embeddings_dir = base_dir / 'embeddings'
    viz_dir = base_dir / 'visualizations'

    if not embeddings_dir.exists():
        print(f"Error: Embeddings directory not found: {embeddings_dir}")
        print("Please run evaluate.py first to extract embeddings")
        return

    # Load embeddings
    print("Loading embeddings...")
    embeddings, metadata, umap_coords = load_embeddings(embeddings_dir)
    print(f"Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Cluster
    print(f"\nClustering with {args.method}...")

    if args.method == 'kmeans':
        clusters, model = cluster_kmeans(embeddings, n_clusters=args.n_clusters)
    elif args.method == 'dbscan':
        clusters, model = cluster_dbscan(embeddings, eps=args.eps,
                                        min_samples=args.min_samples)
    elif args.method == 'hierarchical':
        clusters, model = cluster_hierarchical(embeddings, n_clusters=args.n_clusters)
    elif args.method == 'gmm':
        clusters, model = cluster_gmm(embeddings, n_components=args.n_clusters)

    # Analyze
    analyze_clusters(clusters, metadata)

    # Visualize
    if umap_coords is not None:
        output_path = viz_dir / f'clusters_{args.method}.png'
        visualize_clusters(umap_coords, clusters, metadata, output_path)

        # Confusion matrix
        confusion_path = viz_dir / f'confusion_matrix_{args.method}.png'
        create_confusion_matrix(clusters, metadata, confusion_path)

    # Save cluster assignments
    clusters_path = embeddings_dir / f'clusters_{args.method}.npy'
    np.save(clusters_path, clusters)
    print(f"\nCluster assignments saved to {clusters_path}")

    # Save cluster summary
    summary = {
        'method': args.method,
        'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
        'parameters': vars(args),
        'cluster_sizes': {
            int(c): int(sum(1 for x in clusters if x == c))
            for c in set(clusters)
        }
    }

    summary_path = embeddings_dir / f'cluster_summary_{args.method}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Cluster summary saved to {summary_path}")


if __name__ == '__main__':
    main()
