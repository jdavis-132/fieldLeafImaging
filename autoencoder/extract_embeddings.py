"""
Extract embeddings from trained autoencoder.

Process all images through the encoder to get bottleneck embeddings.
Save embeddings with metadata for downstream analysis.
"""

import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from autoencoder.config import Config
from autoencoder.model import create_model
from autoencoder.dataset import MaskedLeafDataset
from autoencoder.utils import load_checkpoint


def extract_embeddings(model, dataset, config, split_name):
    """
    Extract embeddings for all images in a dataset.

    Args:
        model: Trained autoencoder
        dataset: MaskedLeafDataset
        config: Config object
        split_name: 'train', 'val', or 'test'

    Returns:
        embeddings: np.ndarray of shape (N, embedding_dim)
        metadata: List of dicts with image metadata
    """
    model.eval()

    embeddings_list = []
    metadata_list = []

    print(f"Extracting embeddings for {split_name} set ({len(dataset)} images)...")

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f'Extracting {split_name}'):
            # Get sample
            sample = dataset[idx]
            image = sample['masked_image'].unsqueeze(0).to(config.device)

            # Extract embedding
            embedding = model.encode(image)

            # Store
            embeddings_list.append(embedding.cpu().numpy().flatten())
            metadata_list.append({
                'filename': sample['filename'],
                'plot': sample['plot'],
                'genotype': sample['genotype'],
                'device': sample['device'],
                'split': split_name
            })

    embeddings = np.stack(embeddings_list, axis=0)

    return embeddings, metadata_list


def save_embeddings(embeddings_dict, metadata_dict, config):
    """
    Save embeddings and metadata to files.

    Args:
        embeddings_dict: Dict with keys 'train', 'val', 'test' -> np.ndarray
        metadata_dict: Dict with keys 'train', 'val', 'test' -> list of dicts
        config: Config object
    """
    # Combine all splits
    all_embeddings = np.concatenate([
        embeddings_dict['train'],
        embeddings_dict['val'],
        embeddings_dict['test']
    ], axis=0)

    all_metadata = (
        metadata_dict['train'] +
        metadata_dict['val'] +
        metadata_dict['test']
    )

    print(f"\nTotal embeddings: {all_embeddings.shape[0]}")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")

    # Save as NPZ (numpy format)
    npz_path = config.embeddings_dir / 'embeddings.npz'
    np.savez(
        npz_path,
        embeddings=all_embeddings,
        train_embeddings=embeddings_dict['train'],
        val_embeddings=embeddings_dict['val'],
        test_embeddings=embeddings_dict['test']
    )
    print(f"✓ Saved embeddings to {npz_path}")

    # Save as CSV with metadata
    csv_path = config.embeddings_dir / 'embeddings.csv'

    # Create DataFrame
    rows = []
    for i, meta in enumerate(all_metadata):
        row = {
            'filename': meta['filename'],
            'plot': meta['plot'],
            'genotype': meta['genotype'],
            'device': meta['device'],
            'split': meta['split']
        }

        # Add embedding dimensions
        for dim in range(all_embeddings.shape[1]):
            row[f'emb_{dim}'] = all_embeddings[i, dim]

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved embeddings CSV to {csv_path}")

    # Save metadata separately as JSON
    json_path = config.embeddings_dir / 'embeddings_metadata.json'
    with open(json_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"✓ Saved metadata to {json_path}")

    # Save summary statistics
    summary_path = config.embeddings_dir / 'embeddings_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Embedding Extraction Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total images: {all_embeddings.shape[0]}\n")
        f.write(f"Embedding dimension: {all_embeddings.shape[1]}\n\n")

        for split_name in ['train', 'val', 'test']:
            n = embeddings_dict[split_name].shape[0]
            f.write(f"{split_name.upper()}: {n} images\n")

        f.write("\nEmbedding statistics:\n")
        f.write(f"  Mean: {all_embeddings.mean():.6f}\n")
        f.write(f"  Std:  {all_embeddings.std():.6f}\n")
        f.write(f"  Min:  {all_embeddings.min():.6f}\n")
        f.write(f"  Max:  {all_embeddings.max():.6f}\n")

        f.write("\nGenotype distribution:\n")
        genotypes = [m['genotype'] for m in all_metadata]
        from collections import Counter
        genotype_counts = Counter(genotypes)
        for genotype, count in genotype_counts.most_common(20):
            f.write(f"  {genotype:20} {count:4d} images\n")

    print(f"✓ Saved summary to {summary_path}")

    return df


def visualize_embeddings_umap(embeddings, metadata, config, save_path=None):
    """
    Visualize embeddings using UMAP.

    Args:
        embeddings: np.ndarray of shape (N, embedding_dim)
        metadata: List of dicts with image metadata
        config: Config object
        save_path: Optional path to save figure
    """
    try:
        import umap
        import matplotlib.pyplot as plt
        import seaborn as sns

        print("\nGenerating UMAP visualization...")

        # Fit UMAP
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=config.random_seed
        )

        embedding_2d = reducer.fit_transform(embeddings)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Plot 1: Color by split
        splits = [m['split'] for m in metadata]
        split_colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}

        for split_name in ['train', 'val', 'test']:
            mask = np.array(splits) == split_name
            axes[0].scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=split_colors[split_name],
                label=split_name,
                alpha=0.6,
                s=20
            )

        axes[0].set_xlabel('UMAP 1', fontsize=12)
        axes[0].set_ylabel('UMAP 2', fontsize=12)
        axes[0].set_title('Embeddings colored by Split', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Color by top genotypes
        genotypes = [m['genotype'] for m in metadata]
        from collections import Counter
        top_genotypes = [g for g, _ in Counter(genotypes).most_common(10)]

        # Assign colors
        colors = sns.color_palette('tab10', n_colors=len(top_genotypes))
        genotype_to_color = {g: colors[i] for i, g in enumerate(top_genotypes)}

        for genotype in top_genotypes:
            mask = np.array(genotypes) == genotype
            axes[1].scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=[genotype_to_color[genotype]],
                label=genotype,
                alpha=0.6,
                s=20
            )

        # Other genotypes in gray
        mask = ~np.isin(genotypes, top_genotypes)
        axes[1].scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c='lightgray',
            label='Other',
            alpha=0.3,
            s=10
        )

        axes[1].set_xlabel('UMAP 1', fontsize=12)
        axes[1].set_ylabel('UMAP 2', fontsize=12)
        axes[1].set_title('Embeddings colored by Top 10 Genotypes', fontsize=14, fontweight='bold')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved UMAP visualization to {save_path}")

        return fig

    except ImportError:
        print("Warning: umap-learn not installed. Skipping UMAP visualization.")
        print("Install with: pip install umap-learn")
        return None


def visualize_embeddings_pca(embeddings, metadata, config, save_path=None):
    """
    Visualize embeddings using PCA (fallback if UMAP not available).

    Args:
        embeddings: np.ndarray of shape (N, embedding_dim)
        metadata: List of dicts with image metadata
        config: Config object
        save_path: Optional path to save figure
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    print("\nGenerating PCA visualization...")

    # Fit PCA
    pca = PCA(n_components=2, random_state=config.random_seed)
    embedding_2d = pca.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    splits = [m['split'] for m in metadata]
    split_colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}

    for split_name in ['train', 'val', 'test']:
        mask = np.array(splits) == split_name
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=split_colors[split_name],
            label=split_name,
            alpha=0.6,
            s=20
        )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('Embeddings PCA', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved PCA visualization to {save_path}")

    return fig


def main():
    """Main entry point."""
    config = Config()

    print("=" * 80)
    print("Extracting Embeddings from Trained Autoencoder")
    print("=" * 80)

    # Load best model
    model_path = config.models_dir / 'best_model.pth'
    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model not found: {model_path}\n"
            "Please train the model first using train.py"
        )

    print(f"\nLoading model from {model_path}...")
    model = create_model(config)
    load_checkpoint(model_path, model, device=config.device)

    # Load splits
    splits_path = config.logs_dir / 'image_splits.json'
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    # Extract embeddings for each split
    embeddings_dict = {}
    metadata_dict = {}

    for split_name in ['train', 'val', 'test']:
        dataset = MaskedLeafDataset(splits[split_name], config)

        embeddings, metadata = extract_embeddings(
            model, dataset, config, split_name
        )

        embeddings_dict[split_name] = embeddings
        metadata_dict[split_name] = metadata

    # Save embeddings
    print("\n" + "=" * 80)
    print("Saving Embeddings")
    print("=" * 80)
    df = save_embeddings(embeddings_dict, metadata_dict, config)

    # Combine for visualization
    all_embeddings = np.concatenate([
        embeddings_dict['train'],
        embeddings_dict['val'],
        embeddings_dict['test']
    ], axis=0)

    all_metadata = (
        metadata_dict['train'] +
        metadata_dict['val'] +
        metadata_dict['test']
    )

    # Visualize with UMAP
    umap_path = config.visualizations_dir / 'embeddings_umap.png'
    visualize_embeddings_umap(all_embeddings, all_metadata, config, save_path=umap_path)

    # Visualize with PCA (always available)
    pca_path = config.visualizations_dir / 'embeddings_pca.png'
    visualize_embeddings_pca(all_embeddings, all_metadata, config, save_path=pca_path)

    print("\n" + "=" * 80)
    print("Embedding Extraction Complete!")
    print("=" * 80)
    print(f"Embeddings saved to: {config.embeddings_dir}")
    print(f"Visualizations saved to: {config.visualizations_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
