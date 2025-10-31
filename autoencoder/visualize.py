"""
Visualization and analysis tools for the autoencoder pipeline.

Generate comprehensive visualizations and summary reports.
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from autoencoder.config import Config
from autoencoder.dataset import MaskedLeafDataset
from autoencoder.utils import visualize_masking_process


def visualize_dataset_overview(splits, config):
    """
    Create overview visualizations of the dataset.

    Args:
        splits: Dict with 'train', 'val', 'test' splits
        config: Config object
    """
    print("Generating dataset overview visualizations...")

    # Create datasets
    datasets = {}
    for split_name in ['train', 'val', 'test']:
        datasets[split_name] = MaskedLeafDataset(splits[split_name], config)

    # Visualize masking process
    print("Visualizing masking process...")
    mask_vis_path = config.visualizations_dir / 'masking_process_examples.png'
    visualize_masking_process(
        datasets['train'],
        num_samples=4,
        save_path=mask_vis_path
    )

    # Plot split distribution
    print("Plotting split distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Images per split
    split_counts = {name: len(splits[name]) for name in ['train', 'val', 'test']}
    axes[0].bar(split_counts.keys(), split_counts.values(),
                color=['steelblue', 'orange', 'green'])
    axes[0].set_ylabel('Number of Images', fontsize=12)
    axes[0].set_title('Images per Split', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for i, (split_name, count) in enumerate(split_counts.items()):
        axes[0].text(i, count + 50, str(count), ha='center', fontweight='bold')

    # Genotypes per split
    genotype_counts = {}
    for split_name in ['train', 'val', 'test']:
        genotypes = set(img['genotype'] for img in splits[split_name])
        genotype_counts[split_name] = len(genotypes)

    axes[1].bar(genotype_counts.keys(), genotype_counts.values(),
                color=['steelblue', 'orange', 'green'])
    axes[1].set_ylabel('Number of Genotypes', fontsize=12)
    axes[1].set_title('Unique Genotypes per Split', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    # Add count labels
    for i, (split_name, count) in enumerate(genotype_counts.items()):
        axes[1].text(i, count + 5, str(count), ha='center', fontweight='bold')

    plt.tight_layout()
    split_dist_path = config.visualizations_dir / 'split_distribution.png'
    plt.savefig(split_dist_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved split distribution to {split_dist_path}")

    plt.close()


def plot_genotype_distribution(splits, config):
    """
    Plot genotype distribution across splits.

    Args:
        splits: Dict with splits
        config: Config object
    """
    from collections import Counter

    print("Plotting genotype distribution...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for idx, split_name in enumerate(['train', 'val', 'test']):
        genotypes = [img['genotype'] for img in splits[split_name]]
        genotype_counts = Counter(genotypes)

        # Get top 20 genotypes
        top_20 = genotype_counts.most_common(20)
        names = [g for g, _ in top_20]
        counts = [c for _, c in top_20]

        axes[idx].barh(range(len(names)), counts, color='steelblue')
        axes[idx].set_yticks(range(len(names)))
        axes[idx].set_yticklabels(names, fontsize=9)
        axes[idx].set_xlabel('Number of Images', fontsize=11)
        axes[idx].set_title(f'{split_name.upper()} - Top 20 Genotypes',
                           fontsize=13, fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)
        axes[idx].invert_yaxis()

        # Add count labels
        for i, count in enumerate(counts):
            axes[idx].text(count + 0.5, i, str(count), va='center', fontsize=8)

    plt.tight_layout()
    genotype_dist_path = config.visualizations_dir / 'genotype_distribution.png'
    plt.savefig(genotype_dist_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved genotype distribution to {genotype_dist_path}")

    plt.close()


def generate_summary_report(config):
    """
    Generate a comprehensive summary report.

    Args:
        config: Config object
    """
    print("\nGenerating summary report...")

    # Load splits
    with open(config.logs_dir / 'image_splits.json', 'r') as f:
        splits = json.load(f)

    with open(config.logs_dir / 'genotype_splits.json', 'r') as f:
        genotype_splits = json.load(f)

    # Create report
    report_path = config.visualizations_dir / 'SUMMARY_REPORT.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AUTOENCODER PIPELINE SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("Dataset Information\n")
        f.write("-" * 80 + "\n")
        total_images = sum(len(splits[s]) for s in ['train', 'val', 'test'])
        f.write(f"Total usable images: {total_images}\n")
        f.write(f"  - With normalized image: ✓\n")
        f.write(f"  - With both masks (strip_0000 + strip_0001): ✓\n")
        f.write(f"  - With genotype metadata: ✓\n\n")

        f.write("Data Splits\n")
        f.write("-" * 80 + "\n")
        for split_name in ['train', 'val', 'test']:
            imgs = splits[split_name]
            genotypes = set(img['genotype'] for img in imgs)
            f.write(f"{split_name.upper()}:\n")
            f.write(f"  Images:    {len(imgs):5d} ({len(imgs)/total_images*100:.1f}%)\n")
            f.write(f"  Genotypes: {len(genotypes):5d}\n")
            f.write(f"  Avg images/genotype: {len(imgs)/len(genotypes):.2f}\n\n")

        f.write("Genotype Split Strategy\n")
        f.write("-" * 80 + "\n")
        f.write("CRITICAL: All images with the same genotype are in the same split\n")
        f.write("This prevents data leakage where the model sees the same genotype\n")
        f.write("in both training and testing.\n\n")

        total_genotypes = (len(genotype_splits['train_genotypes']) +
                          len(genotype_splits['val_genotypes']) +
                          len(genotype_splits['test_genotypes']))
        f.write(f"Total genotypes: {total_genotypes}\n")
        f.write(f"  Train genotypes: {len(genotype_splits['train_genotypes'])}\n")
        f.write(f"  Val genotypes:   {len(genotype_splits['val_genotypes'])}\n")
        f.write(f"  Test genotypes:  {len(genotype_splits['test_genotypes'])}\n\n")

        f.write("Model Configuration\n")
        f.write("-" * 80 + "\n")
        f.write(f"Input image size:    {config.image_size[0]} × {config.image_size[1]}\n")
        f.write(f"Embedding dimension: {config.embedding_dim}\n")
        f.write(f"Batch size:          {config.batch_size}\n")
        f.write(f"Learning rate:       {config.learning_rate}\n")
        f.write(f"Max epochs:          {config.num_epochs}\n")
        f.write(f"Early stopping:      {config.patience} epochs\n")
        f.write(f"Random seed:         {config.random_seed}\n")
        f.write(f"Device:              {config.device}\n\n")

        f.write("Masking Process\n")
        f.write("-" * 80 + "\n")
        f.write("1. Load normalized RGB image\n")
        f.write("2. Load mask 1 (strip_0000_mask.png)\n")
        f.write("3. Load mask 2 (strip_0001_mask.png)\n")
        f.write("4. Combine masks with logical OR\n")
        f.write("5. Set pixels OUTSIDE combined mask to [0, 0, 0] (black)\n")
        f.write("6. Keep pixels INSIDE combined mask at original values\n")
        f.write("7. Resize to target size and normalize\n\n")

        f.write("Output Files\n")
        f.write("-" * 80 + "\n")
        f.write(f"Models:         {config.models_dir}/\n")
        f.write(f"  - best_model.pth (best validation loss)\n")
        f.write(f"  - checkpoint_epoch_*.pth (per-epoch checkpoints)\n\n")

        f.write(f"Embeddings:     {config.embeddings_dir}/\n")
        f.write(f"  - embeddings.csv (with metadata)\n")
        f.write(f"  - embeddings.npz (numpy arrays)\n")
        f.write(f"  - embeddings_metadata.json\n\n")

        f.write(f"Visualizations: {config.visualizations_dir}/\n")
        f.write(f"  - training_curves.png\n")
        f.write(f"  - reconstructions_*.png\n")
        f.write(f"  - masking_process_examples.png\n")
        f.write(f"  - embeddings_umap.png / embeddings_pca.png\n\n")

        f.write(f"Logs:           {config.logs_dir}/\n")
        f.write(f"  - image_splits.json\n")
        f.write(f"  - genotype_splits.json\n")
        f.write(f"  - training_log.csv\n")
        f.write(f"  - dataset_summary.txt\n\n")

        f.write("Next Steps\n")
        f.write("-" * 80 + "\n")
        f.write("1. Review visualizations in {visualizations}/\n")
        f.write("2. Check training curves for overfitting\n")
        f.write("3. Examine reconstructions for quality\n")
        f.write("4. Analyze embeddings for genotype clustering\n")
        f.write("5. Use embeddings for downstream tasks:\n")
        f.write("   - Genotype classification\n")
        f.write("   - Phenotype prediction\n")
        f.write("   - Clustering analysis\n")
        f.write("   - Dimensionality reduction visualization\n\n")

        f.write("=" * 80 + "\n")

    print(f"✓ Summary report saved to {report_path}")


def main():
    """Main entry point."""
    config = Config()

    print("=" * 80)
    print("Generating Visualizations and Summary Report")
    print("=" * 80)

    # Load splits
    with open(config.logs_dir / 'image_splits.json', 'r') as f:
        splits = json.load(f)

    # Generate visualizations
    visualize_dataset_overview(splits, config)
    plot_genotype_distribution(splits, config)

    # Generate summary report
    generate_summary_report(config)

    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print(f"All visualizations saved to: {config.visualizations_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
