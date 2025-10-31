"""
Evaluation and visualization script for disease-aware autoencoder.

- Extract latent representations
- Create UMAP visualization
- Display reconstruction examples
- Save results for clustering
"""

import os
import sys
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import umap

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from disease_autoencoder.config import DiseaseConfig
from disease_autoencoder.model import DiseaseAutoencoder
from disease_autoencoder.dataset import DiseaseLeafDataset


def load_model(config, checkpoint_path):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    # Create model
    model = DiseaseAutoencoder(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()

    print(f"  Loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.6f}")

    return model, checkpoint


def extract_embeddings(model, dataloader, config):
    """Extract latent embeddings for all images."""
    print("\nExtracting embeddings...")

    embeddings_list = []
    metadata_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting embeddings'):
            inputs = batch['input'].to(config.device)

            # Extract embeddings
            embeddings = model.encode(inputs)

            # Store
            embeddings_list.append(embeddings.cpu().numpy())
            # batch['metadata'] is already a list of dicts, one per sample in batch
            for i in range(len(batch['input'])):
                metadata_list.append({
                    'genotype': batch['metadata']['genotype'][i] if isinstance(batch['metadata']['genotype'], (list, tuple)) else batch['metadata']['genotype'],
                    'image_path': batch['metadata']['image_path'][i] if isinstance(batch['metadata']['image_path'], (list, tuple)) else batch['metadata']['image_path'],
                    'mask0_path': batch['metadata']['mask0_path'][i] if isinstance(batch['metadata']['mask0_path'], (list, tuple)) else batch['metadata']['mask0_path'],
                    'mask1_path': batch['metadata']['mask1_path'][i] if isinstance(batch['metadata']['mask1_path'], (list, tuple)) else batch['metadata']['mask1_path']
                })

    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings_list, axis=0)

    print(f"  Extracted {len(embeddings)} embeddings")
    print(f"  Embedding shape: {embeddings.shape}")

    return embeddings, metadata_list


def create_umap_visualization(embeddings, metadata_list, config):
    """Create UMAP visualization of latent space."""
    print("\nCreating UMAP visualization...")

    # Create UMAP projection
    reducer = umap.UMAP(
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=config.random_seed
    )

    embedding_2d = reducer.fit_transform(embeddings)

    print(f"  UMAP projection shape: {embedding_2d.shape}")

    # Create visualization colored by genotype
    genotypes = [m['genotype'] for m in metadata_list]
    unique_genotypes = sorted(list(set(genotypes)))

    # Create color map
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_genotypes), 20)))
    genotype_to_color = {g: colors[i % 20] for i, g in enumerate(unique_genotypes)}

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Colored by genotype (show top 10 most common)
    from collections import Counter
    genotype_counts = Counter(genotypes)
    top_genotypes = [g for g, _ in genotype_counts.most_common(10)]

    for genotype in top_genotypes:
        mask = np.array([g == genotype for g in genotypes])
        axes[0].scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[genotype_to_color[genotype]],
            label=genotype,
            alpha=0.6,
            s=20
        )

    axes[0].set_title('UMAP Latent Space (Top 10 Genotypes)', fontsize=14)
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Plot 2: All points colored by genotype
    genotype_ids = [unique_genotypes.index(g) for g in genotypes]
    scatter = axes[1].scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=genotype_ids,
        cmap='tab20',
        alpha=0.3,
        s=10
    )
    axes[1].set_title(f'UMAP Latent Space (All {len(unique_genotypes)} Genotypes)', fontsize=14)
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')

    plt.tight_layout()

    # Save figure
    umap_path = config.visualizations_dir / 'umap_latent_space.png'
    plt.savefig(umap_path, dpi=300, bbox_inches='tight')
    print(f"  Saved UMAP visualization to {umap_path}")

    plt.close()

    # Save UMAP coordinates
    umap_coords_path = config.embeddings_dir / 'umap_coordinates.npy'
    np.save(umap_coords_path, embedding_2d)
    print(f"  Saved UMAP coordinates to {umap_coords_path}")

    return embedding_2d


def visualize_reconstructions(model, dataset, config, num_samples=8):
    """Visualize reconstruction examples."""
    print("\nCreating reconstruction visualizations...")

    # Select random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get sample
            sample = dataset[idx]
            input_tensor = sample['input'].unsqueeze(0).to(config.device)  # (1, 4, H, W)
            target_tensor = sample['target'].unsqueeze(0)  # (1, 3, H, W)
            mask_tensor = sample['mask'].unsqueeze(0)  # (1, 1, H, W)

            # Get reconstruction
            reconstruction = model(input_tensor).cpu()  # (1, 3, H, W)

            # Denormalize LAB images (convert back to original LAB range)
            with open(config.logs_dir / 'lab_statistics.json', 'r') as f:
                lab_stats = json.load(f)

            lab_mean = np.array(lab_stats['lab_mean']).reshape(1, 3, 1, 1)
            lab_std = np.array(lab_stats['lab_std']).reshape(1, 3, 1, 1)

            target_lab = target_tensor.numpy() * lab_std + lab_mean
            recon_lab = reconstruction.numpy() * lab_std + lab_mean

            # Convert LAB to RGB for visualization
            target_lab_hwc = target_lab[0].transpose(1, 2, 0).astype(np.float32)
            recon_lab_hwc = recon_lab[0].transpose(1, 2, 0).astype(np.float32)

            # Clip to valid LAB range
            target_lab_hwc = np.clip(target_lab_hwc, [0, -128, -128], [100, 127, 127])
            recon_lab_hwc = np.clip(recon_lab_hwc, [0, -128, -128], [100, 127, 127])

            # Convert to uint8 for cv2
            target_lab_uint8 = target_lab_hwc.astype(np.uint8)
            recon_lab_uint8 = recon_lab_hwc.astype(np.uint8)

            target_rgb = cv2.cvtColor(target_lab_uint8, cv2.COLOR_LAB2RGB)
            recon_rgb = cv2.cvtColor(recon_lab_uint8, cv2.COLOR_LAB2RGB)

            # Apply mask for visualization
            mask_vis = mask_tensor[0, 0].numpy()
            target_rgb_masked = target_rgb * mask_vis[:, :, np.newaxis]
            recon_rgb_masked = recon_rgb * mask_vis[:, :, np.newaxis]

            # Plot
            axes[i, 0].imshow(target_rgb_masked.astype(np.uint8))
            axes[i, 0].set_title(f'Original (Genotype: {sample["metadata"]["genotype"][:15]}...)')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(recon_rgb_masked.astype(np.uint8))
            axes[i, 1].set_title('Reconstruction')
            axes[i, 1].axis('off')

            # Compute difference
            diff = np.abs(target_rgb.astype(float) - recon_rgb.astype(float))
            diff = (diff * mask_vis[:, :, np.newaxis]).astype(np.uint8)
            axes[i, 2].imshow(diff)
            axes[i, 2].set_title('Absolute Difference')
            axes[i, 2].axis('off')

    plt.tight_layout()

    # Save figure
    recon_path = config.visualizations_dir / 'reconstructions.png'
    plt.savefig(recon_path, dpi=300, bbox_inches='tight')
    print(f"  Saved reconstructions to {recon_path}")

    plt.close()


def save_embeddings(embeddings, metadata_list, config):
    """Save embeddings and metadata for later clustering."""
    print("\nSaving embeddings for clustering...")

    # Save embeddings
    embeddings_path = config.embeddings_dir / 'latent_embeddings.npy'
    np.save(embeddings_path, embeddings)
    print(f"  Saved embeddings to {embeddings_path}")

    # Save metadata
    metadata_path = config.embeddings_dir / 'embeddings_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")

    # Create summary
    summary = {
        'num_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'num_genotypes': len(set(m['genotype'] for m in metadata_list)),
        'num_devices': len(set(m['device'] for m in metadata_list)),
        'genotype_distribution': dict(
            sorted(
                [(g, sum(1 for m in metadata_list if m['genotype'] == g))
                 for g in set(m['genotype'] for m in metadata_list)],
                key=lambda x: x[1],
                reverse=True
            )[:20]  # Top 20
        )
    }

    summary_path = config.embeddings_dir / 'embeddings_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to {summary_path}")


def main():
    """Main evaluation function."""
    config = DiseaseConfig()

    print("="*60)
    print("Disease-Aware Autoencoder Evaluation")
    print("="*60)

    # Load best model
    checkpoint_path = config.models_dir / 'checkpoint_best.pth'
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return

    model, checkpoint = load_model(config, checkpoint_path)

    # Load data splits
    splits_path = config.logs_dir / 'image_splits.json'
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    # Load LAB statistics
    with open(config.logs_dir / 'lab_statistics.json', 'r') as f:
        lab_stats = json.load(f)

    # Create datasets (combine all splits for evaluation)
    all_images = splits['train'] + splits['val'] + splits['test']

    dataset = DiseaseLeafDataset(
        image_metadata_list=all_images,
        config=config,
        transform=None
    )
    dataset.set_lab_statistics(lab_stats['lab_mean'], lab_stats['lab_std'])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    print(f"\nEvaluating on {len(dataset)} images")

    # Extract embeddings
    embeddings, metadata_list = extract_embeddings(model, dataloader, config)

    # Create UMAP visualization
    umap_coords = create_umap_visualization(embeddings, metadata_list, config)

    # Visualize reconstructions
    visualize_reconstructions(model, dataset, config, num_samples=8)

    # Save embeddings
    save_embeddings(embeddings, metadata_list, config)

    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - Embeddings: {config.embeddings_dir / 'latent_embeddings.npy'}")
    print(f"  - Metadata: {config.embeddings_dir / 'embeddings_metadata.json'}")
    print(f"  - UMAP coords: {config.embeddings_dir / 'umap_coordinates.npy'}")
    print(f"  - UMAP plot: {config.visualizations_dir / 'umap_latent_space.png'}")
    print(f"  - Reconstructions: {config.visualizations_dir / 'reconstructions.png'}")
    print("\nYou can now use the embeddings for clustering analysis!")


if __name__ == '__main__':
    main()
