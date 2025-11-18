"""
Evaluation and visualization script for Variational Autoencoder (VAE).

Generates:
- Latent space visualizations (UMAP, t-SNE, PCA)
- Reconstruction comparisons
- Latent space interpolations
- Distribution plots of latent dimensions
- Metrics (MSE, SSIM)
"""

import os
import sys
import json

# Set matplotlib backend to non-GUI before importing pyplot
import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vae.config import VAEConfig
from src.vae.model import VAE
from src.vae.dataset import VAELeafDataset


def load_model(config, checkpoint_path):
    """Load trained VAE model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    # Create model
    model = VAE(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()

    print(f"  Loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.6f}")

    return model, checkpoint


def extract_embeddings_and_latents(model, dataloader, config):
    """Extract latent embeddings (mu) for all images."""
    print("\nExtracting latent embeddings...")

    embeddings_list = []
    metadata_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting embeddings'):
            inputs = batch['input'].to(config.device)

            # Extract latent representations (mu)
            mu, log_var = model.encode(inputs)

            # Store
            embeddings_list.append(mu.cpu().numpy())

            # Handle metadata
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


def create_latent_space_visualizations(embeddings, metadata_list, config):
    """Create multiple latent space visualizations (UMAP, t-SNE, PCA)."""
    print("\nCreating latent space visualizations...")

    genotypes = [m['genotype'] for m in metadata_list]
    unique_genotypes = sorted(list(set(genotypes)))

    # Create color map
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_genotypes), 20)))
    genotype_to_color = {g: colors[i % 20] for i, g in enumerate(unique_genotypes)}

    # 1. UMAP projection
    print("  Computing UMAP projection...")
    umap_reducer = umap.UMAP(
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=config.random_seed
    )
    umap_embedding = umap_reducer.fit_transform(embeddings)

    # 2. t-SNE projection
    print("  Computing t-SNE projection...")
    tsne = TSNE(
        n_components=2,
        perplexity=config.tsne_perplexity,
        learning_rate=config.tsne_learning_rate,
        random_state=config.random_seed
    )
    tsne_embedding = tsne.fit_transform(embeddings)

    # 3. PCA projection
    print("  Computing PCA projection...")
    pca = PCA(n_components=2, random_state=config.random_seed)
    pca_embedding = pca.fit_transform(embeddings)
    print(f"    PCA explained variance: {pca.explained_variance_ratio_}")

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Get top 10 most common genotypes for labeled plots
    from collections import Counter
    genotype_counts = Counter(genotypes)
    top_genotypes = [g for g, _ in genotype_counts.most_common(10)]

    # Plot UMAP
    for genotype in top_genotypes:
        mask = np.array([g == genotype for g in genotypes])
        axes[0, 0].scatter(
            umap_embedding[mask, 0],
            umap_embedding[mask, 1],
            c=[genotype_to_color[genotype]],
            label=genotype[:20],
            alpha=0.6,
            s=20
        )
    axes[0, 0].set_title('UMAP (Top 10 Genotypes)', fontsize=14)
    axes[0, 0].set_xlabel('UMAP 1')
    axes[0, 0].set_ylabel('UMAP 2')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Plot t-SNE
    for genotype in top_genotypes:
        mask = np.array([g == genotype for g in genotypes])
        axes[0, 1].scatter(
            tsne_embedding[mask, 0],
            tsne_embedding[mask, 1],
            c=[genotype_to_color[genotype]],
            label=genotype[:20],
            alpha=0.6,
            s=20
        )
    axes[0, 1].set_title('t-SNE (Top 10 Genotypes)', fontsize=14)
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')

    # Plot PCA
    for genotype in top_genotypes:
        mask = np.array([g == genotype for g in genotypes])
        axes[0, 2].scatter(
            pca_embedding[mask, 0],
            pca_embedding[mask, 1],
            c=[genotype_to_color[genotype]],
            label=genotype[:20],
            alpha=0.6,
            s=20
        )
    axes[0, 2].set_title(f'PCA (Var: {pca.explained_variance_ratio_.sum():.2%})', fontsize=14)
    axes[0, 2].set_xlabel('PC 1')
    axes[0, 2].set_ylabel('PC 2')

    # Plot all points (colored by genotype)
    genotype_ids = [unique_genotypes.index(g) for g in genotypes]

    scatter = axes[1, 0].scatter(
        umap_embedding[:, 0],
        umap_embedding[:, 1],
        c=genotype_ids,
        cmap='tab20',
        alpha=0.3,
        s=10
    )
    axes[1, 0].set_title(f'UMAP (All {len(unique_genotypes)} Genotypes)', fontsize=14)
    axes[1, 0].set_xlabel('UMAP 1')
    axes[1, 0].set_ylabel('UMAP 2')

    axes[1, 1].scatter(
        tsne_embedding[:, 0],
        tsne_embedding[:, 1],
        c=genotype_ids,
        cmap='tab20',
        alpha=0.3,
        s=10
    )
    axes[1, 1].set_title(f't-SNE (All {len(unique_genotypes)} Genotypes)', fontsize=14)
    axes[1, 1].set_xlabel('t-SNE 1')
    axes[1, 1].set_ylabel('t-SNE 2')

    axes[1, 2].scatter(
        pca_embedding[:, 0],
        pca_embedding[:, 1],
        c=genotype_ids,
        cmap='tab20',
        alpha=0.3,
        s=10
    )
    axes[1, 2].set_title(f'PCA (All {len(unique_genotypes)} Genotypes)', fontsize=14)
    axes[1, 2].set_xlabel('PC 1')
    axes[1, 2].set_ylabel('PC 2')

    plt.tight_layout()

    # Save figure
    vis_path = config.visualizations_dir / 'latent_space_visualizations.png'
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"  Saved latent space visualizations to {vis_path}")
    plt.close()

    # Save coordinates
    np.save(config.embeddings_dir / 'umap_coordinates.npy', umap_embedding)
    np.save(config.embeddings_dir / 'tsne_coordinates.npy', tsne_embedding)
    np.save(config.embeddings_dir / 'pca_coordinates.npy', pca_embedding)

    return umap_embedding, tsne_embedding, pca_embedding


def visualize_reconstructions(model, dataset, config, lab_stats, num_samples=8):
    """Visualize reconstruction examples with metrics (following existing conversion process)."""
    print("\nCreating reconstruction visualizations...")

    # Select random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    mse_values = []
    ssim_values = []

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
            # Following exact process from src/autoencoder_no_weighting_128/evaluate.py
            lab_mean = np.array(lab_stats['lab_mean']).reshape(1, 3, 1, 1)
            lab_std = np.array(lab_stats['lab_std']).reshape(1, 3, 1, 1)

            target_lab = target_tensor.numpy() * lab_std + lab_mean
            recon_lab = reconstruction.numpy() * lab_std + lab_mean

            # Convert LAB to RGB for visualization
            target_lab_hwc = target_lab[0].transpose(1, 2, 0).astype(np.float32)
            recon_lab_hwc = recon_lab[0].transpose(1, 2, 0).astype(np.float32)

            # Data is already in OpenCV's LAB format (all channels 0-255)
            # Just clip and convert to uint8
            target_lab_cv2 = np.clip(target_lab_hwc, 0, 255).astype(np.uint8)
            recon_lab_cv2 = np.clip(recon_lab_hwc, 0, 255).astype(np.uint8)

            # Convert LAB to BGR then to RGB
            target_bgr = cv2.cvtColor(target_lab_cv2, cv2.COLOR_LAB2BGR)
            target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
            recon_bgr = cv2.cvtColor(recon_lab_cv2, cv2.COLOR_LAB2BGR)
            recon_rgb = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2RGB)

            # Apply mask for visualization
            mask_vis = mask_tensor[0, 0].numpy()
            target_rgb_masked = target_rgb  # Keep as is (matching existing code)
            recon_rgb_masked = recon_rgb  # Keep as is (matching existing code)

            # Compute metrics
            mse = np.mean((target_rgb.astype(float) - recon_rgb.astype(float)) ** 2)
            ssim_val = ssim(target_rgb, recon_rgb, channel_axis=2, data_range=255)
            mse_values.append(mse)
            ssim_values.append(ssim_val)

            # Plot
            axes[i, 0].imshow(target_rgb_masked)
            axes[i, 0].set_title(f'Original (Genotype: {sample["metadata"]["genotype"][:15]}...)')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(recon_rgb_masked)
            axes[i, 1].set_title(f'Reconstruction (MSE: {mse:.1f}, SSIM: {ssim_val:.3f})')
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
    print(f"  Average MSE: {np.mean(mse_values):.2f}")
    print(f"  Average SSIM: {np.mean(ssim_values):.3f}")

    plt.close()

    return {'mse': np.mean(mse_values), 'ssim': np.mean(ssim_values)}


def visualize_latent_interpolations(model, dataset, config, lab_stats, num_pairs=4):
    """Visualize latent space interpolations between random pairs."""
    print("\nCreating latent space interpolations...")

    fig, axes = plt.subplots(num_pairs, 7, figsize=(21, 3 * num_pairs))

    if num_pairs == 1:
        axes = axes[np.newaxis, :]

    model.eval()

    with torch.no_grad():
        for i in range(num_pairs):
            # Select two random samples
            idx1, idx2 = np.random.choice(len(dataset), 2, replace=False)
            sample1 = dataset[idx1]
            sample2 = dataset[idx2]

            input1 = sample1['input'].unsqueeze(0).to(config.device)
            input2 = sample2['input'].unsqueeze(0).to(config.device)

            # Get latent representations
            mu1, _ = model.encode(input1)
            mu2, _ = model.encode(input2)

            # Interpolate in latent space
            alphas = np.linspace(0, 1, 5)  # 0, 0.25, 0.5, 0.75, 1.0

            for j, alpha in enumerate(alphas):
                # Interpolate
                z = (1 - alpha) * mu1 + alpha * mu2

                # Reconstruct from interpolated latent
                # First decode to bottleneck features
                bottleneck_features = model.variational_bottleneck.fc_decode(z)
                spatial_size = config.image_size // (2 ** len(config.unet_features))
                bottleneck_channels = config.unet_features[-1] * 2
                bottleneck_recon = bottleneck_features.view(
                    1, bottleneck_channels, spatial_size, spatial_size
                )

                # Apply attention if used
                if model.attention is not None:
                    bottleneck_recon = model.attention(bottleneck_recon)

                # Decode
                reconstruction = model.decoder(bottleneck_recon)

                # Denormalize using same process
                lab_mean = np.array(lab_stats['lab_mean']).reshape(1, 3, 1, 1)
                lab_std = np.array(lab_stats['lab_std']).reshape(1, 3, 1, 1)
                recon_lab = reconstruction.cpu().numpy() * lab_std + lab_mean

                # Convert to RGB
                recon_lab_hwc = recon_lab[0].transpose(1, 2, 0).astype(np.float32)
                recon_lab_cv2 = np.clip(recon_lab_hwc, 0, 255).astype(np.uint8)
                recon_bgr = cv2.cvtColor(recon_lab_cv2, cv2.COLOR_LAB2BGR)
                recon_rgb = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2RGB)

                # Plot
                axes[i, j].imshow(recon_rgb)
                axes[i, j].set_title(f'α={alpha:.2f}', fontsize=10)
                axes[i, j].axis('off')

            # Plot original images using same conversion process
            # Image 1
            target1 = sample1['target'].unsqueeze(0).numpy()
            target1_lab = target1 * np.array(lab_stats['lab_std']).reshape(1, 3, 1, 1) + \
                         np.array(lab_stats['lab_mean']).reshape(1, 3, 1, 1)
            target1_hwc = target1_lab[0].transpose(1, 2, 0).astype(np.float32)
            target1_cv2 = np.clip(target1_hwc, 0, 255).astype(np.uint8)
            target1_bgr = cv2.cvtColor(target1_cv2, cv2.COLOR_LAB2BGR)
            target1_rgb = cv2.cvtColor(target1_bgr, cv2.COLOR_BGR2RGB)

            axes[i, 5].imshow(target1_rgb)
            axes[i, 5].set_title('Original 1', fontsize=10)
            axes[i, 5].axis('off')

            # Image 2
            target2 = sample2['target'].unsqueeze(0).numpy()
            target2_lab = target2 * np.array(lab_stats['lab_std']).reshape(1, 3, 1, 1) + \
                         np.array(lab_stats['lab_mean']).reshape(1, 3, 1, 1)
            target2_hwc = target2_lab[0].transpose(1, 2, 0).astype(np.float32)
            target2_cv2 = np.clip(target2_hwc, 0, 255).astype(np.uint8)
            target2_bgr = cv2.cvtColor(target2_cv2, cv2.COLOR_LAB2BGR)
            target2_rgb = cv2.cvtColor(target2_bgr, cv2.COLOR_BGR2RGB)

            axes[i, 6].imshow(target2_rgb)
            axes[i, 6].set_title('Original 2', fontsize=10)
            axes[i, 6].axis('off')

    plt.suptitle('Latent Space Interpolations (Note: Without Skip Connections)', fontsize=14)
    plt.tight_layout()

    # Save figure
    interp_path = config.visualizations_dir / 'latent_interpolations.png'
    plt.savefig(interp_path, dpi=300, bbox_inches='tight')
    print(f"  Saved latent interpolations to {interp_path}")

    plt.close()


def visualize_latent_distributions(embeddings, config):
    """Visualize distribution of latent dimensions."""
    print("\nCreating latent distribution visualizations...")

    # Select subset of dimensions to visualize (e.g., first 16)
    num_dims_to_plot = min(16, embeddings.shape[1])

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(num_dims_to_plot):
        axes[i].hist(embeddings[:, i], bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Dimension {i}', fontsize=10)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.suptitle('Distribution of Latent Dimensions (First 16)', fontsize=14)
    plt.tight_layout()

    # Save figure
    dist_path = config.visualizations_dir / 'latent_distributions.png'
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"  Saved latent distributions to {dist_path}")

    plt.close()

    # Compute statistics
    stats = {
        'mean': np.mean(embeddings, axis=0).tolist(),
        'std': np.std(embeddings, axis=0).tolist(),
        'min': np.min(embeddings, axis=0).tolist(),
        'max': np.max(embeddings, axis=0).tolist()
    }

    return stats


def save_embeddings(embeddings, metadata_list, config):
    """Save embeddings and metadata for later analysis."""
    print("\nSaving embeddings...")

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
    config = VAEConfig()

    print("="*60)
    print("Variational Autoencoder (VAE) Evaluation")
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

    # Create dataset (combine all splits for evaluation)
    all_images = splits['train'] + splits['val'] + splits['test']

    dataset = VAELeafDataset(
        image_metadata_list=all_images,
        config=config,
        lab_stats=lab_stats,
        transform=None
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    print(f"\nEvaluating on {len(dataset)} images")

    # Extract embeddings
    embeddings, metadata_list = extract_embeddings_and_latents(model, dataloader, config)

    # Create latent space visualizations
    umap_coords, tsne_coords, pca_coords = create_latent_space_visualizations(
        embeddings, metadata_list, config
    )

    # Visualize reconstructions
    metrics = visualize_reconstructions(model, dataset, config, lab_stats, num_samples=8)

    # Visualize latent interpolations
    visualize_latent_interpolations(model, dataset, config, lab_stats, num_pairs=4)

    # Visualize latent distributions
    latent_stats = visualize_latent_distributions(embeddings, config)

    # Save embeddings
    save_embeddings(embeddings, metadata_list, config)

    # Save evaluation metrics
    eval_metrics = {
        'reconstruction_mse': metrics['mse'],
        'reconstruction_ssim': metrics['ssim'],
        'latent_statistics': latent_stats,
        'num_samples': len(embeddings),
        'latent_dim': config.latent_dim
    }

    metrics_path = config.logs_dir / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"\nSaved evaluation metrics to {metrics_path}")

    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - Embeddings: {config.embeddings_dir / 'latent_embeddings.npy'}")
    print(f"  - Metadata: {config.embeddings_dir / 'embeddings_metadata.json'}")
    print(f"  - Latent space viz: {config.visualizations_dir / 'latent_space_visualizations.png'}")
    print(f"  - Reconstructions: {config.visualizations_dir / 'reconstructions.png'}")
    print(f"  - Interpolations: {config.visualizations_dir / 'latent_interpolations.png'}")
    print(f"  - Distributions: {config.visualizations_dir / 'latent_distributions.png'}")
    print(f"  - Metrics: {metrics_path}")
    print(f"\nReconstruction Metrics:")
    print(f"  MSE: {metrics['mse']:.2f}")
    print(f"  SSIM: {metrics['ssim']:.3f}")


if __name__ == '__main__':
    main()
