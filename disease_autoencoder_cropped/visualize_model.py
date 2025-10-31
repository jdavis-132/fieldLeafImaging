"""
Advanced visualization script for disease autoencoder model.

Visualizations:
- Attention maps at bottleneck
- Disease weight heatmaps
- Feature maps from encoder/decoder
- Latent space analysis (PCA, t-SNE)
- Per-channel LAB reconstruction
- Error heatmaps
- Reconstruction quality comparison
- Embedding clustering visualization
"""

import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from disease_autoencoder.config import DiseaseConfig
from disease_autoencoder.model import DiseaseAutoencoder
from disease_autoencoder.dataset import DiseaseLeafDataset
from disease_autoencoder.loss import DiseaseWeightedLoss


def load_trained_model(config, checkpoint_name='checkpoint_best.pth'):
    """Load trained model from checkpoint."""
    checkpoint_path = config.models_dir / checkpoint_name

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return None

    print(f"Loading model from {checkpoint_path}")

    model = DiseaseAutoencoder(config)
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()

    print(f"✓ Loaded model from epoch {checkpoint['epoch'] + 1}")

    return model


def load_sample_data(config, num_samples=8):
    """Load some sample images for visualization."""
    # Load splits
    splits_path = config.logs_dir / 'image_splits.json'
    if not splits_path.exists():
        print("Error: No splits found. Train the model first.")
        return None, None

    with open(splits_path, 'r') as f:
        splits = json.load(f)

    # Load LAB stats
    with open(config.logs_dir / 'lab_statistics.json', 'r') as f:
        lab_stats = json.load(f)

    # Create dataset from test set
    test_images = splits['test'][:num_samples]

    dataset = DiseaseLeafDataset(
        image_metadata_list=test_images,
        config=config,
        transform=None
    )
    dataset.set_lab_statistics(lab_stats['lab_mean'], lab_stats['lab_std'])

    return dataset, lab_stats


# ============================================================================
# 1. Attention Map Visualization
# ============================================================================

def visualize_attention_maps(model, dataset, config, save_dir):
    """Visualize attention maps from the bottleneck."""
    print("\n" + "="*60)
    print("1. Visualizing Attention Maps")
    print("="*60)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Hook to capture attention output
    attention_output = {}

    def hook_fn(module, input, output):
        attention_output['output'] = output.detach().cpu()

    # Register hook on attention module
    if model.attention is not None:
        handle = model.attention.register_forward_hook(hook_fn)
    else:
        print("Model has no attention module")
        return

    num_samples = min(4, len(dataset))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]
            input_tensor = sample['input'].unsqueeze(0).to(config.device)

            # Forward pass to trigger hook
            reconstruction = model(input_tensor)

            # Get original image
            target = sample['target']
            mask = sample['mask'][0]

            # Convert LAB to RGB for visualization
            with open(config.logs_dir / 'lab_statistics.json', 'r') as f:
                lab_stats = json.load(f)

            lab_mean = np.array(lab_stats['lab_mean']).reshape(3, 1, 1)
            lab_std = np.array(lab_stats['lab_std']).reshape(3, 1, 1)

            target_lab = (target.numpy() * lab_std + lab_mean).transpose(1, 2, 0).astype(np.float32)
            target_lab = np.clip(target_lab, [0, -128, -128], [100, 127, 127])
            target_lab_uint8 = target_lab.astype(np.uint8)
            target_rgb = cv2.cvtColor(target_lab_uint8, cv2.COLOR_LAB2RGB)
            target_rgb = target_rgb * mask.numpy()[:, :, np.newaxis]

            # Get attention map (average over channels)
            if 'output' in attention_output:
                attn_map = attention_output['output'][0].mean(dim=0).numpy()
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

                # Resize attention map to match image size
                attn_map_resized = cv2.resize(attn_map, (config.image_size, config.image_size))

                # Plot
                axes[i, 0].imshow(target_rgb.astype(np.uint8))
                axes[i, 0].set_title(f'Sample {i+1}: Original')
                axes[i, 0].axis('off')

                axes[i, 1].imshow(attn_map_resized, cmap='hot')
                axes[i, 1].set_title('Attention Map')
                axes[i, 1].axis('off')

                # Overlay attention on image
                attn_overlay = target_rgb.copy().astype(float)
                attn_heatmap = plt.cm.hot(attn_map_resized)[:, :, :3] * 255
                attn_overlay = 0.6 * attn_overlay + 0.4 * attn_heatmap
                attn_overlay = np.clip(attn_overlay, 0, 255).astype(np.uint8)

                axes[i, 2].imshow(attn_overlay)
                axes[i, 2].set_title('Attention Overlay')
                axes[i, 2].axis('off')

    handle.remove()

    plt.tight_layout()
    save_path = save_dir / 'attention_maps.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved attention maps to {save_path}")
    plt.close()


# ============================================================================
# 2. Disease Weight Visualization
# ============================================================================

def visualize_disease_weights(model, dataset, config, save_dir):
    """Visualize disease weight heatmaps."""
    print("\n" + "="*60)
    print("2. Visualizing Disease Weights")
    print("="*60)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Create loss function
    loss_fn = DiseaseWeightedLoss(config)

    # Load LAB stats
    with open(config.logs_dir / 'lab_statistics.json', 'r') as f:
        lab_stats = json.load(f)

    num_samples = min(6, len(dataset))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]
            target = sample['target'].unsqueeze(0)
            mask = sample['mask'].unsqueeze(0)

            # Compute disease weights
            weights = loss_fn.compute_disease_weights(target, mask, lab_stats['lab_mean'], lab_stats['lab_std'])
            weight_map = weights[0, 0].numpy()

            # Get original image
            lab_mean = np.array(lab_stats['lab_mean']).reshape(3, 1, 1)
            lab_std = np.array(lab_stats['lab_std']).reshape(3, 1, 1)

            target_lab = (target[0].numpy() * lab_std + lab_mean).transpose(1, 2, 0).astype(np.float32)
            target_lab = np.clip(target_lab, [0, -128, -128], [100, 127, 127])
            target_lab_uint8 = target_lab.astype(np.uint8)
            target_rgb = cv2.cvtColor(target_lab_uint8, cv2.COLOR_LAB2RGB)
            target_rgb = target_rgb * mask[0, 0].numpy()[:, :, np.newaxis]

            # Plot
            axes[i, 0].imshow(target_rgb.astype(np.uint8))
            axes[i, 0].set_title(f'Sample {i+1}')
            axes[i, 0].axis('off')

            im = axes[i, 1].imshow(weight_map, cmap='RdYlGn_r', vmin=1.0, vmax=config.disease_weight_strength)
            axes[i, 1].set_title(f'Disease Weights (1.0-{config.disease_weight_strength:.1f})')
            axes[i, 1].axis('off')
            plt.colorbar(im, ax=axes[i, 1], fraction=0.046)

            # Overlay
            weight_overlay = target_rgb.copy().astype(float)
            weight_heatmap = plt.cm.RdYlGn_r((weight_map - 1.0) / (config.disease_weight_strength - 1.0))[:, :, :3] * 255
            weight_overlay = 0.6 * weight_overlay + 0.4 * weight_heatmap
            weight_overlay = np.clip(weight_overlay, 0, 255).astype(np.uint8)

            axes[i, 2].imshow(weight_overlay)
            axes[i, 2].set_title('Weight Overlay (Red=Diseased)')
            axes[i, 2].axis('off')

    plt.tight_layout()
    save_path = save_dir / 'disease_weights.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved disease weights to {save_path}")
    plt.close()


# ============================================================================
# 3. Per-Channel LAB Reconstruction
# ============================================================================

def visualize_lab_channels(model, dataset, config, save_dir):
    """Visualize per-channel LAB reconstruction."""
    print("\n" + "="*60)
    print("3. Visualizing LAB Channel Reconstruction")
    print("="*60)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Load LAB stats
    with open(config.logs_dir / 'lab_statistics.json', 'r') as f:
        lab_stats = json.load(f)

    num_samples = min(3, len(dataset))
    fig, axes = plt.subplots(num_samples, 7, figsize=(21, 3 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]
            input_tensor = sample['input'].unsqueeze(0).to(config.device)
            target = sample['target']
            mask = sample['mask'][0]

            # Reconstruct
            reconstruction = model(input_tensor).cpu()[0]

            # Denormalize
            lab_mean = np.array(lab_stats['lab_mean']).reshape(3, 1, 1)
            lab_std = np.array(lab_stats['lab_std']).reshape(3, 1, 1)

            target_lab = target.numpy() * lab_std + lab_mean
            recon_lab = reconstruction.numpy() * lab_std + lab_mean

            # Convert to RGB
            target_lab_hwc = target_lab.transpose(1, 2, 0).astype(np.float32)
            recon_lab_hwc = recon_lab.transpose(1, 2, 0).astype(np.float32)

            target_lab_hwc = np.clip(target_lab_hwc, [0, -128, -128], [100, 127, 127])
            recon_lab_hwc = np.clip(recon_lab_hwc, [0, -128, -128], [100, 127, 127])

            target_rgb = cv2.cvtColor(target_lab_hwc.astype(np.uint8), cv2.COLOR_LAB2RGB)
            recon_rgb = cv2.cvtColor(recon_lab_hwc.astype(np.uint8), cv2.COLOR_LAB2RGB)

            # Apply mask
            mask_np = mask.numpy()
            target_rgb = target_rgb * mask_np[:, :, np.newaxis]
            recon_rgb = recon_rgb * mask_np[:, :, np.newaxis]

            # Plot RGB
            axes[i, 0].imshow(target_rgb.astype(np.uint8))
            axes[i, 0].set_title(f'Sample {i+1}: Target RGB')
            axes[i, 0].axis('off')

            # Plot L channel
            axes[i, 1].imshow(target_lab[0] * mask_np, cmap='gray', vmin=0, vmax=100)
            axes[i, 1].set_title('Target L*')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(recon_lab[0] * mask_np, cmap='gray', vmin=0, vmax=100)
            axes[i, 2].set_title('Recon L*')
            axes[i, 2].axis('off')

            # Plot a* channel
            axes[i, 3].imshow(target_lab[1] * mask_np, cmap='RdYlGn', vmin=-50, vmax=50)
            axes[i, 3].set_title('Target a* (G-R)')
            axes[i, 3].axis('off')

            axes[i, 4].imshow(recon_lab[1] * mask_np, cmap='RdYlGn', vmin=-50, vmax=50)
            axes[i, 4].set_title('Recon a* (G-R)')
            axes[i, 4].axis('off')

            # Plot b* channel
            axes[i, 5].imshow(target_lab[2] * mask_np, cmap='coolwarm', vmin=-50, vmax=50)
            axes[i, 5].set_title('Target b* (B-Y)')
            axes[i, 5].axis('off')

            axes[i, 6].imshow(recon_lab[2] * mask_np, cmap='coolwarm', vmin=-50, vmax=50)
            axes[i, 6].set_title('Recon b* (B-Y)')
            axes[i, 6].axis('off')

    plt.tight_layout()
    save_path = save_dir / 'lab_channels.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved LAB channels to {save_path}")
    plt.close()


# ============================================================================
# 4. Error Heatmaps
# ============================================================================

def visualize_error_heatmaps(model, dataset, config, save_dir):
    """Visualize reconstruction error heatmaps."""
    print("\n" + "="*60)
    print("4. Visualizing Error Heatmaps")
    print("="*60)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Load LAB stats
    with open(config.logs_dir / 'lab_statistics.json', 'r') as f:
        lab_stats = json.load(f)

    num_samples = min(6, len(dataset))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]
            input_tensor = sample['input'].unsqueeze(0).to(config.device)
            target = sample['target']
            mask = sample['mask'][0]

            # Reconstruct
            reconstruction = model(input_tensor).cpu()[0]

            # Compute error
            error = torch.abs(target - reconstruction)

            # Denormalize
            lab_mean = np.array(lab_stats['lab_mean']).reshape(3, 1, 1)
            lab_std = np.array(lab_stats['lab_std']).reshape(3, 1, 1)

            target_lab = target.numpy() * lab_std + lab_mean
            target_lab = np.clip(target_lab.transpose(1, 2, 0), [0, -128, -128], [100, 127, 127])
            target_rgb = cv2.cvtColor(target_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            target_rgb = target_rgb * mask.numpy()[:, :, np.newaxis]

            # Plot
            axes[i, 0].imshow(target_rgb.astype(np.uint8))
            axes[i, 0].set_title(f'Sample {i+1}')
            axes[i, 0].axis('off')

            # L* error
            im1 = axes[i, 1].imshow(error[0].numpy() * mask.numpy(), cmap='hot', vmin=0, vmax=1)
            axes[i, 1].set_title('L* Error')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

            # a* error
            im2 = axes[i, 2].imshow(error[1].numpy() * mask.numpy(), cmap='hot', vmin=0, vmax=1)
            axes[i, 2].set_title('a* Error')
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

            # b* error
            im3 = axes[i, 3].imshow(error[2].numpy() * mask.numpy(), cmap='hot', vmin=0, vmax=1)
            axes[i, 3].set_title('b* Error')
            axes[i, 3].axis('off')
            plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)

    plt.tight_layout()
    save_path = save_dir / 'error_heatmaps.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved error heatmaps to {save_path}")
    plt.close()


# ============================================================================
# 5. Latent Space Analysis
# ============================================================================

def visualize_latent_space(model, config, save_dir):
    """Visualize latent space with PCA and t-SNE."""
    print("\n" + "="*60)
    print("5. Visualizing Latent Space (PCA/t-SNE)")
    print("="*60)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Check if embeddings exist
    embeddings_path = config.embeddings_dir / 'latent_embeddings.npy'
    metadata_path = config.embeddings_dir / 'embeddings_metadata.json'

    if not embeddings_path.exists():
        print("⚠ Embeddings not found. Run evaluate.py first.")
        return

    # Load embeddings
    embeddings = np.load(embeddings_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(embeddings)

    # t-SNE
    from sklearn.manifold import TSNE
    print("Computing t-SNE... (this may take a while)")
    tsne = TSNE(n_components=2, random_state=config.random_seed, perplexity=30)
    tsne_coords = tsne.fit_transform(embeddings)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PCA plot
    genotypes = [m['genotype'] for m in metadata]
    from collections import Counter
    genotype_counts = Counter(genotypes)
    top_genotypes = [g for g, _ in genotype_counts.most_common(10)]

    for genotype in top_genotypes:
        mask = np.array([g == genotype for g in genotypes])
        axes[0].scatter(pca_coords[mask, 0], pca_coords[mask, 1], label=genotype, alpha=0.6, s=20)

    axes[0].set_title(f'PCA (explained var: {pca.explained_variance_ratio_.sum():.2%})', fontsize=14)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # t-SNE plot
    devices = [m['device'] for m in metadata]
    scatter = axes[1].scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=devices, cmap='tab10', alpha=0.6, s=20)
    axes[1].set_title('t-SNE (colored by device)', fontsize=14)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[1], label='Device')

    plt.tight_layout()
    save_path = save_dir / 'latent_space_pca_tsne.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved latent space visualization to {save_path}")
    plt.close()


# ============================================================================
# 6. Feature Map Visualization
# ============================================================================

def visualize_feature_maps(model, dataset, config, save_dir):
    """Visualize intermediate feature maps from encoder."""
    print("\n" + "="*60)
    print("6. Visualizing Feature Maps")
    print("="*60)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Hook to capture encoder outputs
    encoder_outputs = []

    def hook_fn(module, input, output):
        encoder_outputs.append(output.detach().cpu())

    # Register hooks on encoder blocks
    handles = []
    for i, encoder_block in enumerate(model.encoder.encoders):
        handle = encoder_block.register_forward_hook(hook_fn)
        handles.append(handle)

    # Get a sample
    sample = dataset[0]
    input_tensor = sample['input'].unsqueeze(0).to(config.device)

    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    # Plot feature maps from each encoder level
    num_levels = len(encoder_outputs)
    fig, axes = plt.subplots(num_levels, 8, figsize=(16, 2 * num_levels))

    if num_levels == 1:
        axes = axes[np.newaxis, :]

    for level_idx, features in enumerate(encoder_outputs):
        features = features[0]  # Remove batch dimension
        num_channels = features.shape[0]

        # Show first 8 channels
        for ch_idx in range(min(8, num_channels)):
            feature_map = features[ch_idx].numpy()

            axes[level_idx, ch_idx].imshow(feature_map, cmap='viridis')
            axes[level_idx, ch_idx].axis('off')

            if ch_idx == 0:
                axes[level_idx, ch_idx].set_title(f'Level {level_idx+1}\nCh {ch_idx+1}', fontsize=9)
            else:
                axes[level_idx, ch_idx].set_title(f'Ch {ch_idx+1}', fontsize=9)

    plt.suptitle('Encoder Feature Maps (First 8 Channels per Level)', fontsize=14)
    plt.tight_layout()

    save_path = save_dir / 'feature_maps.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature maps to {save_path}")
    plt.close()

    # Remove hooks
    for handle in handles:
        handle.remove()


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run all visualizations."""
    print("="*60)
    print("Disease Autoencoder - Advanced Visualization")
    print("="*60)

    config = DiseaseConfig()

    # Load trained model
    model = load_trained_model(config)
    if model is None:
        return

    # Load sample data
    dataset, lab_stats = load_sample_data(config, num_samples=8)
    if dataset is None:
        return

    # Create visualization directory
    viz_dir = config.visualizations_dir / 'advanced'
    viz_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nSaving visualizations to: {viz_dir}")

    # Run all visualizations
    try:
        visualize_attention_maps(model, dataset, config, viz_dir)
    except Exception as e:
        print(f"⚠ Attention visualization failed: {e}")

    try:
        visualize_disease_weights(model, dataset, config, viz_dir)
    except Exception as e:
        print(f"⚠ Disease weights visualization failed: {e}")

    try:
        visualize_lab_channels(model, dataset, config, viz_dir)
    except Exception as e:
        print(f"⚠ LAB channels visualization failed: {e}")

    try:
        visualize_error_heatmaps(model, dataset, config, viz_dir)
    except Exception as e:
        print(f"⚠ Error heatmaps visualization failed: {e}")

    try:
        visualize_latent_space(model, config, viz_dir)
    except Exception as e:
        print(f"⚠ Latent space visualization failed: {e}")

    try:
        visualize_feature_maps(model, dataset, config, viz_dir)
    except Exception as e:
        print(f"⚠ Feature maps visualization failed: {e}")

    print("\n" + "="*60)
    print("✓ Visualization Complete!")
    print("="*60)
    print(f"\nAll visualizations saved to: {viz_dir}")


if __name__ == '__main__':
    main()
