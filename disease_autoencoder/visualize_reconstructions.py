"""
Visualize image reconstructions with side-by-side comparisons.

Usage:
    python -m disease_autoencoder.visualize_reconstructions
    python -m disease_autoencoder.visualize_reconstructions --num_samples 16
    python -m disease_autoencoder.visualize_reconstructions --split test
    python -m disease_autoencoder.visualize_reconstructions --split train --num_samples 12
"""

import argparse
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from disease_autoencoder.config import DiseaseConfig
from disease_autoencoder.model import DiseaseAutoencoder
from disease_autoencoder.dataset import DiseaseLeafDataset


def load_model_and_data(config, split='test'):
    """Load trained model and dataset."""
    # Load model
    checkpoint_path = config.models_dir / 'checkpoint_best.pth'
    if not checkpoint_path.exists():
        print(f"Error: No trained model found at {checkpoint_path}")
        return None, None, None

    model = DiseaseAutoencoder(config)
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()

    print(f"✓ Loaded model from epoch {checkpoint['epoch'] + 1}")

    # Load data
    splits_path = config.logs_dir / 'image_splits.json'
    if not splits_path.exists():
        print("Error: No data splits found")
        return None, None, None

    with open(splits_path, 'r') as f:
        splits = json.load(f)

    with open(config.logs_dir / 'lab_statistics.json', 'r') as f:
        lab_stats = json.load(f)

    # Create dataset
    if split not in splits:
        print(f"Error: Split '{split}' not found. Available: {list(splits.keys())}")
        return None, None, None

    dataset = DiseaseLeafDataset(
        image_metadata_list=splits[split],
        config=config,
        transform=None
    )
    dataset.set_lab_statistics(lab_stats['lab_mean'], lab_stats['lab_std'])

    print(f"✓ Loaded {len(dataset)} images from {split} set")

    return model, dataset, lab_stats


def lab_to_rgb(lab_tensor, lab_stats, mask):
    """Convert LAB tensor to RGB numpy array."""
    # Denormalize
    lab_mean = np.array(lab_stats['lab_mean']).reshape(3, 1, 1)
    lab_std = np.array(lab_stats['lab_std']).reshape(3, 1, 1)

    lab = lab_tensor.numpy() * lab_std + lab_mean
    lab = np.clip(lab.transpose(1, 2, 0), [0, -128, -128], [100, 127, 127])

    # Convert to RGB
    rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Apply mask
    rgb = rgb * mask.numpy()[:, :, np.newaxis]

    return rgb.astype(np.uint8)


def create_reconstruction_grid(model, dataset, lab_stats, config, num_samples=8, save_path=None):
    """Create a grid of original vs reconstructed images."""
    print(f"\nCreating reconstruction grid with {num_samples} samples...")

    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    # Calculate grid dimensions
    n_cols = 4  # Show 4 image pairs per row
    n_rows = (num_samples + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(20, 5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.3, wspace=0.2)

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            metadata = sample['metadata']

            input_tensor = sample['input'].unsqueeze(0).to(config.device)
            target = sample['target']
            mask = sample['mask'][0]

            # Reconstruct
            reconstruction = model(input_tensor).cpu()[0]

            # Convert to RGB
            target_rgb = lab_to_rgb(target, lab_stats, mask)
            recon_rgb = lab_to_rgb(reconstruction, lab_stats, mask)

            # Compute error
            error = np.abs(target_rgb.astype(float) - recon_rgb.astype(float))
            mse = ((target - reconstruction) ** 2 * sample['mask']).sum() / sample['mask'].sum()

            # Create subplot with 2 images side by side
            row = i // n_cols
            col = i % n_cols

            ax = fig.add_subplot(gs[row, col])

            # Create combined image: original | reconstructed
            combined = np.hstack([target_rgb, recon_rgb])

            ax.imshow(combined)
            ax.axis('off')

            # Title with metadata
            genotype_short = metadata['genotype'][:20] + '...' if len(metadata['genotype']) > 20 else metadata['genotype']
            title = f"{genotype_short}\nDev {metadata['device']} | MSE: {mse:.5f}"
            ax.set_title(title, fontsize=9)

            # Add separator line
            height = combined.shape[0]
            width = target_rgb.shape[1]
            ax.plot([width, width], [0, height], 'r-', linewidth=2, alpha=0.7)

    plt.suptitle('Image Reconstruction: Original (Left) vs Reconstructed (Right)',
                 fontsize=16, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved reconstruction grid to {save_path}")

    plt.close()


def create_detailed_comparison(model, dataset, lab_stats, config, num_samples=4, save_path=None):
    """Create detailed comparison with error maps."""
    print(f"\nCreating detailed comparison with {num_samples} samples...")

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            metadata = sample['metadata']

            input_tensor = sample['input'].unsqueeze(0).to(config.device)
            target = sample['target']
            mask = sample['mask'][0]

            # Reconstruct
            reconstruction = model(input_tensor).cpu()[0]

            # Convert to RGB
            target_rgb = lab_to_rgb(target, lab_stats, mask)
            recon_rgb = lab_to_rgb(reconstruction, lab_stats, mask)

            # Compute error
            error = np.abs(target_rgb.astype(float) - recon_rgb.astype(float)) * mask.numpy()[:, :, np.newaxis]
            mse = ((target - reconstruction) ** 2 * sample['mask']).sum() / sample['mask'].sum()

            # Per-channel MAE
            L_mae = (torch.abs(target[0] - reconstruction[0]) * sample['mask'][0]).sum() / sample['mask'][0].sum()
            a_mae = (torch.abs(target[1] - reconstruction[1]) * sample['mask'][0]).sum() / sample['mask'][0].sum()
            b_mae = (torch.abs(target[2] - reconstruction[2]) * sample['mask'][0]).sum() / sample['mask'][0].sum()

            # Plot
            axes[i, 0].imshow(target_rgb)
            axes[i, 0].set_title(f'Original\n{metadata["genotype"][:25]}', fontsize=10)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(recon_rgb)
            axes[i, 1].set_title(f'Reconstructed\nMSE: {mse:.5f}', fontsize=10)
            axes[i, 1].axis('off')

            axes[i, 2].imshow(error.astype(np.uint8))
            axes[i, 2].set_title('Absolute Error', fontsize=10)
            axes[i, 2].axis('off')

            # Error heatmap (mean across channels)
            error_mean = error.mean(axis=2)
            im = axes[i, 3].imshow(error_mean, cmap='hot', vmin=0, vmax=50)
            axes[i, 3].set_title(f'Error Heatmap\nL:{L_mae:.3f} a:{a_mae:.3f} b:{b_mae:.3f}',
                                fontsize=10)
            axes[i, 3].axis('off')
            plt.colorbar(im, ax=axes[i, 3], fraction=0.046)

    plt.suptitle('Detailed Reconstruction Analysis',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved detailed comparison to {save_path}")

    plt.close()


def create_best_worst_comparison(model, dataset, lab_stats, config, save_path=None):
    """Show best and worst reconstructions."""
    print("\nFinding best and worst reconstructions...")

    # Compute MSE for all samples
    mse_list = []

    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            input_tensor = sample['input'].unsqueeze(0).to(config.device)
            target = sample['target']

            reconstruction = model(input_tensor).cpu()[0]

            mse = ((target - reconstruction) ** 2 * sample['mask']).sum() / sample['mask'].sum()
            mse_list.append((idx, mse.item()))

    # Sort by MSE
    mse_list.sort(key=lambda x: x[1])

    # Get best 4 and worst 4
    best_indices = [x[0] for x in mse_list[:4]]
    worst_indices = [x[0] for x in mse_list[-4:]]

    print(f"  Best MSE: {mse_list[0][1]:.6f}")
    print(f"  Worst MSE: {mse_list[-1][1]:.6f}")

    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Plot best reconstructions
    with torch.no_grad():
        for i, idx in enumerate(best_indices):
            sample = dataset[idx]
            metadata = sample['metadata']

            input_tensor = sample['input'].unsqueeze(0).to(config.device)
            target = sample['target']
            mask = sample['mask'][0]

            reconstruction = model(input_tensor).cpu()[0]

            target_rgb = lab_to_rgb(target, lab_stats, mask)
            recon_rgb = lab_to_rgb(reconstruction, lab_stats, mask)

            combined = np.hstack([target_rgb, recon_rgb])

            axes[0, i].imshow(combined)
            axes[0, i].axis('off')

            mse = mse_list[i][1]
            axes[0, i].set_title(f'Best #{i+1}\nMSE: {mse:.6f}', fontsize=11,
                                color='green', fontweight='bold')

        # Plot worst reconstructions
        for i, idx in enumerate(worst_indices):
            sample = dataset[idx]
            metadata = sample['metadata']

            input_tensor = sample['input'].unsqueeze(0).to(config.device)
            target = sample['target']
            mask = sample['mask'][0]

            reconstruction = model(input_tensor).cpu()[0]

            target_rgb = lab_to_rgb(target, lab_stats, mask)
            recon_rgb = lab_to_rgb(reconstruction, lab_stats, mask)

            combined = np.hstack([target_rgb, recon_rgb])

            axes[1, i].imshow(combined)
            axes[1, i].axis('off')

            mse = mse_list[-(4-i)][1]
            axes[1, i].set_title(f'Worst #{i+1}\nMSE: {mse:.6f}', fontsize=11,
                                color='red', fontweight='bold')

    plt.suptitle('Best vs Worst Reconstructions\nOriginal (Left) | Reconstructed (Right)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved best/worst comparison to {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize image reconstructions')
    parser.add_argument('--num_samples', type=int, default=8,
                       help='Number of samples to visualize')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    print("="*60)
    print("Disease Autoencoder - Reconstruction Visualization")
    print("="*60)

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = DiseaseConfig()

    # Load model and data
    model, dataset, lab_stats = load_model_and_data(config, args.split)
    if model is None:
        return

    # Create output directory
    output_dir = config.visualizations_dir / 'reconstructions'
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nSaving visualizations to: {output_dir}")

    # Generate visualizations
    create_reconstruction_grid(
        model, dataset, lab_stats, config,
        num_samples=args.num_samples,
        save_path=output_dir / f'reconstruction_grid_{args.split}.png'
    )

    create_detailed_comparison(
        model, dataset, lab_stats, config,
        num_samples=4,
        save_path=output_dir / f'detailed_comparison_{args.split}.png'
    )

    create_best_worst_comparison(
        model, dataset, lab_stats, config,
        save_path=output_dir / f'best_worst_{args.split}.png'
    )

    print("\n" + "="*60)
    print("✓ Reconstruction Visualization Complete!")
    print("="*60)
    print(f"\nFiles created in {output_dir}:")
    print(f"  - reconstruction_grid_{args.split}.png")
    print(f"  - detailed_comparison_{args.split}.png")
    print(f"  - best_worst_{args.split}.png")


if __name__ == '__main__':
    main()
