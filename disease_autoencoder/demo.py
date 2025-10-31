"""
Interactive demo script for disease autoencoder.

Usage:
    # Test on random samples
    python -m disease_autoencoder.demo

    # Test on specific number of samples
    python -m disease_autoencoder.demo --num_samples 10

    # Generate detailed report
    python -m disease_autoencoder.demo --report

    # Compare specific genotypes
    python -m disease_autoencoder.demo --compare --genotype "PI123456"
"""

import argparse
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from disease_autoencoder.config import DiseaseConfig
from disease_autoencoder.model import DiseaseAutoencoder
from disease_autoencoder.dataset import DiseaseLeafDataset
from disease_autoencoder.loss import DiseaseWeightedLoss


def load_model(config):
    """Load trained model."""
    checkpoint_path = config.models_dir / 'checkpoint_best.pth'

    if not checkpoint_path.exists():
        print(f"Error: No trained model found at {checkpoint_path}")
        print("Please train the model first: python -m disease_autoencoder.train")
        return None

    model = DiseaseAutoencoder(config)
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()

    print(f"✓ Loaded model (epoch {checkpoint['epoch'] + 1}, loss: {checkpoint['best_val_loss']:.6f})")

    return model


def load_test_dataset(config):
    """Load test dataset."""
    splits_path = config.logs_dir / 'image_splits.json'
    if not splits_path.exists():
        print("Error: No data splits found. Train the model first.")
        return None, None

    with open(splits_path, 'r') as f:
        splits = json.load(f)

    with open(config.logs_dir / 'lab_statistics.json', 'r') as f:
        lab_stats = json.load(f)

    test_images = splits['test']
    dataset = DiseaseLeafDataset(
        image_metadata_list=test_images,
        config=config,
        transform=None
    )
    dataset.set_lab_statistics(lab_stats['lab_mean'], lab_stats['lab_std'])

    print(f"✓ Loaded {len(dataset)} test images")

    return dataset, lab_stats


def compute_reconstruction_metrics(target, reconstruction, mask):
    """Compute reconstruction quality metrics."""
    mask_bool = mask > 0

    # MSE
    mse = ((target - reconstruction) ** 2 * mask).sum() / mask.sum()

    # MAE
    mae = (torch.abs(target - reconstruction) * mask).sum() / mask.sum()

    # PSNR
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))

    # Per-channel metrics
    metrics = {
        'mse': mse.item(),
        'mae': mae.item(),
        'psnr': psnr.item(),
        'L_mae': (torch.abs(target[0] - reconstruction[0]) * mask[0]).sum().item() / mask[0].sum().item(),
        'a_mae': (torch.abs(target[1] - reconstruction[1]) * mask[0]).sum().item() / mask[0].sum().item(),
        'b_mae': (torch.abs(target[2] - reconstruction[2]) * mask[0]).sum().item() / mask[0].sum().item(),
    }

    return metrics


def visualize_single_reconstruction(model, dataset, lab_stats, idx, save_path):
    """Visualize reconstruction for a single image."""
    sample = dataset[idx]
    metadata = sample['metadata']

    input_tensor = sample['input'].unsqueeze(0).to(model.encoder.bottleneck.conv[0].weight.device)
    target = sample['target']
    mask = sample['mask']

    # Reconstruct
    with torch.no_grad():
        reconstruction = model(input_tensor).cpu()[0]
        embedding = model.encode(input_tensor).cpu()[0]

    # Compute metrics
    metrics = compute_reconstruction_metrics(target, reconstruction, mask)

    # Denormalize
    lab_mean = np.array(lab_stats['lab_mean']).reshape(3, 1, 1)
    lab_std = np.array(lab_stats['lab_std']).reshape(3, 1, 1)

    target_lab = target.numpy() * lab_std + lab_mean
    recon_lab = reconstruction.numpy() * lab_std + lab_mean

    # Convert to RGB
    target_lab_hwc = np.clip(target_lab.transpose(1, 2, 0), [0, -128, -128], [100, 127, 127])
    recon_lab_hwc = np.clip(recon_lab.transpose(1, 2, 0), [0, -128, -128], [100, 127, 127])

    target_rgb = cv2.cvtColor(target_lab_hwc.astype(np.uint8), cv2.COLOR_LAB2RGB)
    recon_rgb = cv2.cvtColor(recon_lab_hwc.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Apply mask
    mask_np = mask[0].numpy()
    target_rgb = target_rgb * mask_np[:, :, np.newaxis]
    recon_rgb = recon_rgb * mask_np[:, :, np.newaxis]

    # Compute error map
    error = np.abs(target_rgb.astype(float) - recon_rgb.astype(float)) * mask_np[:, :, np.newaxis]

    # Plot
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Row 1: Main visualization
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(target_rgb.astype(np.uint8))
    ax1.set_title('Original Image', fontsize=12)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(recon_rgb.astype(np.uint8))
    ax2.set_title('Reconstruction', fontsize=12)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(error.astype(np.uint8))
    ax3.set_title('Absolute Error', fontsize=12)
    ax3.axis('off')

    # Metrics text
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    metrics_text = f"""
Image Info:
  Genotype: {metadata['genotype'][:25]}
  Device: {metadata['device']}
  Strip: {metadata['strip_number']}

Reconstruction Metrics:
  MSE:  {metrics['mse']:.6f}
  MAE:  {metrics['mae']:.6f}
  PSNR: {metrics['psnr']:.2f} dB

Per-Channel MAE:
  L* (lightness): {metrics['L_mae']:.4f}
  a* (green-red): {metrics['a_mae']:.4f}
  b* (blue-yellow): {metrics['b_mae']:.4f}

Embedding Stats:
  Mean: {embedding.mean().item():.4f}
  Std:  {embedding.std().item():.4f}
  L2 norm: {embedding.norm().item():.4f}
    """
    ax4.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center')

    # Row 2: LAB channels
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(target_lab[0] * mask_np, cmap='gray', vmin=0, vmax=100)
    ax5.set_title('Target L*', fontsize=10)
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(target_lab[1] * mask_np, cmap='RdYlGn', vmin=-50, vmax=50)
    ax6.set_title('Target a* (G-R)', fontsize=10)
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(target_lab[2] * mask_np, cmap='coolwarm', vmin=-50, vmax=50)
    ax7.set_title('Target b* (B-Y)', fontsize=10)
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 3])
    error_per_channel = np.abs(target_lab - recon_lab).mean(axis=0) * mask_np
    im = ax8.imshow(error_per_channel, cmap='hot')
    ax8.set_title('Mean Channel Error', fontsize=10)
    ax8.axis('off')
    plt.colorbar(im, ax=ax8, fraction=0.046)

    plt.suptitle(f'Sample {idx} - Reconstruction Analysis', fontsize=14, y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")

    plt.close()

    return metrics


def demo_random_samples(model, dataset, lab_stats, config, num_samples=5):
    """Demo with random samples."""
    print(f"\n{'='*60}")
    print(f"Testing on {num_samples} Random Samples")
    print(f"{'='*60}\n")

    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    output_dir = config.visualizations_dir / 'demo_samples'
    output_dir.mkdir(exist_ok=True, parents=True)

    all_metrics = []

    for i, idx in enumerate(indices):
        print(f"\nSample {i+1}/{num_samples} (Index: {idx})")
        print("-" * 40)

        save_path = output_dir / f'sample_{idx}_reconstruction.png'
        metrics = visualize_single_reconstruction(model, dataset, lab_stats, idx, save_path)

        sample = dataset[idx]
        metadata = sample['metadata']

        print(f"Genotype: {metadata['genotype'][:30]}")
        print(f"Device: {metadata['device']}, Strip: {metadata['strip_number']}")
        print(f"Metrics: MSE={metrics['mse']:.6f}, PSNR={metrics['psnr']:.2f} dB")

        all_metrics.append(metrics)

    # Summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}\n")

    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }

    print(f"Average MSE:  {avg_metrics['mse']:.6f}")
    print(f"Average MAE:  {avg_metrics['mae']:.6f}")
    print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"\nPer-channel MAE:")
    print(f"  L*: {avg_metrics['L_mae']:.4f}")
    print(f"  a*: {avg_metrics['a_mae']:.4f}")
    print(f"  b*: {avg_metrics['b_mae']:.4f}")

    print(f"\n✓ Visualizations saved to: {output_dir}")


def generate_report(model, dataset, lab_stats, config):
    """Generate comprehensive report."""
    print(f"\n{'='*60}")
    print("Generating Comprehensive Report")
    print(f"{'='*60}\n")

    # Compute metrics for all samples
    print("Computing metrics for all test samples...")

    all_metrics = []
    genotype_metrics = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            metadata = sample['metadata']

            input_tensor = sample['input'].unsqueeze(0).to(config.device)
            target = sample['target']
            mask = sample['mask']

            reconstruction = model(input_tensor).cpu()[0]

            metrics = compute_reconstruction_metrics(target, reconstruction, mask)
            metrics['genotype'] = metadata['genotype']
            metrics['device'] = metadata['device']

            all_metrics.append(metrics)
            genotype_metrics[metadata['genotype']].append(metrics)

    # Overall statistics
    print("\nOverall Statistics:")
    print("-" * 40)

    avg_mse = np.mean([m['mse'] for m in all_metrics])
    avg_mae = np.mean([m['mae'] for m in all_metrics])
    avg_psnr = np.mean([m['psnr'] for m in all_metrics])

    std_mse = np.std([m['mse'] for m in all_metrics])
    std_psnr = np.std([m['psnr'] for m in all_metrics])

    print(f"MSE:  {avg_mse:.6f} ± {std_mse:.6f}")
    print(f"MAE:  {avg_mae:.6f}")
    print(f"PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")

    # Per-genotype statistics (top 10)
    print("\nTop 10 Genotypes by Sample Count:")
    print("-" * 40)

    sorted_genotypes = sorted(genotype_metrics.items(), key=lambda x: len(x[1]), reverse=True)[:10]

    for genotype, metrics_list in sorted_genotypes:
        avg_mse_g = np.mean([m['mse'] for m in metrics_list])
        avg_psnr_g = np.mean([m['psnr'] for m in metrics_list])

        print(f"{genotype[:30]:30} | n={len(metrics_list):3d} | MSE={avg_mse_g:.6f} | PSNR={avg_psnr_g:.2f} dB")

    # Create visualizations
    output_dir = config.visualizations_dir / 'report'
    output_dir.mkdir(exist_ok=True, parents=True)

    # Plot 1: Distribution of metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist([m['mse'] for m in all_metrics], bins=50, color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('MSE')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of MSE')
    axes[0, 0].axvline(avg_mse, color='red', linestyle='--', label=f'Mean: {avg_mse:.6f}')
    axes[0, 0].legend()

    axes[0, 1].hist([m['psnr'] for m in all_metrics], bins=50, color='green', alpha=0.7)
    axes[0, 1].set_xlabel('PSNR (dB)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of PSNR')
    axes[0, 1].axvline(avg_psnr, color='red', linestyle='--', label=f'Mean: {avg_psnr:.2f}')
    axes[0, 1].legend()

    # Per-channel errors
    channel_names = ['L*', 'a*', 'b*']
    channel_keys = ['L_mae', 'a_mae', 'b_mae']
    channel_errors = [[m[key] for m in all_metrics] for key in channel_keys]

    axes[1, 0].boxplot(channel_errors, labels=channel_names)
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('Per-Channel MAE Distribution')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Device comparison
    devices = list(set([m['device'] for m in all_metrics]))
    device_psnr = [[m['psnr'] for m in all_metrics if m['device'] == d] for d in devices]

    axes[1, 1].boxplot(device_psnr, labels=[f'D{d}' for d in devices])
    axes[1, 1].set_ylabel('PSNR (dB)')
    axes[1, 1].set_title('PSNR by Device')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    report_path = output_dir / 'metrics_report.png'
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved report to {report_path}")
    plt.close()

    # Save detailed report
    report_json = {
        'overall': {
            'mean_mse': avg_mse,
            'std_mse': std_mse,
            'mean_mae': avg_mae,
            'mean_psnr': avg_psnr,
            'std_psnr': std_psnr,
            'num_samples': len(all_metrics)
        },
        'per_genotype': {
            genotype: {
                'count': len(metrics_list),
                'mean_mse': float(np.mean([m['mse'] for m in metrics_list])),
                'mean_psnr': float(np.mean([m['psnr'] for m in metrics_list]))
            }
            for genotype, metrics_list in sorted_genotypes
        }
    }

    report_json_path = output_dir / 'metrics_report.json'
    with open(report_json_path, 'w') as f:
        json.dump(report_json, f, indent=2)

    print(f"✓ Saved detailed report to {report_json_path}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Disease Autoencoder Demo')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of random samples to test')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive report')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    print("="*60)
    print("Disease Autoencoder - Interactive Demo")
    print("="*60)

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config
    config = DiseaseConfig()

    # Load model
    model = load_model(config)
    if model is None:
        return

    # Load dataset
    dataset, lab_stats = load_test_dataset(config)
    if dataset is None:
        return

    # Run demo or report
    if args.report:
        generate_report(model, dataset, lab_stats, config)
    else:
        demo_random_samples(model, dataset, lab_stats, config, args.num_samples)

    print("\n" + "="*60)
    print("✓ Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
