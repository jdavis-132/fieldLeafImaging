"""
Debug script to analyze VAE training and diagnose issues.

This script:
1. Loads a trained VAE checkpoint
2. Analyzes the latent space distribution
3. Checks for posterior collapse
4. Compares reconstruction quality with regular autoencoder
5. Visualizes latent space statistics
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vae.config import VAEConfig
from src.vae.model import create_model as create_vae_model
from src.vae.dataset import get_dataloaders


def load_vae_checkpoint(checkpoint_path, config):
    """Load VAE model from checkpoint."""
    model = create_vae_model(config)
    # Use weights_only=False for legacy checkpoints (pre-PyTorch 2.6)
    # Only do this if you trust the checkpoint source
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def analyze_latent_space(model, dataloader, device, num_samples=1000):
    """
    Analyze the latent space distribution.

    For a well-trained VAE:
    - mu should be centered around 0
    - log_var should be around 0 (variance ~1)
    - Latent samples should follow N(0, 1)
    """
    all_mu = []
    all_log_var = []
    all_z = []

    print("\nAnalyzing latent space distribution...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            inputs = batch['input'].to(device)

            # Get latent representations
            mu, log_var = model.encode(inputs)

            # Sample z
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std

            all_mu.append(mu.cpu().numpy())
            all_log_var.append(log_var.cpu().numpy())
            all_z.append(z.cpu().numpy())

            # Stop after num_samples
            if batch_idx * dataloader.batch_size >= num_samples:
                break

    # Concatenate all batches
    all_mu = np.concatenate(all_mu, axis=0)[:num_samples]
    all_log_var = np.concatenate(all_log_var, axis=0)[:num_samples]
    all_z = np.concatenate(all_z, axis=0)[:num_samples]

    # Compute statistics
    stats = {
        'mu': {
            'mean': all_mu.mean(axis=0),
            'std': all_mu.std(axis=0),
            'global_mean': all_mu.mean(),
            'global_std': all_mu.std()
        },
        'log_var': {
            'mean': all_log_var.mean(axis=0),
            'std': all_log_var.std(axis=0),
            'global_mean': all_log_var.mean(),
            'global_std': all_log_var.std()
        },
        'z': {
            'mean': all_z.mean(axis=0),
            'std': all_z.std(axis=0),
            'global_mean': all_z.mean(),
            'global_std': all_z.std()
        }
    }

    return stats, all_mu, all_log_var, all_z


def check_posterior_collapse(stats, threshold=0.01):
    """
    Check for posterior collapse.

    Posterior collapse occurs when:
    - log_var becomes very negative (variance → 0)
    - Certain latent dimensions are not being used
    - KL divergence is near zero
    """
    log_var_mean = stats['log_var']['mean']

    # Check for inactive dimensions (very negative log_var)
    inactive_dims = np.sum(log_var_mean < -5.0)
    total_dims = len(log_var_mean)

    # Check variance
    variance = np.exp(log_var_mean)
    low_variance_dims = np.sum(variance < threshold)

    # Compute KL divergence per dimension
    mu = stats['mu']['mean']
    kl_per_dim = -0.5 * (1 + log_var_mean - mu**2 - variance)
    total_kl = kl_per_dim.sum()

    collapse_report = {
        'total_dimensions': total_dims,
        'inactive_dimensions': int(inactive_dims),
        'low_variance_dimensions': int(low_variance_dims),
        'active_ratio': 1.0 - (inactive_dims / total_dims),
        'mean_kl_per_dim': float(kl_per_dim.mean()),
        'total_kl': float(total_kl),
        'min_log_var': float(log_var_mean.min()),
        'max_log_var': float(log_var_mean.max())
    }

    return collapse_report, kl_per_dim


def compute_reconstruction_metrics(model, dataloader, device, num_batches=10):
    """Compute reconstruction metrics."""
    mse_losses = []

    print("\nComputing reconstruction metrics...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            if batch_idx >= num_batches:
                break

            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            # Get reconstruction
            predictions = model(inputs)

            # Compute MSE
            mse = torch.nn.functional.mse_loss(predictions, targets, reduction='mean')
            mse_losses.append(mse.item())

    metrics = {
        'mse_mean': np.mean(mse_losses),
        'mse_std': np.std(mse_losses),
        'mse_min': np.min(mse_losses),
        'mse_max': np.max(mse_losses)
    }

    return metrics


def visualize_latent_distributions(stats, kl_per_dim, output_dir):
    """Create visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Distribution of mu
    ax = axes[0, 0]
    ax.hist(stats['mu']['mean'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='Expected (0)')
    ax.set_xlabel('mu (mean across samples)')
    ax.set_ylabel('Frequency')
    ax.set_title(f"Distribution of mu\nMean: {stats['mu']['global_mean']:.3f}, Std: {stats['mu']['global_std']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Distribution of log_var
    ax = axes[0, 1]
    ax.hist(stats['log_var']['mean'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='Expected (0)')
    ax.set_xlabel('log_var (mean across samples)')
    ax.set_ylabel('Frequency')
    ax.set_title(f"Distribution of log_var\nMean: {stats['log_var']['global_mean']:.3f}, Std: {stats['log_var']['global_std']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Distribution of sampled z
    ax = axes[0, 2]
    ax.hist(stats['z']['mean'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='Expected (0)')
    ax.set_xlabel('z (mean across samples)')
    ax.set_ylabel('Frequency')
    ax.set_title(f"Distribution of z (sampled)\nMean: {stats['z']['global_mean']:.3f}, Std: {stats['z']['global_std']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. KL divergence per dimension
    ax = axes[1, 0]
    ax.bar(range(len(kl_per_dim)), kl_per_dim, alpha=0.7, color='orange')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('KL Divergence')
    ax.set_title(f'KL Divergence per Dimension\nTotal KL: {kl_per_dim.sum():.2f}')
    ax.grid(True, alpha=0.3)

    # 5. Variance per dimension
    ax = axes[1, 1]
    variance = np.exp(stats['log_var']['mean'])
    ax.bar(range(len(variance)), variance, alpha=0.7, color='cyan')
    ax.axhline(1.0, color='red', linestyle='--', label='Expected (1.0)')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Variance (exp(log_var))')
    ax.set_title(f'Variance per Dimension\nMean: {variance.mean():.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Active dimensions analysis
    ax = axes[1, 2]
    active_mask = variance > 0.01
    colors = ['green' if active else 'red' for active in active_mask]
    ax.bar(range(len(active_mask)), active_mask.astype(int), alpha=0.7, color=colors)
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Active (1) / Inactive (0)')
    ax.set_title(f'Active Latent Dimensions\nActive: {active_mask.sum()}/{len(active_mask)} ({100*active_mask.sum()/len(active_mask):.1f}%)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'latent_space_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {output_path}")
    plt.close()


def main():
    """Main debugging function."""
    print("=" * 80)
    print("VAE Training Debug Analysis")
    print("=" * 80)

    # Load config
    config = VAEConfig()

    # Check for checkpoint
    checkpoint_path = config.models_dir / 'checkpoint_best.pth'
    if not checkpoint_path.exists():
        print(f"Error: No checkpoint found at {checkpoint_path}")
        print("Please train the VAE first.")
        return

    print(f"\nLoading checkpoint from {checkpoint_path}...")

    # Load model
    model, checkpoint = load_vae_checkpoint(checkpoint_path, config)
    print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")

    # Load data
    print("\nLoading data...")
    with open(config.existing_splits_path, 'r') as f:
        splits = json.load(f)

    with open(config.existing_lab_stats_path, 'r') as f:
        lab_stats = json.load(f)

    dataloaders = get_dataloaders(splits, lab_stats, config)

    # Analyze latent space
    stats, all_mu, all_log_var, all_z = analyze_latent_space(
        model, dataloaders['val'], config.device, num_samples=1000
    )

    print("\n" + "=" * 80)
    print("LATENT SPACE STATISTICS")
    print("=" * 80)
    print(f"\nMu (mean of encoder):")
    print(f"  Global mean: {stats['mu']['global_mean']:.4f} (should be ≈ 0)")
    print(f"  Global std:  {stats['mu']['global_std']:.4f}")

    print(f"\nLog-variance (log of variance):")
    print(f"  Global mean: {stats['log_var']['global_mean']:.4f} (should be ≈ 0)")
    print(f"  Global std:  {stats['log_var']['global_std']:.4f}")

    print(f"\nSampled z:")
    print(f"  Global mean: {stats['z']['global_mean']:.4f} (should be ≈ 0)")
    print(f"  Global std:  {stats['z']['global_std']:.4f} (should be ≈ 1)")

    # Check for posterior collapse
    collapse_report, kl_per_dim = check_posterior_collapse(stats)

    print("\n" + "=" * 80)
    print("POSTERIOR COLLAPSE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal latent dimensions: {collapse_report['total_dimensions']}")
    print(f"Inactive dimensions (log_var < -5): {collapse_report['inactive_dimensions']}")
    print(f"Low variance dimensions (var < 0.01): {collapse_report['low_variance_dimensions']}")
    print(f"Active dimension ratio: {collapse_report['active_ratio']:.2%}")
    print(f"\nKL Divergence:")
    print(f"  Total KL: {collapse_report['total_kl']:.2f} (should be 10-100 for 128D)")
    print(f"  Mean KL per dim: {collapse_report['mean_kl_per_dim']:.4f}")
    print(f"  Log_var range: [{collapse_report['min_log_var']:.2f}, {collapse_report['max_log_var']:.2f}]")

    # Interpret results
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if collapse_report['total_kl'] < 5.0:
        print("\n⚠ WARNING: Severe posterior collapse detected!")
        print("  → KL divergence is very low, model is ignoring the latent space")
    elif collapse_report['total_kl'] < 20.0:
        print("\n⚠ WARNING: Moderate posterior collapse detected")
        print("  → KL divergence is lower than expected for 128D latent space")
    else:
        print("\n✓ KL divergence looks reasonable")

    if collapse_report['active_ratio'] < 0.5:
        print("\n⚠ WARNING: Many inactive latent dimensions!")
        print(f"  → Only {collapse_report['active_ratio']:.1%} of dimensions are being used")
    elif collapse_report['active_ratio'] < 0.8:
        print("\n⚠ Some latent dimensions are underutilized")
    else:
        print("\n✓ Most latent dimensions are active")

    if abs(stats['z']['global_mean']) > 0.5:
        print(f"\n⚠ WARNING: Latent samples are not centered (mean = {stats['z']['global_mean']:.3f})")
    else:
        print("\n✓ Latent samples are well-centered")

    if abs(stats['z']['global_std'] - 1.0) > 0.3:
        print(f"\n⚠ WARNING: Latent sample std dev is off (std = {stats['z']['global_std']:.3f}, should be ≈ 1.0)")
    else:
        print("\n✓ Latent sample variance looks good")

    # Reconstruction metrics
    recon_metrics = compute_reconstruction_metrics(model, dataloaders['val'], config.device)

    print("\n" + "=" * 80)
    print("RECONSTRUCTION METRICS")
    print("=" * 80)
    print(f"\nMSE Loss:")
    print(f"  Mean: {recon_metrics['mse_mean']:.6f}")
    print(f"  Std:  {recon_metrics['mse_std']:.6f}")
    print(f"  Min:  {recon_metrics['mse_min']:.6f}")
    print(f"  Max:  {recon_metrics['mse_max']:.6f}")

    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    visualize_latent_distributions(stats, kl_per_dim, config.logs_dir / 'debug')

    # Save detailed report
    report = {
        'checkpoint': {
            'epoch': checkpoint['epoch'] + 1,
            'best_val_loss': checkpoint['best_val_loss']
        },
        'latent_stats': {
            'mu_mean': float(stats['mu']['global_mean']),
            'mu_std': float(stats['mu']['global_std']),
            'log_var_mean': float(stats['log_var']['global_mean']),
            'log_var_std': float(stats['log_var']['global_std']),
            'z_mean': float(stats['z']['global_mean']),
            'z_std': float(stats['z']['global_std'])
        },
        'posterior_collapse': collapse_report,
        'reconstruction': recon_metrics
    }

    report_path = config.logs_dir / 'debug' / 'debug_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved detailed report to {report_path}")

    print("\n" + "=" * 80)
    print("DEBUG ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
