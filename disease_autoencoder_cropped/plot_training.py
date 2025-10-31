"""
Plot training and validation loss curves from training history.

Usage:
    python -m disease_autoencoder.plot_training
    python -m disease_autoencoder.plot_training --detailed  # Include all loss components
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from disease_autoencoder.config import DiseaseConfig


def load_training_history(config):
    """Load training history from JSON file."""
    history_path = config.logs_dir / 'training_history.json'

    if not history_path.exists():
        print(f"Error: Training history not found at {history_path}")
        print("Please train the model first: python -m disease_autoencoder.train")
        return None

    with open(history_path, 'r') as f:
        history = json.load(f)

    print(f"✓ Loaded training history")
    print(f"  Total epochs: {history['total_epochs']}")
    print(f"  Training time: {history['training_time_hours']:.2f} hours")
    print(f"  Best validation loss: {history['best_val_loss']:.6f}")

    return history


def plot_loss_curves(history, config, save_path):
    """Plot training and validation loss curves."""
    print("\nPlotting loss curves...")

    # Extract data
    epochs = np.arange(1, len(history['train_losses']) + 1)

    train_total = [x['total_loss'] for x in history['train_losses']]
    val_total = [x['total_loss'] for x in history['val_losses']]

    train_recon = [x['reconstruction_loss'] for x in history['train_losses']]
    val_recon = [x['reconstruction_loss'] for x in history['val_losses']]

    # Find best epoch
    best_epoch = np.argmin(val_total) + 1
    best_val_loss = min(val_total)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Total Loss
    axes[0, 0].plot(epochs, train_total, label='Training', linewidth=2, alpha=0.8)
    axes[0, 0].plot(epochs, val_total, label='Validation', linewidth=2, alpha=0.8)
    axes[0, 0].axvline(best_epoch, color='red', linestyle='--', alpha=0.5,
                       label=f'Best Epoch ({best_epoch})')
    axes[0, 0].scatter([best_epoch], [best_val_loss], color='red', s=100,
                       zorder=5, marker='*')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Total Loss', fontsize=12)
    axes[0, 0].set_title('Total Loss vs Epoch', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Reconstruction Loss
    axes[0, 1].plot(epochs, train_recon, label='Training', linewidth=2, alpha=0.8)
    axes[0, 1].plot(epochs, val_recon, label='Validation', linewidth=2, alpha=0.8)
    axes[0, 1].axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Reconstruction Loss', fontsize=12)
    axes[0, 1].set_title('Reconstruction Loss vs Epoch', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Log Scale Total Loss
    axes[1, 0].semilogy(epochs, train_total, label='Training', linewidth=2, alpha=0.8)
    axes[1, 0].semilogy(epochs, val_total, label='Validation', linewidth=2, alpha=0.8)
    axes[1, 0].axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Total Loss (log scale)', fontsize=12)
    axes[1, 0].set_title('Total Loss vs Epoch (Log Scale)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, which='both')

    # Plot 4: Loss Difference (Overfitting Detection)
    loss_diff = np.array(val_total) - np.array(train_total)
    axes[1, 1].plot(epochs, loss_diff, linewidth=2, color='purple', alpha=0.8)
    axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].fill_between(epochs, 0, loss_diff, where=(loss_diff > 0),
                            alpha=0.3, color='red', label='Overfitting')
    axes[1, 1].fill_between(epochs, 0, loss_diff, where=(loss_diff <= 0),
                            alpha=0.3, color='green', label='Underfitting')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Val Loss - Train Loss', fontsize=12)
    axes[1, 1].set_title('Generalization Gap', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Disease Autoencoder Training History\nBest Val Loss: {best_val_loss:.6f} at Epoch {best_epoch}',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved loss curves to {save_path}")
    plt.close()


def plot_detailed_loss_components(history, config, save_path):
    """Plot detailed loss components."""
    print("\nPlotting detailed loss components...")

    epochs = np.arange(1, len(history['train_losses']) + 1)

    # Extract all components
    train_total = [x['total_loss'] for x in history['train_losses']]
    val_total = [x['total_loss'] for x in history['val_losses']]

    train_recon = [x['reconstruction_loss'] for x in history['train_losses']]
    val_recon = [x['reconstruction_loss'] for x in history['val_losses']]

    train_l1 = [x['l1_loss'] for x in history['train_losses']]
    val_l1 = [x['l1_loss'] for x in history['val_losses']]

    train_weight = [x['mean_weight'] for x in history['train_losses']]
    val_weight = [x['mean_weight'] for x in history['val_losses']]

    best_epoch = np.argmin(val_total) + 1

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: All losses stacked
    axes[0, 0].plot(epochs, train_total, label='Total (Train)', linewidth=2)
    axes[0, 0].plot(epochs, val_total, label='Total (Val)', linewidth=2)
    axes[0, 0].plot(epochs, train_recon, label='Recon (Train)', linewidth=1,
                    linestyle='--', alpha=0.7)
    axes[0, 0].plot(epochs, val_recon, label='Recon (Val)', linewidth=1,
                    linestyle='--', alpha=0.7)
    axes[0, 0].axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss Components Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: L1 Regularization
    axes[0, 1].plot(epochs, train_l1, label='Training', linewidth=2, alpha=0.8)
    axes[0, 1].plot(epochs, val_l1, label='Validation', linewidth=2, alpha=0.8)
    axes[0, 1].axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('L1 Loss', fontsize=12)
    axes[0, 1].set_title('L1 Regularization Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Plot 3: Mean Disease Weight
    axes[1, 0].plot(epochs, train_weight, label='Training', linewidth=2, alpha=0.8)
    axes[1, 0].plot(epochs, val_weight, label='Validation', linewidth=2, alpha=0.8)
    axes[1, 0].axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(config.disease_weight_strength, color='gray', linestyle=':',
                       label=f'Max Weight ({config.disease_weight_strength})')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Mean Disease Weight', fontsize=12)
    axes[1, 0].set_title('Disease Weighting Over Time', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([1.0, config.disease_weight_strength + 0.1])

    # Plot 4: Loss improvement rate (derivative)
    val_improvement = -np.diff(val_total)  # Negative because we want decrease
    smoothed_improvement = np.convolve(val_improvement, np.ones(5)/5, mode='valid')

    axes[1, 1].plot(epochs[:-1], val_improvement, alpha=0.3, linewidth=1,
                    label='Raw', color='blue')
    axes[1, 1].plot(epochs[2:-2], smoothed_improvement, linewidth=2,
                    label='Smoothed (5-epoch)', color='blue')
    axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss Improvement', fontsize=12)
    axes[1, 1].set_title('Validation Loss Improvement Rate', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Detailed Loss Component Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved detailed components to {save_path}")
    plt.close()


def plot_training_summary(history, config, save_path):
    """Create a summary visualization with key metrics."""
    print("\nCreating training summary...")

    epochs = np.arange(1, len(history['train_losses']) + 1)
    val_total = [x['total_loss'] for x in history['val_losses']]
    train_total = [x['total_loss'] for x in history['train_losses']]

    best_epoch = np.argmin(val_total) + 1
    best_val_loss = min(val_total)
    final_train_loss = train_total[-1]
    final_val_loss = val_total[-1]

    # Calculate statistics
    epochs_to_best = best_epoch
    final_gap = final_val_loss - final_train_loss
    improvement = (val_total[0] - best_val_loss) / val_total[0] * 100

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main plot: Loss curves
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    ax_main.plot(epochs, train_total, label='Training', linewidth=2.5, alpha=0.8)
    ax_main.plot(epochs, val_total, label='Validation', linewidth=2.5, alpha=0.8)
    ax_main.axvline(best_epoch, color='red', linestyle='--', alpha=0.5,
                    label=f'Best Epoch')
    ax_main.scatter([best_epoch], [best_val_loss], color='red', s=200,
                    zorder=5, marker='*', edgecolors='darkred', linewidths=2)
    ax_main.set_xlabel('Epoch', fontsize=14)
    ax_main.set_ylabel('Total Loss', fontsize=14)
    ax_main.set_title('Training Progress', fontsize=16, fontweight='bold')
    ax_main.legend(fontsize=12, loc='upper right')
    ax_main.grid(True, alpha=0.3)

    # Stats panel
    ax_stats = fig.add_subplot(gs[0:2, 2])
    ax_stats.axis('off')

    stats_text = f"""
TRAINING SUMMARY
{'='*35}

Total Epochs:      {history['total_epochs']}
Training Time:     {history['training_time_hours']:.2f} hours
Time per Epoch:    {history['training_time_hours']/history['total_epochs']*60:.1f} min

BEST MODEL
{'='*35}

Best Epoch:        {best_epoch}
Best Val Loss:     {best_val_loss:.6f}
Train Loss:        {train_total[best_epoch-1]:.6f}
Gap:               {val_total[best_epoch-1] - train_total[best_epoch-1]:.6f}

FINAL MODEL
{'='*35}

Final Train Loss:  {final_train_loss:.6f}
Final Val Loss:    {final_val_loss:.6f}
Final Gap:         {final_gap:.6f}

IMPROVEMENT
{'='*35}

Initial Val Loss:  {val_total[0]:.6f}
Best Val Loss:     {best_val_loss:.6f}
Improvement:       {improvement:.1f}%
Epochs to Best:    {epochs_to_best}

CONFIGURATION
{'='*35}

Image Size:        {config.image_size}×{config.image_size}
Batch Size:        {config.batch_size}
Learning Rate:     {config.learning_rate}
Disease Weight:    {config.disease_weight_strength}
Embedding Dim:     {config.embedding_dim}
    """

    ax_stats.text(0.05, 0.95, stats_text, fontsize=10, family='monospace',
                  verticalalignment='top', bbox=dict(boxstyle='round',
                  facecolor='wheat', alpha=0.3))

    # Bottom left: Loss distribution
    ax_dist = fig.add_subplot(gs[2, 0])
    ax_dist.hist(train_total, bins=30, alpha=0.7, label='Train', color='blue')
    ax_dist.hist(val_total, bins=30, alpha=0.7, label='Val', color='orange')
    ax_dist.axvline(best_val_loss, color='red', linestyle='--', linewidth=2)
    ax_dist.set_xlabel('Loss', fontsize=11)
    ax_dist.set_ylabel('Frequency', fontsize=11)
    ax_dist.set_title('Loss Distribution', fontsize=12, fontweight='bold')
    ax_dist.legend(fontsize=9)
    ax_dist.grid(True, alpha=0.3, axis='y')

    # Bottom middle: Recent epochs
    recent_epochs = min(20, len(epochs))
    ax_recent = fig.add_subplot(gs[2, 1])
    ax_recent.plot(epochs[-recent_epochs:], train_total[-recent_epochs:],
                   linewidth=2, marker='o', label='Train', markersize=4)
    ax_recent.plot(epochs[-recent_epochs:], val_total[-recent_epochs:],
                   linewidth=2, marker='o', label='Val', markersize=4)
    ax_recent.set_xlabel('Epoch', fontsize=11)
    ax_recent.set_ylabel('Loss', fontsize=11)
    ax_recent.set_title(f'Last {recent_epochs} Epochs', fontsize=12, fontweight='bold')
    ax_recent.legend(fontsize=9)
    ax_recent.grid(True, alpha=0.3)

    # Bottom right: Convergence indicator
    ax_conv = fig.add_subplot(gs[2, 2])
    window = 10
    if len(val_total) >= window:
        val_rolling_std = [np.std(val_total[max(0,i-window):i+1])
                          for i in range(len(val_total))]
        ax_conv.plot(epochs, val_rolling_std, linewidth=2, color='green')
        ax_conv.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
        ax_conv.set_xlabel('Epoch', fontsize=11)
        ax_conv.set_ylabel('Std Dev (10-epoch window)', fontsize=11)
        ax_conv.set_title('Convergence Indicator', fontsize=12, fontweight='bold')
        ax_conv.grid(True, alpha=0.3)

    plt.suptitle('Disease Autoencoder - Training Summary',
                 fontsize=18, fontweight='bold', y=0.995)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training summary to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training history')
    parser.add_argument('--detailed', action='store_true',
                       help='Create detailed loss component plots')
    args = parser.parse_args()

    print("="*60)
    print("Disease Autoencoder - Training Visualization")
    print("="*60)

    config = DiseaseConfig()

    # Load history
    history = load_training_history(config)
    if history is None:
        return

    # Create output directory
    output_dir = config.visualizations_dir / 'training'
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate plots
    plot_loss_curves(history, config, output_dir / 'loss_curves.png')
    plot_training_summary(history, config, output_dir / 'training_summary.png')

    if args.detailed:
        plot_detailed_loss_components(history, config, output_dir / 'detailed_components.png')

    print("\n" + "="*60)
    print("✓ Training Visualization Complete!")
    print("="*60)
    print(f"\nVisualization saved to: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - loss_curves.png           (Main training/validation curves)")
    print(f"  - training_summary.png      (Comprehensive summary)")
    if args.detailed:
        print(f"  - detailed_components.png   (Detailed loss components)")


if __name__ == '__main__':
    main()
