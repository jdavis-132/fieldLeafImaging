"""
Utility functions for training, visualization, and logging.
"""

import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import csv
import json


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether lower or higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """
        Returns True if training should stop.

        Args:
            score: Current metric value (e.g., validation loss)
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class TrainingLogger:
    """Log training metrics to CSV file."""

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize CSV file with headers
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate'])

    def log(self, epoch, train_loss, val_loss, lr):
        """Append metrics for one epoch."""
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, lr])


def save_checkpoint(state, filepath, is_best=False):
    """
    Save model checkpoint.

    Args:
        state: Dict with 'epoch', 'model_state_dict', 'optimizer_state_dict', etc.
        filepath: Path to save checkpoint
        is_best: If True, also save as 'best_model.pth'
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state, filepath)

    if is_best:
        best_path = filepath.parent / 'best_model.pth'
        torch.save(state, best_path)
        print(f"✓ Saved best model to {best_path}")


def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load model on

    Returns:
        epoch: Epoch number from checkpoint
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)

    print(f"✓ Loaded checkpoint from {filepath} (epoch {epoch})")

    return epoch


def plot_training_curves(log_path, save_path=None):
    """
    Plot training and validation loss curves.

    Args:
        log_path: Path to training log CSV
        save_path: Optional path to save plot
    """
    import pandas as pd

    df = pd.read_csv(log_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    ax.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")

    return fig


def visualize_reconstructions(model, dataloader, config, num_samples=8, save_path=None):
    """
    Visualize original and reconstructed images.

    Args:
        model: Trained autoencoder
        dataloader: DataLoader
        config: Config object
        num_samples: Number of samples to visualize
        save_path: Optional path to save figure
    """
    from autoencoder.dataset import denormalize

    model.eval()

    # Get one batch
    batch = next(iter(dataloader))
    images = batch['masked_image'][:num_samples].to(config.device)

    # Get reconstructions
    with torch.no_grad():
        reconstructions = model(images)

    # Denormalize for visualization
    images = denormalize(images, config)
    reconstructions = denormalize(reconstructions, config)

    # Move to CPU and convert to numpy
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    reconstructions = reconstructions.cpu().permute(0, 2, 3, 1).numpy()

    # Clip to [0, 1]
    images = np.clip(images, 0, 1)
    reconstructions = np.clip(reconstructions, 0, 1)

    # Plot
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))

    for i in range(num_samples):
        # Original
        axes[0, i].imshow(images[i])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12, fontweight='bold')

        # Reconstruction
        axes[1, i].imshow(reconstructions[i])
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved reconstructions to {save_path}")

    return fig


def visualize_masking_process(dataset, num_samples=4, save_path=None):
    """
    Visualize the masking process: original, mask1, mask2, combined, final.

    Args:
        dataset: MaskedLeafDataset
        num_samples: Number of samples to show
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, num_samples * 3))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        sample = dataset.get_raw_item(i)

        # Column 0: Original image
        axes[i, 0].imshow(sample['original_image'])
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original Image', fontsize=11, fontweight='bold')

        # Column 1: Mask 0
        axes[i, 1].imshow(sample['mask0'], cmap='gray')
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Mask 1', fontsize=11, fontweight='bold')

        # Column 2: Mask 1
        axes[i, 2].imshow(sample['mask1'], cmap='gray')
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title('Mask 2', fontsize=11, fontweight='bold')

        # Column 3: Combined mask
        axes[i, 3].imshow(sample['combined_mask'], cmap='gray')
        axes[i, 3].axis('off')
        if i == 0:
            axes[i, 3].set_title('Combined Mask', fontsize=11, fontweight='bold')

        # Column 4: Final masked image
        axes[i, 4].imshow(sample['masked_image'])
        axes[i, 4].axis('off')
        if i == 0:
            axes[i, 4].set_title('Masked Image', fontsize=11, fontweight='bold')

        # Add genotype label on the left
        genotype = sample['metadata']['genotype']
        axes[i, 0].text(-20, axes[i, 0].get_ylim()[0] / 2,
                       f"{genotype}",
                       rotation=90, va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved masking process visualization to {save_path}")

    return fig


def calculate_mse(predictions, targets):
    """Calculate mean squared error."""
    return torch.mean((predictions - targets) ** 2).item()


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
