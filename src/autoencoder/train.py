"""
Training script for Convolutional Autoencoder.

This script provides comprehensive training functionality with:
- Experiment tracking and reproducibility
- Checkpointing and resume capability
- VAE support with KL divergence
- Learning rate scheduling and early stopping
- Visualization and logging

Usage:
    python src/autoencoder/train.py --config src/autoencoder/config.yaml
    python src/autoencoder/train.py --config config.yaml --resume models/experiment/checkpoints/checkpoint_epoch_050.pt
"""

import os
import sys
import argparse
import json
import csv
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import yaml
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import ConvolutionalAutoencoder
from loss import create_loss_from_config, VAELoss
from data_utils import create_dataloaders_from_config


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Python's random module
    import random
    random.seed(seed)

    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_experiment_dir(config: Dict[str, Any]) -> Path:
    """
    Create organized experiment directory with descriptive naming.

    Args:
        config: Configuration dictionary

    Returns:
        Path to experiment directory
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate descriptive name from config
    model_type = "vae" if config['model'].get('is_vae', False) else "standard"
    lr = config['training']['learning_rate']
    bs = config['training']['batch_size']
    loss_type = config['loss']['type']
    attention = "attention" if config['model'].get('use_attention', False) else ""

    desc_parts = [model_type, f"lr{lr}", f"bs{bs}", loss_type]
    if attention:
        desc_parts.append(attention)

    desc_name = "_".join(desc_parts)
    dir_name = f"autoencoder_{timestamp}_{desc_name}"

    # Create directory structure
    base_dir = Path("models") / dir_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (base_dir / "checkpoints").mkdir(exist_ok=True)
    (base_dir / "logs").mkdir(exist_ok=True)
    (base_dir / "plots").mkdir(exist_ok=True)
    (base_dir / "plots" / "reconstructions").mkdir(exist_ok=True)

    # Save config copy
    config_path = base_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Created experiment directory: {base_dir}")
    return base_dir


def initialize_model(
    config: Dict[str, Any],
    device: torch.device
) -> Tuple[nn.Module, optim.Optimizer, nn.Module, Optional[Any]]:
    """
    Initialize model, optimizer, loss function, and scheduler.

    Args:
        config: Configuration dictionary
        device: Device to use (CPU or CUDA)

    Returns:
        Tuple of (model, optimizer, loss_fn, scheduler)
    """
    # Create model
    model = ConvolutionalAutoencoder(config)
    model = model.to(device)

    print(f"\nModel architecture:")
    print("=" * 80)
    model.summary()

    # Create loss function
    loss_fn = create_loss_from_config(config)
    loss_fn = loss_fn.to(device)

    # Create optimizer
    optimizer_name = config['training'].get('optimizer', 'adam').lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0.0)

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config['training'].get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                             weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    print(f"\nOptimizer: {optimizer_name} (lr={lr}, weight_decay={weight_decay})")

    # Create scheduler (optional)
    scheduler = None
    if config['training'].get('use_scheduler', False):
        scheduler_type = config['training'].get('scheduler_type', 'plateau').lower()

        if scheduler_type == 'plateau':
            patience = config['training'].get('scheduler_patience', 10)
            factor = config['training'].get('scheduler_factor', 0.5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience
            )
        elif scheduler_type == 'step':
            step_size = config['training'].get('scheduler_step_size', 30)
            gamma = config['training'].get('scheduler_gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == 'cosine':
            T_max = config['training']['num_epochs']
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        print(f"Scheduler: {scheduler_type}")

    return model, optimizer, loss_fn, scheduler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    is_vae: bool,
    epoch: int
) -> Dict[str, float]:
    """
    Run one training epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to use
        is_vae: Whether model is VAE
        epoch: Current epoch number

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch in pbar:
        images = batch['image'].to(device)

        optimizer.zero_grad()

        # Forward pass
        if is_vae:
            reconstruction, mu, logvar = model(images)

            # Calculate loss with components
            if isinstance(loss_fn, VAELoss):
                loss_dict = loss_fn(
                    reconstruction, images, mu, logvar,
                    return_components=True
                )
                loss = loss_dict['total']
                recon_loss = loss_dict['reconstruction'].item()
                kl_loss = loss_dict['kl'].item()
            else:
                # Fallback if loss_fn is not VAELoss
                loss = loss_fn(reconstruction, images)
                recon_loss = loss.item()
                kl_loss = 0.0
        else:
            reconstruction = model(images)
            loss = loss_fn(reconstruction, images)
            recon_loss = loss.item()
            kl_loss = 0.0

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss
        total_kl_loss += kl_loss

        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.6f}"})

    # Calculate averages
    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon_loss / len(dataloader)
    avg_kl = total_kl_loss / len(dataloader)

    return {
        'train_loss': avg_loss,
        'train_recon_loss': avg_recon,
        'train_kl_loss': avg_kl
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    is_vae: bool,
    epoch: int
) -> Dict[str, float]:
    """
    Run one validation epoch.

    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        loss_fn: Loss function
        device: Device to use
        is_vae: Whether model is VAE
        epoch: Current epoch number

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)

    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)

            # Forward pass
            if is_vae:
                reconstruction, mu, logvar = model(images)

                # Calculate loss with components
                if isinstance(loss_fn, VAELoss):
                    loss_dict = loss_fn(
                        reconstruction, images, mu, logvar,
                        return_components=True
                    )
                    loss = loss_dict['total']
                    recon_loss = loss_dict['reconstruction'].item()
                    kl_loss = loss_dict['kl'].item()
                else:
                    loss = loss_fn(reconstruction, images)
                    recon_loss = loss.item()
                    kl_loss = 0.0
            else:
                reconstruction = model(images)
                loss = loss_fn(reconstruction, images)
                recon_loss = loss.item()
                kl_loss = 0.0

            # Update metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss
            total_kl_loss += kl_loss

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})

    # Calculate averages
    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon_loss / len(dataloader)
    avg_kl = total_kl_loss / len(dataloader)

    return {
        'val_loss': avg_loss,
        'val_recon_loss': avg_recon,
        'val_kl_loss': avg_kl
    }


def save_checkpoint(
    checkpoint_dict: Dict[str, Any],
    filepath: Path
):
    """
    Save checkpoint with all necessary information.

    Args:
        checkpoint_dict: Dictionary with checkpoint data
        filepath: Path to save checkpoint
    """
    torch.save(checkpoint_dict, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Load checkpoint and restore states.

    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Optional scheduler to load state into

    Returns:
        Dictionary with checkpoint data
    """
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if 'random_state' in checkpoint:
        torch.set_rng_state(checkpoint['random_state'])

    print(f"Checkpoint loaded from: {filepath}")
    print(f"Resuming from epoch {checkpoint['epoch'] + 1}")

    return checkpoint


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path,
    train_recon: Optional[List[float]] = None,
    val_recon: Optional[List[float]] = None,
    train_kl: Optional[List[float]] = None,
    val_kl: Optional[List[float]] = None
):
    """
    Generate and save loss curves.

    Args:
        train_losses: Training loss history
        val_losses: Validation loss history
        save_path: Path to save plot
        train_recon: Optional training reconstruction loss
        val_recon: Optional validation reconstruction loss
        train_kl: Optional training KL loss
        val_kl: Optional validation KL loss
    """
    epochs = range(1, len(train_losses) + 1)

    # Determine number of subplots needed
    is_vae = train_kl is not None and any(x > 0 for x in train_kl)
    n_plots = 3 if is_vae else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Plot total loss
    axes[0].plot(epochs, train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot reconstruction loss if available
    if is_vae and train_recon is not None:
        axes[1].plot(epochs, train_recon, label='Train Recon', linewidth=2)
        axes[1].plot(epochs, val_recon, label='Val Recon', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Reconstruction Loss')
        axes[1].set_title('Reconstruction Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot KL divergence
        axes[2].plot(epochs, train_kl, label='Train KL', linewidth=2)
        axes[2].plot(epochs, val_kl, label='Val KL', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('KL Divergence')
        axes[2].set_title('KL Divergence')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved: {save_path}")


def visualize_reconstructions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: Path,
    num_samples: int = 8,
    is_vae: bool = False,
    colorspace: str = 'RGB'
):
    """
    Generate and save sample reconstructions.

    Args:
        model: Model to use for reconstruction
        dataloader: Data loader to sample from
        device: Device to use
        save_path: Path to save visualization
        num_samples: Number of samples to visualize
        is_vae: Whether model is VAE
        colorspace: Colorspace of images ('RGB', 'LAB', etc.)
    """
    model.eval()

    # Get one batch
    batch = next(iter(dataloader))
    images = batch['image'][:num_samples].to(device)

    with torch.no_grad():
        if is_vae:
            reconstructions, _, _ = model(images)
        else:
            reconstructions = model(images)

    # Move to CPU and convert to numpy
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    reconstructions = reconstructions.cpu().permute(0, 2, 3, 1).numpy()

    # Define colorspace maximum values
    colorspace_max = {
        'RGB': np.array([255, 255, 255]),
        'LAB': np.array([100, 128, 128]),
        'HSV': np.array([360, 1, 1]),
        'YCrCb': np.array([255, 255, 255]),
    }

    # Get maximum values for current colorspace (default to RGB)
    max_values = colorspace_max.get(colorspace.upper(), np.array([255, 255, 255]))

    # Create visualization
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))

    for i in range(num_samples):
        # Original image - multiply by colorspace max before plotting
        img_scaled = np.clip(images[i], 0, 1) * max_values
        axes[0, i].imshow(img_scaled.astype(np.uint8) if colorspace.upper() == 'RGB' else img_scaled)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        # Reconstructed image - multiply by colorspace max before plotting
        recon_scaled = np.clip(reconstructions[i], 0, 1) * max_values
        axes[1, i].imshow(recon_scaled.astype(np.uint8) if colorspace.upper() == 'RGB' else recon_scaled)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Reconstructions saved: {save_path}")


def initialize_logging(log_dir: Path, is_vae: bool):
    """
    Initialize CSV log file with headers.

    Args:
        log_dir: Directory to save log file
        is_vae: Whether model is VAE (affects columns)
    """
    log_file = log_dir / "training_log.csv"

    headers = ['epoch', 'train_loss', 'val_loss', 'learning_rate', 'epoch_time']
    if is_vae:
        headers.extend(['train_recon_loss', 'train_kl_loss',
                       'val_recon_loss', 'val_kl_loss'])

    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

    return log_file


def log_epoch(log_file: Path, metrics: Dict[str, Any]):
    """
    Log epoch metrics to CSV.

    Args:
        log_file: Path to log file
        metrics: Dictionary with metrics to log
    """
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writerow(metrics)


def main(config_path: str, resume_path: Optional[str] = None):
    """
    Main training loop.

    Args:
        config_path: Path to configuration file
        resume_path: Optional path to checkpoint to resume from
    """
    print("=" * 80)
    print("Convolutional Autoencoder Training")
    print("=" * 80)

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed
    seed = config['training'].get('random_seed', 42)
    set_seed(seed)
    print(f"\nRandom seed: {seed}")

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create experiment directory
    if resume_path is None:
        exp_dir = create_experiment_dir(config)
    else:
        # Use existing directory
        exp_dir = Path(resume_path).parent.parent
        print(f"Resuming experiment: {exp_dir}")

    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders_from_config(config)

    # Initialize model, optimizer, loss, scheduler
    print("\nInitializing model...")
    model, optimizer, loss_fn, scheduler = initialize_model(config, device)

    # Training parameters
    num_epochs = config['training']['num_epochs']
    save_freq = config['training'].get('save_frequency', 10)
    val_freq = config['training'].get('validation_frequency', 1)
    early_stop_patience = config['training'].get('early_stopping_patience', 20)
    is_vae = config['model'].get('is_vae', False)

    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_loss_history = []
    val_loss_history = []
    train_recon_history = []
    val_recon_history = []
    train_kl_history = []
    val_kl_history = []

    # Resume from checkpoint if specified
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        train_loss_history = checkpoint.get('train_loss_history', [])
        val_loss_history = checkpoint.get('val_loss_history', [])
        train_recon_history = checkpoint.get('train_recon_history', [])
        val_recon_history = checkpoint.get('val_recon_history', [])
        train_kl_history = checkpoint.get('train_kl_history', [])
        val_kl_history = checkpoint.get('val_kl_history', [])

    # Initialize logging
    log_file = initialize_logging(exp_dir / "logs", is_vae)

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training")
    print("=" * 80)

    training_start_time = time.time()

    try:
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()

            # Training
            train_metrics = train_epoch(
                model, train_loader, optimizer, loss_fn, device, is_vae, epoch + 1
            )

            # Validation
            if (epoch + 1) % val_freq == 0:
                val_metrics = validate_epoch(
                    model, val_loader, loss_fn, device, is_vae, epoch + 1
                )
            else:
                # Skip validation this epoch
                val_metrics = {'val_loss': float('nan'),
                             'val_recon_loss': 0.0,
                             'val_kl_loss': 0.0}

            # Update histories
            train_loss_history.append(train_metrics['train_loss'])
            val_loss_history.append(val_metrics['val_loss'])

            if is_vae:
                train_recon_history.append(train_metrics['train_recon_loss'])
                val_recon_history.append(val_metrics['val_recon_loss'])
                train_kl_history.append(train_metrics['train_kl_loss'])
                val_kl_history.append(val_metrics['val_kl_loss'])

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Epoch time
            epoch_time = time.time() - epoch_start_time

            # Log metrics
            log_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['train_loss'],
                'val_loss': val_metrics['val_loss'],
                'learning_rate': current_lr,
                'epoch_time': epoch_time
            }

            if is_vae:
                log_metrics.update({
                    'train_recon_loss': train_metrics['train_recon_loss'],
                    'train_kl_loss': train_metrics['train_kl_loss'],
                    'val_recon_loss': val_metrics['val_recon_loss'],
                    'val_kl_loss': val_metrics['val_kl_loss']
                })

            log_epoch(log_file, log_metrics)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['train_loss']:.6f}")
            print(f"  Val Loss:   {val_metrics['val_loss']:.6f}")
            if is_vae:
                print(f"  Train Recon: {train_metrics['train_recon_loss']:.6f}, "
                      f"KL: {train_metrics['train_kl_loss']:.6f}")
                print(f"  Val Recon:   {val_metrics['val_recon_loss']:.6f}, "
                      f"KL: {val_metrics['val_kl_loss']:.6f}")
            print(f"  LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")

            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['val_loss'])
                else:
                    scheduler.step()

            # Check for best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                epochs_without_improvement = 0

                # Save best model
                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'train_loss': train_metrics['train_loss'],
                    'val_loss': val_metrics['val_loss'],
                    'best_val_loss': best_val_loss,
                    'train_loss_history': train_loss_history,
                    'val_loss_history': val_loss_history,
                    'train_recon_history': train_recon_history,
                    'val_recon_history': val_recon_history,
                    'train_kl_history': train_kl_history,
                    'val_kl_history': val_kl_history,
                    'config': config,
                    'random_state': torch.get_rng_state()
                }
                save_checkpoint(
                    best_checkpoint,
                    exp_dir / "checkpoints" / "best_model.pt"
                )
                print(f"  ✓ New best model saved (val_loss: {best_val_loss:.6f})")
            else:
                epochs_without_improvement += 1

            # Regular checkpoint saving
            if (epoch + 1) % save_freq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'train_loss': train_metrics['train_loss'],
                    'val_loss': val_metrics['val_loss'],
                    'best_val_loss': best_val_loss,
                    'train_loss_history': train_loss_history,
                    'val_loss_history': val_loss_history,
                    'train_recon_history': train_recon_history,
                    'val_recon_history': val_recon_history,
                    'train_kl_history': train_kl_history,
                    'val_kl_history': val_kl_history,
                    'config': config,
                    'random_state': torch.get_rng_state()
                }
                checkpoint_path = exp_dir / "checkpoints" / f"checkpoint_epoch_{epoch+1:03d}.pt"
                save_checkpoint(checkpoint, checkpoint_path)

            # Early stopping
            if epochs_without_improvement >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"No improvement for {early_stop_patience} epochs")
                break

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        # Save checkpoint on interrupt
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_metrics['train_loss'],
            'val_loss': val_metrics['val_loss'],
            'best_val_loss': best_val_loss,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'train_recon_history': train_recon_history,
            'val_recon_history': val_recon_history,
            'train_kl_history': train_kl_history,
            'val_kl_history': val_kl_history,
            'config': config,
            'random_state': torch.get_rng_state()
        }
        checkpoint_path = exp_dir / "checkpoints" / "checkpoint_interrupted.pt"
        save_checkpoint(checkpoint, checkpoint_path)

    # Training complete
    total_training_time = time.time() - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)

    print("\n" + "=" * 80)
    print("Training Complete")
    print("=" * 80)
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final training loss: {train_loss_history[-1]:.6f}")
    print(f"Final validation loss: {val_loss_history[-1]:.6f}")

    # Generate final visualizations
    print("\nGenerating visualizations...")

    # Plot training curves
    plot_training_curves(
        train_loss_history, val_loss_history,
        exp_dir / "plots" / "loss_curves.png",
        train_recon_history if is_vae else None,
        val_recon_history if is_vae else None,
        train_kl_history if is_vae else None,
        val_kl_history if is_vae else None
    )

    # Generate sample reconstructions
    colorspace = config.get('image', {}).get('colorspace', 'RGB')
    visualize_reconstructions(
        model, val_loader, device,
        exp_dir / "plots" / "reconstructions" / "final_reconstructions.png",
        num_samples=8, is_vae=is_vae, colorspace=colorspace
    )

    # Save training summary
    summary = {
        'total_epochs': epoch + 1,
        'best_epoch': train_loss_history.index(min(train_loss_history)) + 1
                     if train_loss_history else 0,
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(train_loss_history[-1]) if train_loss_history else None,
        'final_val_loss': float(val_loss_history[-1]) if val_loss_history else None,
        'total_training_time': f"{hours}h {minutes}m {seconds}s",
        'early_stopped': epochs_without_improvement >= early_stop_patience,
        'device': str(device),
        'random_seed': seed
    }

    summary_path = exp_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Training summary saved: {summary_path}")
    print(f"\nAll outputs saved to: {exp_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Convolutional Autoencoder")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to configuration file (YAML)"
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    args = parser.parse_args()

    main(args.config, args.resume)
