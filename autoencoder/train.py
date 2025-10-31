"""
Training script for autoencoder.

Trains the autoencoder on masked leaf images with train/validation splits.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json
from pathlib import Path

from autoencoder.config import Config
from autoencoder.model import create_model
from autoencoder.dataset import get_dataloaders
from autoencoder.utils import (
    AverageMeter, EarlyStopping, TrainingLogger,
    save_checkpoint, plot_training_curves, visualize_reconstructions
)


def train_one_epoch(model, dataloader, criterion, optimizer, device, config, epoch):
    """
    Train for one epoch.

    Returns:
        Average training loss for the epoch
    """
    model.train()
    losses = AverageMeter()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, batch in enumerate(pbar):
        images = batch['masked_image'].to(device)

        # Forward pass
        reconstructions = model(images)
        loss = criterion(reconstructions, images)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

        # Update progress bar
        pbar.set_postfix({'loss': f'{losses.avg:.6f}'})

    return losses.avg


def validate(model, dataloader, criterion, device, epoch):
    """
    Validate model.

    Returns:
        Average validation loss
    """
    model.eval()
    losses = AverageMeter()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]  ')

    with torch.no_grad():
        for batch in pbar:
            images = batch['masked_image'].to(device)

            # Forward pass
            reconstructions = model(images)
            loss = criterion(reconstructions, images)

            # Update metrics
            losses.update(loss.item(), images.size(0))

            # Update progress bar
            pbar.set_postfix({'loss': f'{losses.avg:.6f}'})

    return losses.avg


def train(config):
    """
    Main training function.

    Args:
        config: Config object with hyperparameters
    """
    print("=" * 80)
    print("Starting Autoencoder Training")
    print("=" * 80)
    print(config)

    # Load splits
    splits_path = config.logs_dir / 'image_splits.json'
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Splits file not found: {splits_path}\n"
            "Please run prepare_splits.py first!"
        )

    with open(splits_path, 'r') as f:
        splits = json.load(f)

    print(f"\nLoaded splits:")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val: {len(splits['val'])} images")
    print(f"  Test: {len(splits['test'])} images")

    # Create dataloaders
    print("\nCreating DataLoaders...")
    dataloaders = get_dataloaders(splits, config)

    # Create model
    print("\nCreating model...")
    model = create_model(config)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.lr_scheduler_min
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        mode='min'
    )

    # Training logger
    logger = TrainingLogger(config.logs_dir / 'training_log.csv')

    # Save configuration
    config.save(config.logs_dir / 'training_config.json')

    # Training loop
    print("\n" + "=" * 80)
    print("Training Loop")
    print("=" * 80)

    best_val_loss = float('inf')

    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print("-" * 80)

        # Train
        train_loss = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer,
            config.device, config, epoch
        )

        # Validate
        val_loss = validate(
            model, dataloaders['val'], criterion, config.device, epoch
        )

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        logger.log(epoch, train_loss, val_loss, current_lr)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  LR:         {current_lr:.2e}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  ✓ New best validation loss: {best_val_loss:.6f}")

        checkpoint_path = config.models_dir / f'checkpoint_epoch_{epoch}.pth'
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'config': config.__dict__
        }, checkpoint_path, is_best=is_best)

        # Save visualizations periodically
        if epoch % config.save_reconstruction_every == 0 or epoch == 1:
            vis_path = config.visualizations_dir / f'reconstructions_epoch_{epoch}.png'
            visualize_reconstructions(
                model, dataloaders['val'], config,
                num_samples=config.num_vis_samples,
                save_path=vis_path
            )

        # Early stopping check
        if early_stopping(val_loss):
            print(f"\n{'='*80}")
            print(f"Early stopping triggered after {epoch} epochs")
            print(f"Best validation loss: {best_val_loss:.6f}")
            print(f"{'='*80}")
            break

    # Training complete
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {config.models_dir / 'best_model.pth'}")

    # Plot training curves
    curves_path = config.visualizations_dir / 'training_curves.png'
    plot_training_curves(config.logs_dir / 'training_log.csv', save_path=curves_path)

    # Final visualizations on all splits
    print("\nGenerating final visualizations...")
    for split_name in ['train', 'val', 'test']:
        vis_path = config.visualizations_dir / f'reconstructions_{split_name}_final.png'
        visualize_reconstructions(
            model, dataloaders[split_name], config,
            num_samples=config.num_vis_samples,
            save_path=vis_path
        )

    print("\n" + "=" * 80)
    print("All outputs saved to:")
    print(f"  Models:         {config.models_dir}")
    print(f"  Visualizations: {config.visualizations_dir}")
    print(f"  Logs:           {config.logs_dir}")
    print("=" * 80)


def main():
    """Main entry point."""
    config = Config()

    # Set random seeds for reproducibility
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train(config)


if __name__ == '__main__':
    main()
