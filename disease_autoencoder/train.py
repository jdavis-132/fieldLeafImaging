"""
Training script for disease-aware autoencoder.
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import prepare_splits
sys.path.insert(0, str(Path(__file__).parent.parent))

from autoencoder.prepare_splits import find_all_images, split_by_genotype
from disease_autoencoder.config import DiseaseConfig
from disease_autoencoder.dataset import get_dataloaders
from disease_autoencoder.model import create_model
from disease_autoencoder.loss import create_loss_function


class Trainer:
    """Training manager for disease-aware autoencoder."""

    def __init__(self, config):
        self.config = config
        self.device = config.device

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # History
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, model, dataloader, optimizer, loss_fn, lab_stats):
        """Train for one epoch."""
        model.train()
        epoch_losses = []
        epoch_loss_components = {
            'reconstruction_loss': [],
            'l1_loss': [],
            'mean_weight': []
        }

        pbar = tqdm(dataloader, desc=f'Epoch {self.current_epoch + 1}/{self.config.num_epochs}')

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            inputs = batch['input'].to(self.device)  # (B, 4, H, W)
            targets = batch['target'].to(self.device)  # (B, 3, H, W)
            masks = batch['mask'].to(self.device)  # (B, 1, H, W)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(inputs)

            # Extract embeddings for L1 regularization
            with torch.no_grad():
                embeddings = model.encode(inputs)

            # Compute loss
            loss, loss_dict = loss_fn(predictions, targets, masks, embeddings, lab_stats)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track losses
            epoch_losses.append(loss.item())
            for key in ['reconstruction_loss', 'l1_loss', 'mean_weight']:
                epoch_loss_components[key].append(loss_dict[key])

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'rec': loss_dict['reconstruction_loss'],
                'w': loss_dict['mean_weight']
            })

        # Compute epoch statistics
        epoch_stats = {
            'total_loss': np.mean(epoch_losses),
            'reconstruction_loss': np.mean(epoch_loss_components['reconstruction_loss']),
            'l1_loss': np.mean(epoch_loss_components['l1_loss']),
            'mean_weight': np.mean(epoch_loss_components['mean_weight'])
        }

        return epoch_stats

    def validate(self, model, dataloader, loss_fn, lab_stats):
        """Validate model."""
        model.eval()
        val_losses = []
        val_loss_components = {
            'reconstruction_loss': [],
            'l1_loss': [],
            'mean_weight': []
        }

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validating'):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                masks = batch['mask'].to(self.device)

                # Forward pass
                predictions = model(inputs)
                embeddings = model.encode(inputs)

                # Compute loss
                loss, loss_dict = loss_fn(predictions, targets, masks, embeddings, lab_stats)

                val_losses.append(loss.item())
                for key in ['reconstruction_loss', 'l1_loss', 'mean_weight']:
                    val_loss_components[key].append(loss_dict[key])

        # Compute validation statistics
        val_stats = {
            'total_loss': np.mean(val_losses),
            'reconstruction_loss': np.mean(val_loss_components['reconstruction_loss']),
            'l1_loss': np.mean(val_loss_components['l1_loss']),
            'mean_weight': np.mean(val_loss_components['mean_weight'])
        }

        return val_stats

    def save_checkpoint(self, model, optimizer, scheduler, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': {
                'image_size': self.config.image_size,
                'embedding_dim': self.config.embedding_dim,
                'unet_features': self.config.unet_features,
                'use_attention': self.config.use_attention,
            }
        }

        # Save latest checkpoint
        checkpoint_path = self.config.models_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.config.models_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"  Saved best model (val_loss: {self.best_val_loss:.6f})")

        # Save epoch checkpoint
        if (self.current_epoch + 1) % 10 == 0:
            epoch_path = self.config.models_dir / f'checkpoint_epoch_{self.current_epoch + 1}.pth'
            torch.save(checkpoint, epoch_path)

    def train(self, model, dataloaders, lab_stats):
        """Main training loop."""
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Create learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
            min_lr=self.config.lr_scheduler_min
        )

        # Create loss function
        loss_fn = create_loss_function(self.config, use_disease_weighting=True)

        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Number of epochs: {self.config.num_epochs}")
        print(f"Early stopping patience: {self.config.patience}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_stats = self.train_epoch(
                model, dataloaders['train'], optimizer, loss_fn, lab_stats
            )
            self.train_losses.append(train_stats)

            # Validate
            val_stats = self.validate(
                model, dataloaders['val'], loss_fn, lab_stats
            )
            self.val_losses.append(val_stats)

            # Update learning rate
            scheduler.step(val_stats['total_loss'])

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_stats['total_loss']:.6f} "
                  f"(rec: {train_stats['reconstruction_loss']:.6f}, "
                  f"l1: {train_stats['l1_loss']:.6f}, "
                  f"weight: {train_stats['mean_weight']:.3f})")
            print(f"  Val Loss:   {val_stats['total_loss']:.6f} "
                  f"(rec: {val_stats['reconstruction_loss']:.6f}, "
                  f"l1: {val_stats['l1_loss']:.6f}, "
                  f"weight: {val_stats['mean_weight']:.3f})")

            # Check for improvement
            is_best = val_stats['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_stats['total_loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(model, optimizer, scheduler, is_best)

            # Early stopping
            if self.epochs_without_improvement >= self.config.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation loss: {self.best_val_loss:.6f}")
                break

        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Total time: {elapsed_time / 3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Save training history
        history_path = self.config.logs_dir / 'training_history.json'
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'training_time_hours': elapsed_time / 3600
        }
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to {history_path}")


def main():
    """Main training function."""
    # Create config
    config = DiseaseConfig()
    print(config)

    # Prepare data splits (using existing prepare_splits function)
    print("\n" + "="*60)
    print("Preparing Data Splits")
    print("="*60)

    # Find all images
    print("Finding all usable images...")
    from autoencoder.config import Config as OldConfig
    old_config = OldConfig()
    old_config.data_dir = config.data_dir
    old_config.output_dir = config.output_dir
    old_config.csv_path = config.csv_path
    old_config.train_ratio = config.train_ratio
    old_config.val_ratio = config.val_ratio
    old_config.test_ratio = config.test_ratio
    old_config.random_seed = config.random_seed

    images = find_all_images(old_config)
    print(f"Found {len(images)} usable images")

    # Split by genotype
    print("\nSplitting by genotype...")
    splits, genotype_splits = split_by_genotype(images, old_config)

    # Save splits to disease autoencoder logs
    splits_path = config.logs_dir / 'image_splits.json'
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Image splits saved to {splits_path}")

    # Create dataloaders
    print("\n" + "="*60)
    print("Creating DataLoaders")
    print("="*60)
    dataloaders, lab_stats = get_dataloaders(splits, config)

    # Create model
    print("\n" + "="*60)
    print("Creating Model")
    print("="*60)
    model = create_model(config)

    # Train
    trainer = Trainer(config)
    trainer.train(model, dataloaders, lab_stats)

    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
