"""
Training script for Variational Autoencoder (VAE).

This script trains a VAE with U-Net architecture on sorghum leaf images
using the same data splits and preprocessing as the standard autoencoder.
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vae.config import VAEConfig
from src.vae.dataset import get_dataloaders
from src.vae.model import create_model
from src.vae.loss import create_loss_function, get_kl_weight


class VAETrainer:
    """Training manager for VAE."""

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

    def train_epoch(self, model, dataloader, optimizer, loss_fn, kl_weight):
        """Train for one epoch."""
        model.train()
        epoch_losses = []
        epoch_loss_components = {
            'reconstruction_loss': [],
            'kl_loss': [],
            'weighted_kl_loss': [],
            'kl_weight': []
        }

        # Track latent space statistics for debugging
        epoch_mu_stats = []
        epoch_log_var_stats = []

        pbar = tqdm(dataloader, desc=f'Epoch {self.current_epoch + 1}/{self.config.num_epochs}')

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            inputs = batch['input'].to(self.device)  # (B, 4, H, W)
            targets = batch['target'].to(self.device)  # (B, 3, H, W)
            masks = batch['mask'].to(self.device)  # (B, 1, H, W)

            # Forward pass
            optimizer.zero_grad()
            predictions, mu, log_var = model(inputs, return_latent=True)

            # Compute loss
            loss, loss_dict = loss_fn(predictions, targets, mu, log_var, masks, kl_weight)

            # Check for NaN before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠ Warning: NaN/Inf detected in loss at batch {batch_idx}")
                print(f"  Predictions - min: {predictions.min().item():.4f}, max: {predictions.max().item():.4f}")
                print(f"  Mu - min: {mu.min().item():.4f}, max: {mu.max().item():.4f}")
                print(f"  Log_var - min: {log_var.min().item():.4f}, max: {log_var.max().item():.4f}")
                print(f"  Skipping this batch...")
                continue

            # Backward pass
            loss.backward()

            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Track losses
            epoch_losses.append(loss.item())
            for key in ['reconstruction_loss', 'kl_loss', 'weighted_kl_loss', 'kl_weight']:
                epoch_loss_components[key].append(loss_dict[key])

            # Track latent statistics (detach to save memory)
            epoch_mu_stats.append({
                'mean': mu.mean().item(),
                'std': mu.std().item(),
                'min': mu.min().item(),
                'max': mu.max().item()
            })
            epoch_log_var_stats.append({
                'mean': log_var.mean().item(),
                'std': log_var.std().item(),
                'min': log_var.min().item(),
                'max': log_var.max().item()
            })

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'rec': loss_dict['reconstruction_loss'],
                'kl': loss_dict['kl_loss'],
                'β': kl_weight
            })

        # Compute epoch statistics
        epoch_stats = {
            'total_loss': np.mean(epoch_losses),
            'reconstruction_loss': np.mean(epoch_loss_components['reconstruction_loss']),
            'kl_loss': np.mean(epoch_loss_components['kl_loss']),
            'weighted_kl_loss': np.mean(epoch_loss_components['weighted_kl_loss']),
            'kl_weight': kl_weight,
            # Add latent space statistics
            'mu_mean': np.mean([s['mean'] for s in epoch_mu_stats]),
            'mu_std': np.mean([s['std'] for s in epoch_mu_stats]),
            'log_var_mean': np.mean([s['mean'] for s in epoch_log_var_stats]),
            'log_var_std': np.mean([s['std'] for s in epoch_log_var_stats]),
            'variance_mean': np.exp(np.mean([s['mean'] for s in epoch_log_var_stats]))
        }

        return epoch_stats

    def validate(self, model, dataloader, loss_fn, kl_weight):
        """Validate model."""
        model.eval()
        val_losses = []
        val_loss_components = {
            'reconstruction_loss': [],
            'kl_loss': [],
            'weighted_kl_loss': [],
            'kl_weight': []
        }

        # Track latent space statistics for debugging
        val_mu_stats = []
        val_log_var_stats = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validating'):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                masks = batch['mask'].to(self.device)

                # Forward pass
                predictions, mu, log_var = model(inputs, return_latent=True)

                # Compute loss
                loss, loss_dict = loss_fn(predictions, targets, mu, log_var, masks, kl_weight)

                val_losses.append(loss.item())
                for key in ['reconstruction_loss', 'kl_loss', 'weighted_kl_loss', 'kl_weight']:
                    val_loss_components[key].append(loss_dict[key])

                # Track latent statistics
                val_mu_stats.append({
                    'mean': mu.mean().item(),
                    'std': mu.std().item()
                })
                val_log_var_stats.append({
                    'mean': log_var.mean().item(),
                    'std': log_var.std().item()
                })

        # Compute validation statistics
        val_stats = {
            'total_loss': np.mean(val_losses),
            'reconstruction_loss': np.mean(val_loss_components['reconstruction_loss']),
            'kl_loss': np.mean(val_loss_components['kl_loss']),
            'weighted_kl_loss': np.mean(val_loss_components['weighted_kl_loss']),
            'kl_weight': kl_weight,
            # Add latent space statistics
            'mu_mean': np.mean([s['mean'] for s in val_mu_stats]),
            'mu_std': np.mean([s['std'] for s in val_mu_stats]),
            'log_var_mean': np.mean([s['mean'] for s in val_log_var_stats]),
            'log_var_std': np.mean([s['std'] for s in val_log_var_stats]),
            'variance_mean': np.exp(np.mean([s['mean'] for s in val_log_var_stats]))
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
                'latent_dim': self.config.latent_dim,
                'unet_features': self.config.unet_features,
                'use_attention': self.config.use_attention,
                'reconstruction_loss_type': self.config.reconstruction_loss_type,
                'kl_weight': self.config.kl_weight,
                'kl_annealing': self.config.kl_annealing,
                'kl_annealing_epochs': self.config.kl_annealing_epochs,
                'model_type': 'vae'
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
        loss_fn = create_loss_function(self.config)

        print(f"\n{'='*60}")
        print("Starting VAE Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Number of epochs: {self.config.num_epochs}")
        print(f"Early stopping patience: {self.config.patience}")
        print(f"Reconstruction loss: {self.config.reconstruction_loss_type.upper()}")
        print(f"KL weight (beta): {self.config.kl_weight}")
        if self.config.kl_annealing:
            print(f"KL annealing: {self.config.kl_annealing_epochs} epochs")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Compute KL weight for this epoch (with annealing)
            kl_weight = get_kl_weight(epoch, self.config)

            # Train
            train_stats = self.train_epoch(
                model, dataloaders['train'], optimizer, loss_fn, kl_weight
            )
            self.train_losses.append(train_stats)

            # Validate
            val_stats = self.validate(
                model, dataloaders['val'], loss_fn, kl_weight
            )
            self.val_losses.append(val_stats)

            # Update learning rate
            scheduler.step(val_stats['total_loss'])

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_stats['total_loss']:.6f} "
                  f"(rec: {train_stats['reconstruction_loss']:.6f}, "
                  f"kl: {train_stats['kl_loss']:.6f}, "
                  f"β: {kl_weight:.4f})")
            print(f"  Val Loss:   {val_stats['total_loss']:.6f} "
                  f"(rec: {val_stats['reconstruction_loss']:.6f}, "
                  f"kl: {val_stats['kl_loss']:.6f}, "
                  f"β: {kl_weight:.4f})")
            print(f"  Latent Stats (val): "
                  f"μ={val_stats['mu_mean']:.3f}±{val_stats['mu_std']:.3f}, "
                  f"log(σ²)={val_stats['log_var_mean']:.3f}, "
                  f"σ²={val_stats['variance_mean']:.3f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

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
            'training_time_hours': elapsed_time / 3600,
            'model_type': 'vae'
        }
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to {history_path}")


def main():
    """Main training function."""
    # Create config
    config = VAEConfig()
    print(config)

    # Load existing data splits
    print("\n" + "="*60)
    print("Loading Existing Data Splits")
    print("="*60)

    if not config.existing_splits_path.exists():
        print(f"Error: Could not find existing splits at {config.existing_splits_path}")
        print("Please run the standard autoencoder training first to generate splits.")
        return

    with open(config.existing_splits_path, 'r') as f:
        splits = json.load(f)

    print(f"Loaded splits from {config.existing_splits_path}")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val: {len(splits['val'])} images")
    print(f"  Test: {len(splits['test'])} images")

    # Load LAB statistics
    if not config.existing_lab_stats_path.exists():
        print(f"Error: Could not find LAB statistics at {config.existing_lab_stats_path}")
        print("Please run the standard autoencoder training first to generate LAB stats.")
        return

    with open(config.existing_lab_stats_path, 'r') as f:
        lab_stats = json.load(f)

    print(f"\nLoaded LAB statistics from {config.existing_lab_stats_path}")
    print(f"  Mean: {lab_stats['lab_mean']}")
    print(f"  Std: {lab_stats['lab_std']}")

    # Save copies to VAE logs directory
    splits_copy_path = config.logs_dir / 'image_splits.json'
    with open(splits_copy_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"\nSaved splits copy to {splits_copy_path}")

    lab_stats_copy_path = config.logs_dir / 'lab_statistics.json'
    with open(lab_stats_copy_path, 'w') as f:
        json.dump(lab_stats, f, indent=2)
    print(f"Saved LAB stats copy to {lab_stats_copy_path}")

    # Save config
    config_path = config.logs_dir / 'config.json'
    config.save(config_path)

    # Create dataloaders
    print("\n" + "="*60)
    print("Creating DataLoaders")
    print("="*60)
    dataloaders = get_dataloaders(splits, lab_stats, config)

    # Create model
    print("\n" + "="*60)
    print("Creating Model")
    print("="*60)
    model = create_model(config)

    # Train
    trainer = VAETrainer(config)
    trainer.train(model, dataloaders, lab_stats)

    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
