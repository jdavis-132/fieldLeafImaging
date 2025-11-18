"""
Quick test script to verify training works without NaN issues.
"""

import sys
import torch
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.vae.config import VAEConfig
from src.vae.model import create_model
from src.vae.loss import create_loss_function, get_kl_weight
from src.vae.dataset import get_dataloaders


def test_training():
    """Test training for a few batches."""
    print("="*60)
    print("Testing VAE Training with Fixes")
    print("="*60)

    # Create config
    config = VAEConfig()
    config.num_epochs = 2  # Just test 2 epochs
    print(f"\nDevice: {config.device}")

    # Load splits and LAB stats
    with open(config.existing_splits_path, 'r') as f:
        splits = json.load(f)

    with open(config.existing_lab_stats_path, 'r') as f:
        lab_stats = json.load(f)

    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = get_dataloaders(splits, lab_stats, config)

    # Create model
    print("\nCreating model...")
    model = create_model(config)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Create loss function
    loss_fn = create_loss_function(config)

    print("\n" + "="*60)
    print("Testing Training Loop")
    print("="*60)

    for epoch in range(config.num_epochs):
        model.train()
        kl_weight = get_kl_weight(epoch, config)

        print(f"\nEpoch {epoch + 1}/{config.num_epochs} (KL weight: {kl_weight:.4f})")

        batch_count = 0
        max_batches = 10  # Only test first 10 batches per epoch

        for batch_idx, batch in enumerate(dataloaders['train']):
            if batch_count >= max_batches:
                break

            # Move to device
            inputs = batch['input'].to(config.device)
            targets = batch['target'].to(config.device)
            masks = batch['mask'].to(config.device)

            # Forward pass
            optimizer.zero_grad()
            predictions, mu, log_var = model(inputs, return_latent=True)

            # Compute loss
            loss, loss_dict = loss_fn(predictions, targets, mu, log_var, masks, kl_weight)

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ❌ Batch {batch_idx}: Loss is NaN/Inf!")
                print(f"     Predictions: min={predictions.min():.4f}, max={predictions.max():.4f}")
                print(f"     Mu: min={mu.min():.4f}, max={mu.max():.4f}")
                print(f"     Log_var: min={log_var.min():.4f}, max={log_var.max():.4f}")
                return False

            # Backward pass
            loss.backward()

            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Print progress
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx:3d}: "
                      f"Loss={loss.item():.4f}, "
                      f"Recon={loss_dict['reconstruction_loss']:.4f}, "
                      f"KL={loss_dict['kl_loss']:.6f}, "
                      f"GradNorm={grad_norm:.4f}")

            batch_count += 1

    print("\n" + "="*60)
    print("✓ Training test completed successfully!")
    print("  All losses are finite (no NaN or Inf detected)")
    print("="*60)
    return True


if __name__ == '__main__':
    try:
        success = test_training()
        if success:
            print("\n✓ VAE training is working correctly with the fixes!")
        else:
            print("\n❌ VAE training still has issues")
    except Exception as e:
        print(f"\n❌ Error during training test: {e}")
        import traceback
        traceback.print_exc()
