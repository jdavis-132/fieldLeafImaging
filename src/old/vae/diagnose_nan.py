"""
Diagnostic script to identify source of NaN losses in VAE training.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.vae.config import VAEConfig
from src.vae.model import create_model
from src.vae.loss import create_loss_function
from src.vae.dataset import get_dataloaders


def check_tensor(name, tensor):
    """Check if tensor contains NaN or Inf."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        print(f"  ❌ {name}: NaN={has_nan}, Inf={has_inf}")
        print(f"     Min={tensor.min().item():.4f}, Max={tensor.max().item():.4f}")
        return False
    else:
        print(f"  ✓ {name}: Min={tensor.min().item():.4f}, Max={tensor.max().item():.4f}, Mean={tensor.mean().item():.4f}")
        return True


def diagnose_with_real_data():
    """Run diagnosis with real data from the dataset."""
    print("="*60)
    print("VAE NaN Diagnosis with Real Data")
    print("="*60)

    # Load config
    config = VAEConfig()
    print(f"\nConfig loaded. Device: {config.device}")

    # Load splits and LAB stats
    with open(config.existing_splits_path, 'r') as f:
        splits = json.load(f)

    with open(config.existing_lab_stats_path, 'r') as f:
        lab_stats = json.load(f)

    print(f"LAB stats: mean={lab_stats['lab_mean']}, std={lab_stats['lab_std']}")

    # Create dataloader
    dataloaders = get_dataloaders(splits, lab_stats, config)

    # Get one batch
    batch = next(iter(dataloaders['train']))
    inputs = batch['input'].to(config.device)
    targets = batch['target'].to(config.device)
    masks = batch['mask'].to(config.device)

    print(f"\n{'='*60}")
    print("Step 1: Check Input Data")
    print(f"{'='*60}")
    check_tensor("Input (LAB + mask)", inputs)
    check_tensor("Target (LAB)", targets)
    check_tensor("Mask", masks)

    # Create model
    print(f"\n{'='*60}")
    print("Step 2: Create Model and Check Initialization")
    print(f"{'='*60}")
    model = create_model(config)

    # Check model parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"  ❌ {name} has NaN or Inf after initialization!")

    print("Model parameters initialized successfully")

    # Forward pass through encoder
    print(f"\n{'='*60}")
    print("Step 3: Forward Pass - Encoder")
    print(f"{'='*60}")
    model.eval()
    with torch.no_grad():
        bottleneck = model.encoder(inputs)
        check_tensor("Encoder output (bottleneck)", bottleneck)

    # Forward pass through variational bottleneck
    print(f"\n{'='*60}")
    print("Step 4: Forward Pass - Variational Bottleneck")
    print(f"{'='*60}")
    with torch.no_grad():
        batch_size = bottleneck.size(0)
        bottleneck_flat = bottleneck.view(batch_size, -1)

        # Check flattened bottleneck
        check_tensor("Bottleneck (flattened)", bottleneck_flat)

        # Compute mu and log_var
        mu = model.variational_bottleneck.fc_mu(bottleneck_flat)
        log_var = model.variational_bottleneck.fc_log_var(bottleneck_flat)

        check_tensor("Mu", mu)
        check_tensor("Log_var", log_var)

        # Reparameterization
        std = torch.exp(0.5 * log_var)
        check_tensor("Std (exp(0.5 * log_var))", std)

        eps = torch.randn_like(std)
        z = mu + eps * std
        check_tensor("Latent z", z)

        # Decode back
        x_decoded = model.variational_bottleneck.fc_decode(z)
        check_tensor("Decoded bottleneck (flat)", x_decoded)

        bottleneck_recon, mu, log_var = model.variational_bottleneck(bottleneck)
        check_tensor("Reconstructed bottleneck", bottleneck_recon)

    # Forward pass through decoder
    print(f"\n{'='*60}")
    print("Step 5: Forward Pass - Decoder")
    print(f"{'='*60}")
    with torch.no_grad():
        if model.attention is not None:
            bottleneck_attn = model.attention(bottleneck_recon)
            check_tensor("After attention", bottleneck_attn)
        else:
            bottleneck_attn = bottleneck_recon

        reconstruction = model.decoder(bottleneck_attn)
        check_tensor("Reconstruction output", reconstruction)

    # Full forward pass
    print(f"\n{'='*60}")
    print("Step 6: Full Forward Pass")
    print(f"{'='*60}")
    with torch.no_grad():
        predictions, mu, log_var = model(inputs, return_latent=True)
        check_tensor("Predictions", predictions)
        check_tensor("Mu (from full pass)", mu)
        check_tensor("Log_var (from full pass)", log_var)

    # Loss computation
    print(f"\n{'='*60}")
    print("Step 7: Loss Computation")
    print(f"{'='*60}")
    loss_fn = create_loss_function(config)

    with torch.no_grad():
        # Reconstruction loss
        recon_loss = loss_fn.reconstruction_loss(predictions, targets, masks)
        print(f"  Reconstruction loss: {recon_loss.item()}")

        # KL divergence
        kl_div = loss_fn.kl_divergence(mu, log_var)
        print(f"  KL divergence: {kl_div.item()}")

        # Total loss
        total_loss, loss_dict = loss_fn(predictions, targets, mu, log_var, masks, kl_weight=0.0)
        print(f"  Total loss: {total_loss.item()}")
        print(f"  Loss dict: {loss_dict}")

    # Test backward pass
    print(f"\n{'='*60}")
    print("Step 8: Test Training Step with Backward Pass")
    print(f"{'='*60}")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    optimizer.zero_grad()
    predictions, mu, log_var = model(inputs, return_latent=True)

    print("After forward pass (train mode):")
    check_tensor("Predictions", predictions)
    check_tensor("Mu", mu)
    check_tensor("Log_var", log_var)

    loss, loss_dict = loss_fn(predictions, targets, mu, log_var, masks, kl_weight=0.0)
    print(f"\nLoss before backward: {loss.item()}")

    if torch.isnan(loss) or torch.isinf(loss):
        print("❌ Loss is NaN or Inf before backward pass!")
        return

    loss.backward()

    # Check gradients
    print("\nChecking gradients:")
    has_bad_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"  ❌ {name}: gradient has NaN")
                has_bad_grad = True
            elif torch.isinf(param.grad).any():
                print(f"  ❌ {name}: gradient has Inf")
                has_bad_grad = True
            else:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1000:
                    print(f"  ⚠ {name}: large gradient norm = {grad_norm:.2f}")

    if not has_bad_grad:
        print("  ✓ All gradients are finite")

    optimizer.step()

    print("\n" + "="*60)
    print("Diagnosis Complete")
    print("="*60)


def diagnose_with_dummy_data():
    """Run diagnosis with dummy data to isolate model issues."""
    print("\n" + "="*60)
    print("Additional Test: VAE with Dummy Normalized Data")
    print("="*60)

    config = VAEConfig()
    model = create_model(config)
    loss_fn = create_loss_function(config)

    # Create dummy data similar to normalized LAB
    # LAB after normalization should be roughly in range [-3, 3]
    batch_size = 4
    inputs = torch.randn(batch_size, 4, 224, 224).to(config.device) * 2.0  # Simulate normalized data
    targets = torch.randn(batch_size, 3, 224, 224).to(config.device) * 2.0
    masks = torch.ones(batch_size, 1, 224, 224).to(config.device)

    print("\nDummy input stats:")
    check_tensor("Dummy inputs", inputs)
    check_tensor("Dummy targets", targets)

    model.eval()
    with torch.no_grad():
        predictions, mu, log_var = model(inputs, return_latent=True)

        print("\nModel outputs:")
        check_tensor("Predictions", predictions)
        check_tensor("Mu", mu)
        check_tensor("Log_var", log_var)

        loss, loss_dict = loss_fn(predictions, targets, mu, log_var, masks, kl_weight=0.0)
        print(f"\nLoss: {loss.item()}")
        print(f"Loss dict: {loss_dict}")


if __name__ == '__main__':
    try:
        # Test with dummy data first
        diagnose_with_dummy_data()

        # Then test with real data
        diagnose_with_real_data()

    except Exception as e:
        print(f"\n❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()
