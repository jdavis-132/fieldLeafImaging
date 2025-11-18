"""
Loss function for Variational Autoencoder (VAE).

VAE loss = Reconstruction loss + KL divergence loss
- Reconstruction loss: MSE or BCE between predicted and target images
- KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    """
    VAE loss with reconstruction and KL divergence.

    Loss = Reconstruction_loss + beta * KL_divergence

    where:
    - Reconstruction_loss: MSE or BCE
    - KL_divergence: KL(q(z|x) || N(0, I))
    - beta: Weight for KL divergence (supports annealing)
    """

    def __init__(self, config):
        super(VAELoss, self).__init__()
        self.config = config
        self.reconstruction_loss_type = config.reconstruction_loss_type

    def reconstruction_loss(self, predictions, targets, masks=None):
        """
        Compute reconstruction loss.

        Args:
            predictions: (B, 3, H, W) - Predicted LAB image
            targets: (B, 3, H, W) - Target LAB image
            masks: (B, 1, H, W) - Binary mask (optional, not used currently)

        Returns:
            loss: Scalar reconstruction loss
        """
        if self.reconstruction_loss_type == 'mse':
            # Mean Squared Error
            loss = F.mse_loss(predictions, targets, reduction='mean')

        elif self.reconstruction_loss_type == 'bce':
            # Binary Cross Entropy (requires normalized inputs in [0, 1])
            # Note: This assumes LAB values are normalized to [0, 1]
            loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='mean')

        else:
            raise ValueError(f"Unknown reconstruction loss type: {self.reconstruction_loss_type}")

        return loss

    def kl_divergence(self, mu, log_var):
        """
        Compute KL divergence between learned distribution and standard normal.

        KL(q(z|x) || p(z)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

        Args:
            mu: (B, latent_dim) - Mean of latent distribution
            log_var: (B, latent_dim) - Log-variance of latent distribution

        Returns:
            kl: Scalar KL divergence loss
        """
        # Clamp log_var for numerical stability (should already be clamped in model, but double-check)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)

        # KL divergence formula for Gaussian distributions
        # Using numerically stable computation
        # KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_per_element = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

        # Sum over latent dimensions, then average over batch
        # Standard VAE formulation: sum across latent space per sample, then batch average
        # Shape: (B, latent_dim) -> sum(dim=1) -> (B,) -> mean() -> scalar
        kl = kl_per_element.sum(dim=1).mean()

        return kl

    def forward(self, predictions, targets, mu, log_var, masks=None, kl_weight=None):
        """
        Compute total VAE loss.

        Args:
            predictions: (B, 3, H, W) - Predicted LAB image
            targets: (B, 3, H, W) - Target LAB image
            mu: (B, latent_dim) - Mean of latent distribution
            log_var: (B, latent_dim) - Log-variance of latent distribution
            masks: (B, 1, H, W) - Binary mask (optional)
            kl_weight: KL weight for this step (overrides config if provided)

        Returns:
            total_loss: Scalar total loss
            loss_dict: Dict with loss components for logging
        """
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(predictions, targets, masks)

        # KL divergence
        kl_loss = self.kl_divergence(mu, log_var)

        # Apply KL weight (beta)
        if kl_weight is None:
            kl_weight = self.config.kl_weight

        weighted_kl_loss = kl_weight * kl_loss

        # Total loss
        total_loss = recon_loss + weighted_kl_loss

        # Prepare loss dict for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'weighted_kl_loss': weighted_kl_loss.item(),
            'kl_weight': kl_weight
        }

        return total_loss, loss_dict


def get_kl_weight(epoch, config):
    """
    Compute KL weight for current epoch (with optional annealing).

    KL annealing gradually increases the weight of KL divergence from 0 to full weight
    over the first N epochs. This helps prevent "KL vanishing" where the model
    ignores the latent space.

    Args:
        epoch: Current epoch (0-indexed)
        config: VAEConfig object

    Returns:
        kl_weight: KL weight for this epoch
    """
    if not config.kl_annealing:
        # No annealing, use fixed weight
        return config.kl_weight

    if epoch >= config.kl_annealing_epochs:
        # Annealing complete, use full weight
        return config.kl_weight

    # Linear annealing
    progress = epoch / config.kl_annealing_epochs
    kl_weight = progress * config.kl_weight

    return kl_weight


def create_loss_function(config):
    """
    Create VAE loss function.

    Args:
        config: VAEConfig object

    Returns:
        Loss function (VAELoss)
    """
    loss_fn = VAELoss(config)

    print(f"Using VAE Loss:")
    print(f"  Reconstruction: {config.reconstruction_loss_type.upper()}")
    print(f"  KL weight (beta): {config.kl_weight}")
    print(f"  KL annealing: {config.kl_annealing}")
    if config.kl_annealing:
        print(f"  KL annealing epochs: {config.kl_annealing_epochs}")

    return loss_fn


if __name__ == '__main__':
    # Test loss function
    from src.vae.config import VAEConfig

    config = VAEConfig()
    loss_fn = create_loss_function(config)

    # Create dummy data
    batch_size = 4
    latent_dim = 128
    predictions = torch.randn(batch_size, 3, 224, 224)
    targets = torch.randn(batch_size, 3, 224, 224)
    mu = torch.randn(batch_size, latent_dim)
    log_var = torch.randn(batch_size, latent_dim)
    masks = torch.randint(0, 2, (batch_size, 1, 224, 224)).float()

    # Test with no annealing
    loss, loss_dict = loss_fn(predictions, targets, mu, log_var, masks)

    print(f"\nTest Loss Computation (No Annealing):")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Reconstruction loss: {loss_dict['reconstruction_loss']:.4f}")
    print(f"  KL loss: {loss_dict['kl_loss']:.4f}")
    print(f"  Weighted KL loss: {loss_dict['weighted_kl_loss']:.4f}")
    print(f"  KL weight: {loss_dict['kl_weight']:.4f}")

    # Test with annealing
    print(f"\nTest KL Annealing:")
    for epoch in [0, 10, 25, 50, 100]:
        kl_weight = get_kl_weight(epoch, config)
        print(f"  Epoch {epoch:3d}: KL weight = {kl_weight:.4f}")

    print("\n✓ Loss function test passed!")
