"""
Standard MSE loss function for autoencoder training (no disease weighting).

This version uses simple Mean Squared Error reconstruction loss,
removing the disease-based pixel weighting from the original implementation.
"""

import torch
import torch.nn as nn


class SimpleMSELoss(nn.Module):
    """
    Simple MSE loss for all pixels in cropped region.

    Removed disease weighting - using standard reconstruction loss.
    Original: Computed disease weights based on LAB color deviation from healthy tissue
    Modified: Simple MSE loss treating all pixels equally
    """

    def __init__(self, config=None):
        super(SimpleMSELoss, self).__init__()
        self.config = config
        # Optional L1 regularization on embeddings
        self.l1_weight = config.l1_regularization if config is not None else 0.0

    def forward(self, predictions, targets, masks, embeddings=None, lab_stats=None):
        """
        Compute simple MSE loss on ALL pixels.

        Args:
            predictions: (B, 3, H, W) - Predicted LAB image
            targets: (B, 3, H, W) - Target LAB image
            masks: (B, 1, H, W) - Binary mask (kept for compatibility but not used for weighting)
            embeddings: (B, embedding_dim) - Latent codes (for optional L1 regularization)
            lab_stats: dict with 'lab_mean' and 'lab_std' (kept for compatibility, not used)

        Returns:
            loss: Scalar loss value
            loss_dict: Dict with loss components for logging
        """
        # Removed disease weighting - using standard MSE loss
        # Original: loss = (weighted_mse_loss(output, target, disease_weights)
        # Modified: loss = F.mse_loss(output, target)

        squared_error = (predictions - targets) ** 2
        reconstruction_loss = squared_error.mean()  # Average over all dimensions

        # Total loss starts with reconstruction
        total_loss = reconstruction_loss

        # Optional L1 regularization on embeddings (encourage sparsity)
        l1_loss = torch.tensor(0.0, device=predictions.device)
        if embeddings is not None and self.l1_weight > 0:
            l1_loss = self.l1_weight * torch.abs(embeddings).mean()
            total_loss = total_loss + l1_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'l1_loss': l1_loss.item(),
            'mean_weight': 1.0  # No weighting, always 1.0
        }

        return total_loss, loss_dict


def create_loss_function(config):
    """
    Create loss function.

    Args:
        config: AutoencoderConfig object

    Returns:
        Loss function (SimpleMSELoss)
    """
    # Removed disease weighting option - always use simple MSE
    loss_fn = SimpleMSELoss(config)
    print("Using Simple MSE Loss (no disease weighting)")
    if config.l1_regularization > 0:
        print(f"  L1 regularization: {config.l1_regularization}")

    return loss_fn


if __name__ == '__main__':
    # Test loss function
    from src.autoencoder_no_weighting_128.config import AutoencoderConfig

    config = AutoencoderConfig()
    loss_fn = create_loss_function(config)

    # Create dummy data
    batch_size = 4
    predictions = torch.randn(batch_size, 3, 224, 224)
    targets = torch.randn(batch_size, 3, 224, 224)
    masks = torch.randint(0, 2, (batch_size, 1, 224, 224)).float()
    embeddings = torch.randn(batch_size, 256)

    lab_stats = {
        'lab_mean': [50.0, 0.0, 20.0],
        'lab_std': [20.0, 30.0, 30.0]
    }

    # Compute loss
    loss, loss_dict = loss_fn(predictions, targets, masks, embeddings, lab_stats)

    print(f"\nTest Loss Computation:")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Loss dict: {loss_dict}")
    print(f"  Mean weight: {loss_dict['mean_weight']} (always 1.0 for unweighted loss)")
    print("✓ Loss function test passed!")
