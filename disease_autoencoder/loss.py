"""
Disease-weighted loss function for autoencoder training.

Prioritizes reconstruction of diseased regions based on color deviation from healthy tissue.
"""

import torch
import torch.nn as nn
import numpy as np


class DiseaseWeightedLoss(nn.Module):
    """
    Weighted MSE loss that prioritizes diseased regions.

    Disease weighting is based on color deviation from healthy green tissue in LAB space:
    - Healthy tissue: Green (negative a*), moderate b*
    - Diseased tissue: Shifts toward red/yellow (less negative a*, higher b*)

    The loss automatically computes per-pixel weights based on disease severity.
    """

    def __init__(self, config):
        super(DiseaseWeightedLoss, self).__init__()
        self.config = config

        # Healthy tissue color ranges in normalized LAB space
        # These will be converted based on the LAB normalization statistics
        self.disease_weight_strength = config.disease_weight_strength
        self.l1_weight = config.l1_regularization

    def compute_disease_weights(self, lab_target, mask, lab_mean, lab_std):
        """
        Compute disease weights based on color deviation from healthy tissue.

        In LAB space:
        - L*: Lightness (0-100)
        - a*: Green (-128) to Red (+127)
        - b*: Blue (-128) to Yellow (+127)

        Healthy green tissue typically has:
        - a* around -40 to -20 (green)
        - b* around 20 to 40 (yellow-green)

        Diseased tissue deviates toward:
        - a* less negative (toward red)
        - b* more positive (toward yellow)

        Args:
            lab_target: (B, 3, H, W) - Normalized LAB image
            mask: (B, 1, H, W) - Binary mask
            lab_mean: (3,) - LAB mean used for normalization
            lab_std: (3,) - LAB std used for normalization

        Returns:
            weights: (B, 1, H, W) - Disease weights (1.0 to disease_weight_strength)
        """
        # Denormalize to get actual LAB values
        lab_mean = torch.tensor(lab_mean, device=lab_target.device).view(1, 3, 1, 1)
        lab_std = torch.tensor(lab_std, device=lab_target.device).view(1, 3, 1, 1)

        lab_denorm = lab_target * lab_std + lab_mean

        # Extract a* and b* channels
        a_channel = lab_denorm[:, 1:2, :, :]  # (B, 1, H, W)
        b_channel = lab_denorm[:, 2:3, :, :]  # (B, 1, H, W)

        # Define healthy tissue ranges (in actual LAB values)
        healthy_a_center = -30.0  # Center of healthy green range
        healthy_b_center = 30.0   # Center of healthy yellow-green range

        # Compute deviation from healthy tissue
        # Diseased tissue: a* becomes less negative (moves toward 0 or positive)
        # Diseased tissue: b* becomes more positive (toward yellow)

        # Distance from healthy green in a* channel (focus on red shift)
        a_deviation = torch.relu(a_channel - healthy_a_center)  # Positive when shifting to red

        # Distance from healthy in b* channel (focus on excessive yellow)
        b_deviation = torch.relu(b_channel - healthy_b_center)  # Positive when too yellow

        # Combined disease score (higher = more diseased)
        # Normalize to 0-1 range approximately
        disease_score = (a_deviation / 50.0 + b_deviation / 50.0).clamp(0, 1)

        # Convert to weights: 1.0 (healthy) to disease_weight_strength (diseased)
        weights = 1.0 + disease_score * (self.disease_weight_strength - 1.0)

        # Apply mask (only compute weights for valid pixels)
        weights = weights * mask

        return weights

    def forward(self, predictions, targets, masks, embeddings=None, lab_stats=None):
        """
        Compute disease-weighted reconstruction loss.

        Args:
            predictions: (B, 3, H, W) - Predicted LAB image
            targets: (B, 3, H, W) - Target LAB image
            masks: (B, 1, H, W) - Binary mask
            embeddings: (B, embedding_dim) - Latent codes (for L1 regularization)
            lab_stats: dict with 'lab_mean' and 'lab_std' for weight computation

        Returns:
            loss: Scalar loss value
            loss_dict: Dict with loss components for logging
        """
        # Compute disease weights
        if lab_stats is not None:
            weights = self.compute_disease_weights(
                targets, masks,
                lab_stats['lab_mean'],
                lab_stats['lab_std']
            )
        else:
            # Fallback to uniform weights
            weights = masks

        # Weighted MSE loss (only on masked regions)
        squared_error = (predictions - targets) ** 2  # (B, 3, H, W)

        # Apply weights
        weighted_error = squared_error * weights  # Broadcasting: (B, 3, H, W) * (B, 1, H, W)

        # Normalize by sum of weights (avoid division by zero)
        sum_weights = weights.sum() + 1e-8
        reconstruction_loss = weighted_error.sum() / sum_weights

        # Total loss
        total_loss = reconstruction_loss

        # L1 regularization on embeddings (encourage sparsity)
        l1_loss = torch.tensor(0.0, device=predictions.device)
        if embeddings is not None and self.l1_weight > 0:
            l1_loss = self.l1_weight * torch.abs(embeddings).mean()
            total_loss = total_loss + l1_loss

        # Return loss and components for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'l1_loss': l1_loss.item(),
            'mean_weight': weights[masks > 0].mean().item() if masks.sum() > 0 else 1.0
        }

        return total_loss, loss_dict


class SimpleMSELoss(nn.Module):
    """Simple masked MSE loss (for comparison/debugging)."""

    def __init__(self):
        super(SimpleMSELoss, self).__init__()

    def forward(self, predictions, targets, masks, embeddings=None, lab_stats=None):
        """Compute simple masked MSE loss."""
        # MSE only on masked regions
        squared_error = (predictions - targets) ** 2
        masked_error = squared_error * masks

        # Normalize by number of valid pixels
        num_valid = masks.sum() + 1e-8
        reconstruction_loss = masked_error.sum() / num_valid

        loss_dict = {
            'total_loss': reconstruction_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'l1_loss': 0.0,
            'mean_weight': 1.0
        }

        return reconstruction_loss, loss_dict


def create_loss_function(config, use_disease_weighting=True):
    """
    Create loss function.

    Args:
        config: DiseaseConfig object
        use_disease_weighting: If True, use disease-weighted loss; else simple MSE

    Returns:
        Loss function
    """
    if use_disease_weighting:
        loss_fn = DiseaseWeightedLoss(config)
        print("Using Disease-Weighted Loss")
        print(f"  Disease weight strength: {config.disease_weight_strength}")
        print(f"  L1 regularization: {config.l1_regularization}")
    else:
        loss_fn = SimpleMSELoss()
        print("Using Simple MSE Loss")

    return loss_fn


if __name__ == '__main__':
    # Test loss function
    from disease_autoencoder.config import DiseaseConfig

    config = DiseaseConfig()
    loss_fn = create_loss_function(config, use_disease_weighting=True)

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
    print("✓ Loss function test passed!")
