"""
Variational Autoencoder (VAE) for sorghum leaf images.

Architecture:
- Encoder with progressive down-sampling
- Variational bottleneck (mean and log-variance)
- Decoder with progressive up-sampling
- Attention mechanism in bottleneck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block with batch normalization."""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism."""

    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute attention map
        attention = self.sigmoid(self.conv(x))
        # Apply attention
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention mechanism (Squeeze-and-Excitation)."""

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Apply attention
        return x * y.expand_as(x)


class AttentionBlock(nn.Module):
    """Combined spatial and channel attention."""

    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.spatial_attention = SpatialAttention(in_channels)
        self.channel_attention = ChannelAttention(in_channels)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class Encoder(nn.Module):
    """
    Encoder without skip connections.

    Input: (B, 4, 224, 224) - LAB + mask
    Outputs: Bottleneck features
    """

    def __init__(self, in_channels=4, features=[64, 128, 256, 512]):
        super(Encoder, self).__init__()
        self.features = features

        # Encoder blocks
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_channels = in_channels
        for feature in features:
            self.encoders.append(ConvBlock(current_channels, feature))
            current_channels = feature

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        return x


class VariationalBottleneck(nn.Module):
    """
    Variational bottleneck for VAE.

    Takes bottleneck features and produces mean and log-variance,
    then samples from the latent distribution using reparameterization trick.
    """

    def __init__(self, bottleneck_channels, latent_dim, spatial_size):
        super(VariationalBottleneck, self).__init__()
        self.bottleneck_channels = bottleneck_channels
        self.latent_dim = latent_dim
        self.spatial_size = spatial_size

        # Flatten bottleneck features
        flattened_size = bottleneck_channels * spatial_size * spatial_size

        # Mean and log-variance layers
        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_log_var = nn.Linear(flattened_size, latent_dim)

        # Decode latent back to bottleneck features
        self.fc_decode = nn.Linear(latent_dim, flattened_size)

        # Initialize log_var layer with small weights to prevent initial overflow
        # This ensures log_var starts small and stable
        nn.init.xavier_normal_(self.fc_log_var.weight, gain=0.01)
        nn.init.constant_(self.fc_log_var.bias, -3.0)  # Start with small variance

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + eps * sigma
        where eps ~ N(0, 1) and sigma = exp(0.5 * log_var)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        """
        Forward pass through variational bottleneck.

        Args:
            x: Bottleneck features (B, C, H, W)

        Returns:
            x_reconstructed: Reconstructed bottleneck features (B, C, H, W)
            mu: Mean of latent distribution (B, latent_dim)
            log_var: Log-variance of latent distribution (B, latent_dim)
        """
        batch_size = x.size(0)

        # Flatten
        x_flat = x.view(batch_size, -1)

        # Compute mu and log_var
        mu = self.fc_mu(x_flat)
        log_var = self.fc_log_var(x_flat)

        # Clamp log_var to prevent numerical overflow in exp()
        # exp(10) ≈ 22000, which is large but safe for float32
        # This prevents exp() from overflowing to infinity
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)

        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)

        # Decode back to bottleneck shape
        x_decoded = self.fc_decode(z)
        x_reconstructed = x_decoded.view(
            batch_size,
            self.bottleneck_channels,
            self.spatial_size,
            self.spatial_size
        )

        return x_reconstructed, mu, log_var


class Decoder(nn.Module):
    """
    Decoder without skip connections.

    Input: Bottleneck features
    Output: (B, 3, 224, 224) - Reconstructed LAB image
    """

    def __init__(self, features=[512, 256, 128, 64], out_channels=3):
        super(Decoder, self).__init__()
        self.features = features

        # Decoder blocks
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Start from bottleneck channels
        current_channels = features[0] * 2

        for feature in features:
            self.upconvs.append(
                nn.ConvTranspose2d(current_channels, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock(feature, feature))
            current_channels = feature

        # Final output layer
        self.final_conv = nn.Conv2d(features[-1], out_channels, kernel_size=1)

    def forward(self, x):
        for upconv, decoder in zip(self.upconvs, self.decoders):
            x = upconv(x)
            x = decoder(x)

        x = self.final_conv(x)
        return x


class VAE(nn.Module):
    """
    Variational Autoencoder.

    Architecture:
    - Encoder: 4 down-sampling blocks
    - Variational Bottleneck: Mean and log-variance with reparameterization
    - Attention: Spatial and channel attention (optional)
    - Decoder: 4 up-sampling blocks
    - Input: LAB + mask (4 channels)
    - Output: LAB image (3 channels) + mu + log_var
    """

    def __init__(self, config):
        super(VAE, self).__init__()
        self.config = config

        features = config.unet_features

        # Encoder
        self.encoder = Encoder(
            in_channels=config.num_input_channels,
            features=features
        )

        # Calculate spatial size after 4 pooling operations: 224 -> 112 -> 56 -> 28 -> 14
        spatial_size = config.image_size // (2 ** len(features))
        bottleneck_channels = features[-1] * 2

        # Variational bottleneck
        self.variational_bottleneck = VariationalBottleneck(
            bottleneck_channels=bottleneck_channels,
            latent_dim=config.latent_dim,
            spatial_size=spatial_size
        )

        # Attention (applied after variational bottleneck)
        if config.use_attention:
            self.attention = AttentionBlock(bottleneck_channels)
        else:
            self.attention = None

        # Decoder
        self.decoder = Decoder(
            features=features[::-1],  # Reverse for decoding
            out_channels=config.num_output_channels
        )

    def forward(self, x, return_latent=False):
        """
        Forward pass.

        Args:
            x: Input tensor (B, 4, H, W) - LAB + mask
            return_latent: If True, also return mu and log_var

        Returns:
            reconstruction: (B, 3, H, W) - Reconstructed LAB
            mu: (B, latent_dim) - Mean of latent distribution (if return_latent=True)
            log_var: (B, latent_dim) - Log-variance of latent distribution (if return_latent=True)
        """
        # Encode
        bottleneck = self.encoder(x)

        # Variational bottleneck
        bottleneck_recon, mu, log_var = self.variational_bottleneck(bottleneck)

        # Apply attention
        if self.attention is not None:
            bottleneck_recon = self.attention(bottleneck_recon)

        # Decode
        reconstruction = self.decoder(bottleneck_recon)

        if return_latent:
            return reconstruction, mu, log_var
        else:
            return reconstruction

    def encode(self, x):
        """
        Extract latent representation (mean) from input.

        Args:
            x: Input tensor (B, 4, H, W)

        Returns:
            mu: Mean of latent distribution (B, latent_dim)
            log_var: Log-variance of latent distribution (B, latent_dim)
        """
        bottleneck = self.encoder(x)

        # Get mean and log_var from variational bottleneck
        batch_size = bottleneck.size(0)
        bottleneck_flat = bottleneck.view(batch_size, -1)
        mu = self.variational_bottleneck.fc_mu(bottleneck_flat)
        log_var = self.variational_bottleneck.fc_log_var(bottleneck_flat)

        # Clamp log_var for stability
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)

        return mu, log_var

    def sample(self, num_samples, device):
        """
        Sample from the learned latent distribution.

        Args:
            num_samples: Number of samples to generate
            device: Device to use

        Returns:
            samples: Generated images (num_samples, 3, H, W)
        """
        # Sample from standard normal
        z = torch.randn(num_samples, self.config.latent_dim).to(device)

        # Decode
        with torch.no_grad():
            # Pass through decoder part of variational bottleneck
            bottleneck_features = self.variational_bottleneck.fc_decode(z)
            spatial_size = self.config.image_size // (2 ** len(self.config.unet_features))
            bottleneck_channels = self.config.unet_features[-1] * 2

            bottleneck_recon = bottleneck_features.view(
                num_samples,
                bottleneck_channels,
                spatial_size,
                spatial_size
            )

            # Apply attention
            if self.attention is not None:
                bottleneck_recon = self.attention(bottleneck_recon)

            # Decode through decoder
            samples = self.decoder(bottleneck_recon)

        return samples

    def num_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config):
    """
    Create VAE model.

    Args:
        config: VAEConfig object

    Returns:
        model: VAE on config.device
    """
    model = VAE(config)
    model = model.to(config.device)

    print(f"\nCreated Variational Autoencoder (VAE)")
    print(f"  Total parameters: {model.num_parameters():,}")
    print(f"  Latent dimension: {config.latent_dim}")
    print(f"  Device: {config.device}")
    print(f"  Attention: {config.use_attention}")
    print(f"  Architecture: Encoder-decoder with variational bottleneck")

    return model


def test_model_shapes(config):
    """Test that model produces correct output shapes."""
    print("\nTesting VAE model shapes...")

    model = create_model(config)

    # Create dummy input (LAB + mask)
    batch_size = 4
    dummy_input = torch.randn(
        batch_size,
        config.num_input_channels,
        config.image_size,
        config.image_size
    ).to(config.device)

    print(f"Input shape: {dummy_input.shape}")

    # Test encoding
    with torch.no_grad():
        mu = model.encode(dummy_input)
        print(f"Latent (mu) shape: {mu.shape}")
        assert mu.shape == (batch_size, config.latent_dim)

        # Test full forward pass
        reconstruction, mu, log_var = model(dummy_input, return_latent=True)
        print(f"Reconstruction shape: {reconstruction.shape}")
        print(f"Mu shape: {mu.shape}")
        print(f"Log_var shape: {log_var.shape}")
        assert reconstruction.shape == (batch_size, config.num_output_channels,
                                       config.image_size, config.image_size)
        assert mu.shape == (batch_size, config.latent_dim)
        assert log_var.shape == (batch_size, config.latent_dim)

    print("✓ All shape tests passed!")
    return model


if __name__ == '__main__':
    from src.vae.config import VAEConfig
    config = VAEConfig()
    test_model_shapes(config)
