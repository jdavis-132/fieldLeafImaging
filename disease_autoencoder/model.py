"""
U-Net style autoencoder with attention mechanism for disease-aware reconstruction.
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
    U-Net Encoder with skip connections.

    Input: (B, 4, 224, 224) - LAB + mask
    Outputs: List of feature maps at different scales + bottleneck embedding
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
        skip_connections = []

        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        return x, skip_connections


class Decoder(nn.Module):
    """
    U-Net Decoder with skip connections.

    Input: Bottleneck features + skip connections
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
            self.decoders.append(ConvBlock(feature * 2, feature))  # *2 for skip connection
            current_channels = feature

        # Final output layer
        self.final_conv = nn.Conv2d(features[-1], out_channels, kernel_size=1)

    def forward(self, x, skip_connections):
        # Reverse skip connections (from deepest to shallowest)
        skip_connections = skip_connections[::-1]

        for idx, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)

            # Handle size mismatch
            skip = skip_connections[idx]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate skip connection
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        x = self.final_conv(x)
        return x


class DiseaseAutoencoder(nn.Module):
    """
    U-Net style autoencoder with attention mechanism.

    Architecture:
    - Encoder: 4 down-sampling blocks
    - Bottleneck: With spatial and channel attention
    - Decoder: 4 up-sampling blocks with skip connections
    - Input: LAB + mask (4 channels)
    - Output: LAB image (3 channels)
    """

    def __init__(self, config):
        super(DiseaseAutoencoder, self).__init__()
        self.config = config

        features = config.unet_features

        # Encoder
        self.encoder = Encoder(
            in_channels=config.num_input_channels,
            features=features
        )

        # Attention in bottleneck
        if config.use_attention:
            self.attention = AttentionBlock(features[-1] * 2)
        else:
            self.attention = None

        # Decoder
        self.decoder = Decoder(
            features=features[::-1],  # Reverse for decoding
            out_channels=config.num_output_channels
        )

        # Embedding extraction (from bottleneck)
        bottleneck_size = features[-1] * 2
        # Calculate spatial size after 4 pooling operations: 224 -> 112 -> 56 -> 28 -> 14
        spatial_size = config.image_size // (2 ** len(features))
        self.embedding_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_size, config.embedding_dim)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (B, 4, H, W) - LAB + mask

        Returns:
            reconstruction: (B, 3, H, W) - Reconstructed LAB
        """
        # Encode
        bottleneck, skip_connections = self.encoder(x)

        # Apply attention
        if self.attention is not None:
            bottleneck = self.attention(bottleneck)

        # Decode
        reconstruction = self.decoder(bottleneck, skip_connections)

        return reconstruction

    def encode(self, x):
        """
        Extract embedding from input.

        Args:
            x: Input tensor (B, 4, H, W)

        Returns:
            Embedding tensor (B, embedding_dim)
        """
        bottleneck, _ = self.encoder(x)

        if self.attention is not None:
            bottleneck = self.attention(bottleneck)

        embedding = self.embedding_extractor(bottleneck)
        return embedding

    def num_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config):
    """
    Create disease-aware autoencoder model.

    Args:
        config: DiseaseConfig object

    Returns:
        model: DiseaseAutoencoder on config.device
    """
    model = DiseaseAutoencoder(config)
    model = model.to(config.device)

    print(f"\nCreated Disease-Aware U-Net Autoencoder")
    print(f"  Total parameters: {model.num_parameters():,}")
    print(f"  Embedding dimension: {config.embedding_dim}")
    print(f"  Device: {config.device}")
    print(f"  Attention: {config.use_attention}")

    return model


def test_model_shapes(config):
    """Test that model produces correct output shapes."""
    print("\nTesting model shapes...")

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
        embedding = model.encode(dummy_input)
        print(f"Embedding shape: {embedding.shape}")
        assert embedding.shape == (batch_size, config.embedding_dim)

        # Test full forward pass
        reconstruction = model(dummy_input)
        print(f"Reconstruction shape: {reconstruction.shape}")
        assert reconstruction.shape == (batch_size, config.num_output_channels,
                                       config.image_size, config.image_size)

    print("✓ All shape tests passed!")
    return model


if __name__ == '__main__':
    from disease_autoencoder.config import DiseaseConfig
    config = DiseaseConfig()
    test_model_shapes(config)
