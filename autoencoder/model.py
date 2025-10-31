"""
Convolutional Autoencoder architecture for masked leaf images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder: Convolutional layers to compress image to embedding.

    Input: (B, 3, H, W) - Masked RGB images
    Output: (B, embedding_dim) - Bottleneck embeddings
    """

    def __init__(self, embedding_dim=256, input_channels=3):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim

        # Convolutional layers with batch normalization
        # Input: (B, 3, 384, 512)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # Output: (B, 64, 192, 256)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # Output: (B, 128, 96, 128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # Output: (B, 256, 48, 64)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        # Output: (B, 512, 24, 32)

        # Calculate flattened size: 512 * 24 * 32 = 393,216
        self.flatten_size = 512 * 24 * 32

        # Fully connected layer to bottleneck
        self.fc = nn.Linear(self.flatten_size, embedding_dim)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass through encoder.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Embedding tensor (B, embedding_dim)
        """
        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Bottleneck
        embedding = self.fc(x)

        return embedding


class Decoder(nn.Module):
    """
    Decoder: Deconvolutional layers to reconstruct image from embedding.

    Input: (B, embedding_dim) - Bottleneck embeddings
    Output: (B, 3, H, W) - Reconstructed RGB images
    """

    def __init__(self, embedding_dim=256, output_channels=3):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim

        # Fully connected layer from bottleneck
        self.flatten_size = 512 * 24 * 32
        self.fc = nn.Linear(embedding_dim, self.flatten_size)

        # Deconvolutional layers (transposed convolutions)
        # Input after reshape: (B, 512, 24, 32)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        # Output: (B, 256, 48, 64)

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # Output: (B, 128, 96, 128)

        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # Output: (B, 64, 192, 256)

        self.deconv4 = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)
        # Output: (B, 3, 384, 512)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()  # Output in [0, 1]

    def forward(self, embedding):
        """
        Forward pass through decoder.

        Args:
            embedding: Embedding tensor (B, embedding_dim)

        Returns:
            Reconstructed image tensor (B, 3, H, W)
        """
        # Fully connected
        x = self.fc(embedding)

        # Reshape to (B, 512, 24, 32)
        x = x.view(x.size(0), 512, 24, 32)

        # Deconvolutional layers
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.sigmoid(self.deconv4(x))  # Final layer with sigmoid

        return x


class Autoencoder(nn.Module):
    """
    Complete Autoencoder: Encoder + Decoder.

    Input: (B, 3, H, W) - Masked RGB images
    Output: (B, 3, H, W) - Reconstructed RGB images
    Embedding: (B, embedding_dim) - Bottleneck representation
    """

    def __init__(self, embedding_dim=256):
        super(Autoencoder, self).__init__()

        self.embedding_dim = embedding_dim

        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.decoder = Decoder(embedding_dim=embedding_dim)

    def forward(self, x):
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            reconstruction: Reconstructed tensor (B, 3, H, W)
        """
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction

    def encode(self, x):
        """
        Encode input to embedding (for embedding extraction).

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Embedding tensor (B, embedding_dim)
        """
        return self.encoder(x)

    def decode(self, embedding):
        """
        Decode embedding to reconstruction.

        Args:
            embedding: Embedding tensor (B, embedding_dim)

        Returns:
            Reconstructed tensor (B, 3, H, W)
        """
        return self.decoder(embedding)

    def num_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config):
    """
    Create autoencoder model and move to device.

    Args:
        config: Config object

    Returns:
        model: Autoencoder on config.device
    """
    model = Autoencoder(embedding_dim=config.embedding_dim)
    model = model.to(config.device)

    print(f"Created Autoencoder with {model.num_parameters():,} parameters")
    print(f"Embedding dimension: {config.embedding_dim}")
    print(f"Device: {config.device}")

    return model


def test_model_shapes(config):
    """Test that model produces correct output shapes."""
    print("\nTesting model shapes...")

    model = create_model(config)

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(
        batch_size,
        3,
        config.image_size[1],  # height
        config.image_size[0]   # width
    ).to(config.device)

    print(f"Input shape: {dummy_input.shape}")

    # Test encoding
    embedding = model.encode(dummy_input)
    print(f"Embedding shape: {embedding.shape}")
    assert embedding.shape == (batch_size, config.embedding_dim)

    # Test decoding
    reconstruction = model.decode(embedding)
    print(f"Reconstruction shape: {reconstruction.shape}")
    assert reconstruction.shape == dummy_input.shape

    # Test full forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == dummy_input.shape

    print("✓ All shape tests passed!")


if __name__ == '__main__':
    # Test model
    from autoencoder.config import Config
    config = Config()
    test_model_shapes(config)
