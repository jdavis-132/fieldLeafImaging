"""
Configuration for standard autoencoder pipeline (no disease weighting).

This version uses simple MSE reconstruction loss instead of disease-weighted loss.
All other architecture and training parameters remain the same.
"""

import torch
from pathlib import Path


class AutoencoderConfig:
    """Configuration for standard autoencoder training and inference."""

    def __init__(self):
        # Paths - updated to current working directory structure
        self.base_dir = Path('/home/schnable/Documents/fieldLeafImaging')
        self.data_dir = self.base_dir / 'data' / 'ne2025'
        self.output_dir = self.base_dir / 'output'
        self.csv_path = self.data_dir / 'SbDiv_ne2025_fieldindex.csv'

        # Output directories for autoencoder (no weighting version)
        self.models_dir = self.base_dir / 'src' / 'autoencoder_no_weighting' / 'models'
        self.embeddings_dir = self.base_dir / 'src' / 'autoencoder_no_weighting' / 'embeddings'
        self.visualizations_dir = self.base_dir / 'src' / 'autoencoder_no_weighting' / 'visualizations'
        self.logs_dir = self.base_dir / 'src' / 'autoencoder_no_weighting' / 'logs'

        # Create directories if they don't exist
        for dir_path in [self.models_dir, self.embeddings_dir,
                         self.visualizations_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Data parameters
        self.image_size = 224  # Square images for U-Net
        self.crop_padding = 10  # Padding around bounding box when cropping

        # Model architecture
        self.embedding_dim = 256  # Bottleneck dimension
        self.num_input_channels = 4  # LAB (3 channels) + mask (1 channel)
        self.num_output_channels = 3  # LAB (3 channels)

        # U-Net architecture parameters
        self.unet_features = [64, 128, 256, 512]  # Feature channels at each level
        self.use_attention = True  # Use attention in bottleneck

        # Training hyperparameters
        self.batch_size = 16
        self.num_epochs = 200
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.num_workers = 4

        # Early stopping
        self.patience = 15  # Epochs to wait before early stopping

        # Learning rate scheduling
        self.lr_scheduler_factor = 0.5
        self.lr_scheduler_patience = 5
        self.lr_scheduler_min = 1e-6

        # Loss function parameters
        # Removed: disease_weight_strength - not used in simple MSE loss
        self.l1_regularization = 1e-5  # L1 regularization on latent codes (optional)

        # Data split ratios (at genotype level)
        self.train_ratio = 0.70
        self.val_ratio = 0.15
        self.test_ratio = 0.15

        # Random seed for reproducibility
        self.random_seed = 42

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        # Visualization
        self.num_vis_samples = 8
        self.save_reconstruction_every = 5

        # Logging
        self.log_interval = 10

        # UMAP parameters
        self.umap_n_neighbors = 15
        self.umap_min_dist = 0.1
        self.umap_metric = 'cosine'

    def __repr__(self):
        """String representation of config."""
        config_str = "Standard Autoencoder Configuration (No Disease Weighting):\n"
        config_str += f"  Device: {self.device}\n"
        config_str += f"  Image size: {self.image_size}x{self.image_size}\n"
        config_str += f"  Embedding dim: {self.embedding_dim}\n"
        config_str += f"  Batch size: {self.batch_size}\n"
        config_str += f"  Learning rate: {self.learning_rate}\n"
        config_str += f"  Num epochs: {self.num_epochs}\n"
        config_str += f"  Random seed: {self.random_seed}\n"
        config_str += f"  Loss: Simple MSE (no disease weighting)\n"
        return config_str

    def save(self, path):
        """Save configuration to file."""
        import json

        config_dict = {
            'image_size': self.image_size,
            'embedding_dim': self.embedding_dim,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'random_seed': self.random_seed,
            'l1_regularization': self.l1_regularization,
            'unet_features': self.unet_features,
            'use_attention': self.use_attention,
            'loss_type': 'simple_mse'  # Added to indicate no disease weighting
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Configuration saved to {path}")
