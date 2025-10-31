"""
Configuration management for autoencoder pipeline.
"""

import os
import torch
from pathlib import Path


class Config:
    """Configuration for autoencoder training and inference."""

    def __init__(self):
        # Paths
        self.base_dir = Path('/Users/jensinadavis/Documents/fieldLeafImaging')
        self.data_dir = self.base_dir / 'data' / 'ne2025'  # Raw images directory
        self.output_dir = self.base_dir / 'output'  # For masks
        self.csv_path = self.data_dir / 'SbDiv_ne2025_fieldindex.csv'

        # Output directories
        self.models_dir = self.base_dir / 'models'
        self.embeddings_dir = self.base_dir / 'embeddings'
        self.visualizations_dir = self.base_dir / 'visualizations'
        self.logs_dir = self.base_dir / 'logs'

        # Create directories if they don't exist
        for dir_path in [self.models_dir, self.embeddings_dir,
                         self.visualizations_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)

        # Data parameters
        self.image_size = (512, 384)  # (width, height) - maintains 4:3 aspect ratio
        self.original_size = (3060, 4080)  # Original image size

        # Model architecture
        self.embedding_dim = 256  # Bottleneck dimension
        self.num_channels = 3  # RGB

        # Training hyperparameters
        self.batch_size = 16
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.num_workers = 4  # For DataLoader

        # Early stopping
        self.patience = 15  # Epochs to wait before early stopping

        # Learning rate scheduling
        self.lr_scheduler_factor = 0.5  # Reduce LR by this factor
        self.lr_scheduler_patience = 5  # Epochs to wait before reducing LR
        self.lr_scheduler_min = 1e-6  # Minimum learning rate

        # Data split ratios (at genotype level)
        self.train_ratio = 0.70
        self.val_ratio = 0.15
        self.test_ratio = 0.15

        # Random seed for reproducibility
        self.random_seed = 42

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Normalization (ImageNet stats as default)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Visualization
        self.num_vis_samples = 8  # Number of samples to visualize
        self.save_reconstruction_every = 5  # Save reconstruction images every N epochs

        # Logging
        self.log_interval = 10  # Log every N batches during training

    def __repr__(self):
        """String representation of config."""
        config_str = "Configuration:\n"
        config_str += f"  Device: {self.device}\n"
        config_str += f"  Image size: {self.image_size}\n"
        config_str += f"  Embedding dim: {self.embedding_dim}\n"
        config_str += f"  Batch size: {self.batch_size}\n"
        config_str += f"  Learning rate: {self.learning_rate}\n"
        config_str += f"  Num epochs: {self.num_epochs}\n"
        config_str += f"  Random seed: {self.random_seed}\n"
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
            'mean': self.mean,
            'std': self.std,
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Configuration saved to {path}")


# Create default config instance
default_config = Config()
