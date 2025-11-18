"""
Configuration for Variational Autoencoder (VAE) pipeline.

This configuration is based on the standard autoencoder but adapted for VAE
with parameters for KL divergence weighting and variational inference.
"""

import torch
from pathlib import Path


class VAEConfig:
    """Configuration for VAE training and inference."""

    def __init__(self):
        # Paths - updated to VAE directory structure
        self.base_dir = Path('/home/schnable/Documents/fieldLeafImaging')
        self.data_dir = self.base_dir / 'data' / 'ne2025'
        self.output_dir = self.base_dir / 'output'
        self.csv_path = self.data_dir / 'SbDiv_ne2025_fieldindex.csv'

        # Output directories for VAE
        self.models_dir = self.base_dir / 'src' / 'vae' / 'logs' / 'checkpoints'
        self.embeddings_dir = self.base_dir / 'src' / 'vae' / 'embeddings'
        self.visualizations_dir = self.base_dir / 'src' / 'vae' / 'logs' / 'visualizations'
        self.logs_dir = self.base_dir / 'src' / 'vae' / 'logs'

        # Create directories if they don't exist
        for dir_path in [self.models_dir, self.embeddings_dir,
                         self.visualizations_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Load existing splits and stats from autoencoder
        self.existing_splits_path = self.base_dir / 'src' / 'autoencoder_no_weighting_128' / 'logs' / 'image_splits.json'
        # Use shared LAB statistics from disease_autoencoder_cropped
        self.existing_lab_stats_path = self.base_dir / 'src' / 'disease_autoencoder_cropped' / 'logs' / 'lab_statistics.json'

        # Data parameters
        self.image_size = 224  # Square images for U-Net
        self.crop_padding = 10  # Padding around bounding box when cropping

        # Model architecture
        self.latent_dim = 128  # Latent dimension for VAE (both mu and log_var)
        self.num_input_channels = 4  # LAB (3 channels) + mask (1 channel)
        self.num_output_channels = 3  # LAB (3 channels)

        # U-Net architecture parameters (same as base autoencoder)
        self.unet_features = [64, 128, 256, 512]  # Feature channels at each level
        self.use_attention = True  # Use attention in bottleneck

        # Training hyperparameters
        self.batch_size = 16  # Will adjust based on GPU memory
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

        # VAE-specific loss parameters
        self.reconstruction_loss_type = 'mse'  # 'mse' or 'bce'
        self.kl_weight = 1.0  # Weight for KL divergence loss (beta in beta-VAE)
        self.kl_annealing = True  # Gradually increase KL weight during training
        self.kl_annealing_epochs = 50  # Number of epochs for KL annealing

        # Data split ratios (using existing splits)
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

        # UMAP/t-SNE parameters for latent space visualization
        self.umap_n_neighbors = 15
        self.umap_min_dist = 0.1
        self.umap_metric = 'cosine'

        # t-SNE parameters
        self.tsne_perplexity = 30
        self.tsne_learning_rate = 200

    def __repr__(self):
        """String representation of config."""
        config_str = "Variational Autoencoder (VAE) Configuration:\n"
        config_str += f"  Device: {self.device}\n"
        config_str += f"  Image size: {self.image_size}x{self.image_size}\n"
        config_str += f"  Latent dim: {self.latent_dim}\n"
        config_str += f"  Batch size: {self.batch_size}\n"
        config_str += f"  Learning rate: {self.learning_rate}\n"
        config_str += f"  Num epochs: {self.num_epochs}\n"
        config_str += f"  Random seed: {self.random_seed}\n"
        config_str += f"  Loss: {self.reconstruction_loss_type.upper()} + KL divergence (weight: {self.kl_weight})\n"
        config_str += f"  KL annealing: {self.kl_annealing} (epochs: {self.kl_annealing_epochs})\n"
        return config_str

    def save(self, path):
        """Save configuration to file."""
        import json

        config_dict = {
            'image_size': self.image_size,
            'latent_dim': self.latent_dim,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'random_seed': self.random_seed,
            'unet_features': self.unet_features,
            'use_attention': self.use_attention,
            'reconstruction_loss_type': self.reconstruction_loss_type,
            'kl_weight': self.kl_weight,
            'kl_annealing': self.kl_annealing,
            'kl_annealing_epochs': self.kl_annealing_epochs,
            'model_type': 'vae'
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Configuration saved to {path}")
