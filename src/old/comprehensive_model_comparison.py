"""
Comprehensive Comparison Framework for Three PyTorch UNet Autoencoder Models

This script compares three trained autoencoder models designed for sorghum leaf
disease (anthracnose) analysis:

1. disease_autoencoder: Weighted MSE loss with SAM2.1 masks applied during training
2. disease_autoencoder_cropped: Weighted MSE loss on pre-cropped images
3. autoencoder_no_weighting: Standard MSE loss on pre-cropped images

The framework evaluates models across multiple dimensions:
- Reconstruction quality:
  * MSE, weighted MSE (full image and masked regions)
  * SSIM (full image and masked regions separately)
  * PSNR (full image and masked regions separately)
- Latent space organization (t-SNE/UMAP visualization)
- Inference performance (time, model size)
- Visual quality (reconstruction grids, error heatmaps)

Key Feature:
- Computes SSIM and PSNR both with and without considering masks
  * Full-image metrics: computed over entire image (ignoring mask)
  * Masked-region metrics: computed only over masked pixels (foreground)

Author: Generated for Schnable Lab
Date: 2025-11-12 (Updated with dual metric computation)
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless systems
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    directory: Path
    checkpoint_name: str = "checkpoint_best.pth"
    description: str = ""
    uses_masks: bool = False
    uses_weighting: bool = False

    def __post_init__(self):
        self.directory = Path(self.directory)


@dataclass
class ComparisonConfig:
    """Configuration for the comparison framework."""
    # Model configurations
    models: List[ModelConfig]

    # Output directories
    output_dir: Path = Path("comparison_results")

    # Evaluation settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 16
    num_workers: int = 4

    # Visualization settings
    num_sample_images: int = 10
    random_seed: int = 42

    # Dimensionality reduction
    use_tsne: bool = True
    use_umap: bool = True
    tsne_perplexity: int = 30
    umap_n_neighbors: int = 15

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        for model_config in self.models:
            model_config.directory = Path(model_config.directory)


# ==============================================================================
# Model Loading Utilities
# ==============================================================================

class ModelLoader:
    """Utility class for loading trained autoencoder models."""

    @staticmethod
    def load_config(config_path: Path) -> Any:
        """
        Load configuration from a Python module.

        Args:
            config_path: Path to config.py file

        Returns:
            Configuration object instance
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        # Try to find the config class
        for attr_name in dir(config_module):
            if 'Config' in attr_name and not attr_name.startswith('_'):
                config_class = getattr(config_module, attr_name)
                return config_class()

        raise ValueError(f"No config class found in {config_path}")

    @staticmethod
    def load_model(model_dir: Path, checkpoint_name: str, device: str) -> Tuple[nn.Module, Dict, Dict]:
        """
        Load a trained model from checkpoint.

        Args:
            model_dir: Directory containing model files
            checkpoint_name: Name of checkpoint file
            device: Device to load model on

        Returns:
            Tuple of (model, checkpoint_dict, config_dict)
        """
        # Load configuration
        config_path = model_dir / "config.py"
        config = ModelLoader.load_config(config_path)

        # Import model creation function
        import importlib.util
        model_path = model_dir / "model.py"
        spec = importlib.util.spec_from_file_location("model", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # Create model
        model = model_module.create_model(config)

        # Load checkpoint
        checkpoint_path = model_dir / "models" / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        # Extract config as dict
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}

        return model, checkpoint, config_dict

    @staticmethod
    def load_training_history(model_dir: Path) -> Dict:
        """Load training history JSON."""
        history_path = model_dir / "logs" / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                return json.load(f)
        return {}

    @staticmethod
    def load_lab_statistics(model_dir: Path) -> Dict:
        """Load LAB color statistics."""
        stats_path = model_dir / "logs" / "lab_statistics.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                return json.load(f)
        return {}

    @staticmethod
    def load_data_splits(model_dir: Path) -> Dict:
        """Load data split information."""
        splits_path = model_dir / "logs" / "image_splits.json"
        if splits_path.exists():
            with open(splits_path, 'r') as f:
                return json.load(f)
        return {}


# ==============================================================================
# LAB Color Space Utilities
# ==============================================================================

import cv2

def denormalize_lab(lab_tensor: np.ndarray, lab_stats: Dict) -> np.ndarray:
    """
    Denormalize LAB tensor using statistics.

    Args:
        lab_tensor: Normalized LAB tensor (C, H, W) or (H, W, C)
        lab_stats: Dictionary with 'lab_mean' and 'lab_std'

    Returns:
        Denormalized LAB array
    """
    lab_mean = np.array(lab_stats['lab_mean'])
    lab_std = np.array(lab_stats['lab_std'])

    # Handle both CHW and HWC formats
    if lab_tensor.shape[-1] == 3:  # HWC format
        return lab_tensor * lab_std + lab_mean
    else:  # CHW format
        lab_mean = lab_mean.reshape(-1, 1, 1)
        lab_std = lab_std.reshape(-1, 1, 1)
        return lab_tensor * lab_std + lab_mean


def lab_to_rgb(lab_image: np.ndarray) -> np.ndarray:
    """
    Convert LAB image to RGB following disease_autoencoder_cropped/evaluate.py procedure.

    Args:
        lab_image: LAB image in HWC format (H, W, 3) as float32

    Returns:
        RGB image as uint8 (H, W, 3)
    """
    # Data is already in OpenCV's LAB format (all channels 0-255)
    # Just clip and convert to uint8
    lab_cv2 = np.clip(lab_image, 0, 255).astype(np.uint8)

    # Convert LAB to BGR then to RGB
    bgr = cv2.cvtColor(lab_cv2, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb


# ==============================================================================
# Dataset for Evaluation
# ==============================================================================

class ValidationDataset(torch.utils.data.Dataset):
    """Dataset for loading and preprocessing validation images."""

    def __init__(
        self,
        image_splits: Dict,
        lab_stats: Dict,
        config: Any,
        split: str = 'val',
        mask_background: bool = True
    ):
        """
        Initialize validation dataset.

        Args:
            image_splits: Dictionary containing image metadata
            lab_stats: LAB normalization statistics
            config: Model configuration object
            split: Which split to use ('val', 'test', or 'train')
            mask_background: Whether to zero out background pixels
        """
        self.images = image_splits[split]
        self.split = split
        self.lab_mean = np.array(lab_stats['lab_mean'])
        self.lab_std = np.array(lab_stats['lab_std'])
        self.config = config
        self.mask_background = mask_background

    def __len__(self) -> int:
        return len(self.images)

    def _get_leaf_bbox(self, mask: np.ndarray, padding: int = 20) -> Tuple[int, int, int, int]:
        """Get bounding box of leaf from mask with padding."""
        coords = np.column_stack(np.where(mask > 0))

        if len(coords) == 0:
            return 0, mask.shape[0], 0, mask.shape[1]

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Add padding
        y_min = max(0, y_min - padding)
        y_max = min(mask.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(mask.shape[1], x_max + padding)

        return y_min, y_max, x_min, x_max

    def __getitem__(self, idx: int) -> Dict:
        """
        Load and preprocess a single image.

        Returns:
            Dictionary with:
                'input': torch.Tensor (4, H, W) - LAB + mask
                'target': torch.Tensor (3, H, W) - LAB image
                'mask': torch.Tensor (1, H, W) - binary mask
                'metadata': dict
                'original_rgb': np.ndarray (H, W, 3) - for visualization
        """
        metadata = self.images[idx]

        try:
            # Load image
            image = cv2.imread(metadata['image_path'])
            if image is None:
                raise FileNotFoundError(f"Could not load image: {metadata['image_path']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load masks and combine
            mask0 = cv2.imread(metadata['mask0_path'], cv2.IMREAD_GRAYSCALE)
            mask1 = cv2.imread(metadata['mask1_path'], cv2.IMREAD_GRAYSCALE)

            if mask0 is None or mask1 is None:
                raise FileNotFoundError(f"Could not load masks for image: {metadata['image_path']}")

            combined_mask = np.logical_or(mask0 > 0, mask1 > 0).astype(np.uint8)

            # Get bounding box and crop
            crop_padding = getattr(self.config, 'crop_padding', 20)
            y_min, y_max, x_min, x_max = self._get_leaf_bbox(combined_mask, padding=crop_padding)
            image = image[y_min:y_max, x_min:x_max]
            combined_mask = combined_mask[y_min:y_max, x_min:x_max]

            # Resize
            image_size = getattr(self.config, 'image_size', 224)
            image_resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(combined_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

            # Keep original for visualization
            original_rgb = image_resized.copy()

            # Convert to LAB color space
            lab_image = cv2.cvtColor(image_resized, cv2.COLOR_RGB2LAB).astype(np.float32)

            # Normalize LAB channels
            lab_image = (lab_image - self.lab_mean) / (self.lab_std + 1e-8)

            # Optionally mask background pixels
            if self.mask_background:
                mask_3ch = np.stack([mask_resized] * 3, axis=2)
                lab_image = lab_image * mask_3ch

            # Convert to torch tensors (HWC -> CHW)
            lab_tensor = torch.from_numpy(lab_image).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()

            # Create input (LAB + mask)
            input_tensor = torch.cat([lab_tensor, mask_tensor], dim=0)

            return {
                'input': input_tensor,  # (4, H, W)
                'target': lab_tensor,   # (3, H, W)
                'mask': mask_tensor,    # (1, H, W)
                'metadata': metadata,
                'original_rgb': original_rgb
            }

        except Exception as e:
            print(f"Error loading image {idx}: {e}")
            # Return a dummy sample in case of error
            # image_size = getattr(self.config, 'image_size', 224)
            # return {
            #     'input': torch.zeros(4, image_size, image_size),
            #     'target': torch.zeros(3, image_size, image_size),
            #     'mask': torch.zeros(1, image_size, image_size),
            #     'metadata': metadata,
            #     'original_rgb': np.zeros((image_size, image_size, 3), dtype=np.uint8)
            }


# ==============================================================================
# Loss Function Utilities
# ==============================================================================

class LossCalculator:
    """Utility class for calculating various loss metrics."""

    @staticmethod
    def compute_lab_weights(
        lab_image: torch.Tensor,
        lab_stats: Dict,
        disease_weight_strength: float = 2.0
    ) -> torch.Tensor:
        """
        Compute disease-based weights from LAB color values.

        Weights are computed based on deviation from healthy tissue color:
        - Healthy tissue: a* around -30 (green), b* around 30 (yellow-green)
        - Diseased tissue: shifts toward red (higher a*) and yellow (higher b*)

        Args:
            lab_image: LAB image tensor (B, 3, H, W) or (3, H, W)
            lab_stats: Dictionary with 'lab_mean' and 'lab_std'
            disease_weight_strength: Maximum weight for diseased pixels

        Returns:
            Weight tensor same shape as lab_image
        """
        # Denormalize LAB values
        lab_mean = torch.tensor(lab_stats['lab_mean'], device=lab_image.device).view(-1, 1, 1)
        lab_std = torch.tensor(lab_stats['lab_std'], device=lab_image.device).view(-1, 1, 1)

        if lab_image.dim() == 4:
            lab_mean = lab_mean.unsqueeze(0)
            lab_std = lab_std.unsqueeze(0)

        lab_denorm = lab_image * lab_std + lab_mean

        # Extract a* and b* channels
        a_channel = lab_denorm[..., 1, :, :] if lab_image.dim() == 4 else lab_denorm[1, :, :]
        b_channel = lab_denorm[..., 2, :, :] if lab_image.dim() == 4 else lab_denorm[2, :, :]

        # Healthy tissue ranges (in LAB 0-255 scale, but shifted to -128 to 127)
        # a*: -50 to -10 (green), b*: 10 to 40 (yellow-green)
        healthy_a_min, healthy_a_max = -50, -10
        healthy_b_min, healthy_b_max = 10, 40

        # Compute deviation from healthy range
        a_deviation = torch.clamp(
            torch.maximum(
                healthy_a_min - a_channel,
                a_channel - healthy_a_max
            ),
            min=0
        )

        b_deviation = torch.clamp(
            torch.maximum(
                healthy_b_min - b_channel,
                b_channel - healthy_b_max
            ),
            min=0
        )

        # Combine deviations
        combined_deviation = torch.sqrt(a_deviation**2 + b_deviation**2)

        # Normalize to [0, 1] and scale to [1, disease_weight_strength]
        max_deviation = torch.quantile(combined_deviation, 0.95)
        normalized_deviation = torch.clamp(combined_deviation / (max_deviation + 1e-8), 0, 1)

        weights = 1.0 + normalized_deviation * (disease_weight_strength - 1.0)

        # Expand to match LAB channels
        if lab_image.dim() == 4:
            weights = weights.unsqueeze(1).expand_as(lab_image)
        else:
            weights = weights.unsqueeze(0).expand_as(lab_image)

        return weights

    @staticmethod
    def masked_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute masked and weighted MSE.

        Args:
            pred: Predictions (B, C, H, W)
            target: Targets (B, C, H, W)
            mask: Optional mask (B, 1, H, W)
            weights: Optional per-pixel weights (B, C, H, W)

        Returns:
            Mean squared error
        """
        squared_error = (pred - target) ** 2

        if weights is not None:
            squared_error = squared_error * weights

        if mask is not None:
            mask = mask.expand_as(squared_error)
            squared_error = squared_error * mask
            return squared_error.sum() / (mask.sum() + 1e-8)
        else:
            return squared_error.mean()

    @staticmethod
    def compute_ssim(
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None,
        compute_full: bool = True,
        compute_masked: bool = True
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute SSIM (Structural Similarity Index) with and without mask.

        Args:
            pred: Prediction image (H, W, C) or (H, W)
            target: Target image (H, W, C) or (H, W)
            mask: Optional binary mask (H, W) where 1=foreground, 0=background
            compute_full: Whether to compute full-image SSIM
            compute_masked: Whether to compute masked-region SSIM

        Returns:
            Tuple of (ssim_full, ssim_masked)
            - ssim_full: SSIM over entire image (None if compute_full=False)
            - ssim_masked: SSIM over masked regions only (None if no mask or compute_masked=False)
        """
        # Convert to grayscale if multi-channel
        if pred.ndim == 3:
            pred_gray = pred.mean(axis=2)
        else:
            pred_gray = pred.copy()

        if target.ndim == 3:
            target_gray = target.mean(axis=2)
        else:
            target_gray = target.copy()

        ssim_full = None
        ssim_masked = None

        # Compute full-image SSIM (ignoring mask)
        if compute_full:
            try:
                data_range = max(pred_gray.max() - pred_gray.min(), 1e-8)
                ssim_full = ssim(target_gray, pred_gray, data_range=data_range)
            except Exception as e:
                print(f"Warning: Failed to compute full SSIM: {e}")
                ssim_full = None

        # Compute masked SSIM (only over masked regions)
        if compute_masked and mask is not None:
            try:
                # Ensure mask is binary
                binary_mask = (mask > 0).astype(bool)

                # Check if mask has any foreground pixels
                if binary_mask.sum() == 0:
                    # Empty mask - cannot compute masked SSIM
                    ssim_masked = None
                else:
                    # Apply mask: zero out background pixels
                    pred_masked = pred_gray * binary_mask
                    target_masked = target_gray * binary_mask

                    # Compute data range only over masked region
                    masked_values_pred = pred_gray[binary_mask]
                    masked_values_target = target_gray[binary_mask]
                    data_range = max(
                        masked_values_pred.max() - masked_values_pred.min(),
                        masked_values_target.max() - masked_values_target.min(),
                        1e-8
                    )

                    # Compute SSIM over entire image (background is zeroed)
                    # This effectively computes SSIM only over the masked region
                    ssim_masked = ssim(target_masked, pred_masked, data_range=data_range)

            except Exception as e:
                print(f"Warning: Failed to compute masked SSIM: {e}")
                ssim_masked = None

        return ssim_full, ssim_masked

    @staticmethod
    def compute_psnr(
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None,
        compute_full: bool = True,
        compute_masked: bool = True
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute PSNR (Peak Signal-to-Noise Ratio) with and without mask.

        Args:
            pred: Prediction image (H, W, C) or (H, W)
            target: Target image (H, W, C) or (H, W)
            mask: Optional binary mask (H, W) where 1=foreground, 0=background
            compute_full: Whether to compute full-image PSNR
            compute_masked: Whether to compute masked-region PSNR

        Returns:
            Tuple of (psnr_full, psnr_masked)
            - psnr_full: PSNR over entire image (None if compute_full=False)
            - psnr_masked: PSNR over masked regions only (None if no mask or compute_masked=False)
        """
        psnr_full = None
        psnr_masked = None

        # Compute full-image PSNR (ignoring mask)
        if compute_full:
            try:
                data_range = max(pred.max() - pred.min(), 1e-8)
                psnr_full = psnr(target, pred, data_range=data_range)
            except Exception as e:
                print(f"Warning: Failed to compute full PSNR: {e}")
                psnr_full = None

        # Compute masked PSNR (only over masked regions)
        if compute_masked and mask is not None:
            try:
                # Ensure mask is binary
                binary_mask = (mask > 0).astype(bool)

                # Check if mask has any foreground pixels
                if binary_mask.sum() == 0:
                    # Empty mask - cannot compute masked PSNR
                    psnr_masked = None
                else:
                    # For multi-channel images, expand mask
                    if pred.ndim == 3:
                        binary_mask_expanded = np.stack([binary_mask] * pred.shape[2], axis=2)
                    else:
                        binary_mask_expanded = binary_mask

                    # Extract only masked pixels
                    pred_masked_pixels = pred[binary_mask_expanded].flatten()
                    target_masked_pixels = target[binary_mask_expanded].flatten()

                    # Compute MSE over masked pixels only
                    mse_masked = np.mean((pred_masked_pixels - target_masked_pixels) ** 2)

                    # Compute data range over masked pixels
                    data_range = max(
                        pred_masked_pixels.max() - pred_masked_pixels.min(),
                        target_masked_pixels.max() - target_masked_pixels.min(),
                        1e-8
                    )

                    # Compute PSNR: 10 * log10(data_range^2 / MSE)
                    if mse_masked > 1e-10:
                        psnr_masked = 10 * np.log10((data_range ** 2) / mse_masked)
                    else:
                        # MSE is essentially zero - perfect reconstruction
                        psnr_masked = 100.0  # Cap at a high value

            except Exception as e:
                print(f"Warning: Failed to compute masked PSNR: {e}")
                psnr_masked = None

        return psnr_full, psnr_masked


# ==============================================================================
# Evaluation Pipeline
# ==============================================================================

@dataclass
class ModelMetrics:
    """Metrics for a single model."""
    model_name: str
    mse_mean: float
    mse_std: float
    weighted_mse_mean: float
    weighted_mse_std: float

    # Full-image metrics (entire image, ignoring mask)
    ssim_full_mean: float
    ssim_full_std: float
    psnr_full_mean: float
    psnr_full_std: float

    # Masked-region metrics (only masked pixels considered)
    ssim_masked_mean: float
    ssim_masked_std: float
    psnr_masked_mean: float
    psnr_masked_std: float

    masked_mse_mean: float
    masked_mse_std: float
    inference_time_mean: float
    inference_time_std: float
    num_parameters: int
    model_size_mb: float
    latent_dim: int

    # Per-image metrics
    per_image_mse: List[float] = None
    per_image_weighted_mse: List[float] = None
    per_image_ssim_full: List[float] = None
    per_image_ssim_masked: List[float] = None
    per_image_psnr_full: List[float] = None
    per_image_psnr_masked: List[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding per-image metrics."""
        d = asdict(self)
        d.pop('per_image_mse', None)
        d.pop('per_image_weighted_mse', None)
        d.pop('per_image_ssim_full', None)
        d.pop('per_image_ssim_masked', None)
        d.pop('per_image_psnr_full', None)
        d.pop('per_image_psnr_masked', None)
        return d


class ModelEvaluator:
    """Evaluator for comparing multiple models."""

    def __init__(self, config: ComparisonConfig):
        """
        Initialize evaluator.

        Args:
            config: Comparison configuration
        """
        self.config = config
        self.device = torch.device(config.device)

        # Set random seeds
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Storage for results
        self.models = {}
        self.configs = {}
        self.metrics = {}
        self.latent_embeddings = {}
        self.sample_reconstructions = {}

    def load_models(self):
        """Load all models specified in configuration."""
        print("\n" + "="*80)
        print("LOADING MODELS")
        print("="*80 + "\n")

        for model_config in self.config.models:
            print(f"Loading {model_config.name}...")
            model, checkpoint, config_dict = ModelLoader.load_model(
                model_config.directory,
                model_config.checkpoint_name,
                self.device
            )

            self.models[model_config.name] = model
            self.configs[model_config.name] = {
                'model_config': model_config,
                'config_dict': config_dict,
                'training_history': ModelLoader.load_training_history(model_config.directory),
                'lab_stats': ModelLoader.load_lab_statistics(model_config.directory),
                'checkpoint': checkpoint
            }

            # Print model info
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  ✓ Loaded checkpoint: {checkpoint.get('epoch', 'N/A')} epochs")
            print(f"  ✓ Parameters: {num_params:,}")
            print(f"  ✓ Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
            print()

    def get_model_size(self, model_name: str) -> float:
        """Get model size in MB."""
        model = self.models[model_name]
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    def evaluate_all_models(self, num_samples: Optional[int] = None):
        """
        Evaluate all models on validation set.

        Args:
            num_samples: Number of samples to evaluate (None for all)
        """
        print("\n" + "="*80)
        print("EVALUATING MODELS")
        print("="*80 + "\n")

        # Load validation data from first model (they all use the same splits)
        first_model_name = list(self.models.keys())[0]
        first_model_dir = self.config.models[0].directory
        data_splits = ModelLoader.load_data_splits(first_model_dir)

        # Create a sampled version if requested
        if num_samples:
            # Random sample
            all_images = data_splits['val']
            indices = np.random.choice(len(all_images), min(num_samples, len(all_images)), replace=False)
            sampled_images = [all_images[i] for i in indices]
            # Create modified split dict
            data_splits_sampled = {'val': sampled_images}
            print(f"Randomly sampled {len(sampled_images)} validation samples\n")
        else:
            data_splits_sampled = data_splits
            val_count = len(data_splits['val'])
            print(f"Using all {val_count} validation samples\n")

        # Evaluate each model with its appropriate preprocessing
        for model_name in self.models.keys():
            print(f"Evaluating {model_name}...")

            # Get model-specific info
            model_config = next(cfg for cfg in self.config.models if cfg.name == model_name)
            model_info = self.configs[model_name]
            config_dict = model_info['config_dict']
            lab_stats = model_info['lab_stats']

            # Create config object for dataset
            class SimpleConfig:
                def __init__(self, config_dict):
                    for k, v in config_dict.items():
                        setattr(self, k, v)

            simple_config = SimpleConfig(config_dict)

            # Create dataset with appropriate preprocessing
            # disease_autoencoder uses mask_background=True
            # disease_autoencoder_cropped and autoencoder_no_weighting use mask_background=False
            mask_background = model_config.uses_masks

            val_dataset = ValidationDataset(
                data_splits_sampled,
                lab_stats,
                simple_config,
                split='val',
                mask_background=mask_background
            )

            metrics = self._evaluate_single_model(model_name, val_dataset)
            self.metrics[model_name] = metrics
            print(f"  ✓ MSE: {metrics.mse_mean:.6f} ± {metrics.mse_std:.6f}")
            print(f"  ✓ SSIM (full): {metrics.ssim_full_mean:.4f} ± {metrics.ssim_full_std:.4f}")
            print(f"  ✓ SSIM (masked): {metrics.ssim_masked_mean:.4f} ± {metrics.ssim_masked_std:.4f}")
            print(f"  ✓ PSNR (full): {metrics.psnr_full_mean:.2f} ± {metrics.psnr_full_std:.2f} dB")
            print(f"  ✓ PSNR (masked): {metrics.psnr_masked_mean:.2f} ± {metrics.psnr_masked_std:.2f} dB")
            print()

    def _evaluate_single_model(
        self,
        model_name: str,
        val_dataset: ValidationDataset
    ) -> ModelMetrics:
        """
        Evaluate a single model.

        Args:
            model_name: Name of model to evaluate
            val_dataset: Validation dataset

        Returns:
            Model metrics
        """
        model = self.models[model_name]
        model_info = self.configs[model_name]
        lab_stats = model_info['lab_stats']

        # Get model config
        model_cfg = next(cfg for cfg in self.config.models if cfg.name == model_name)

        # Get disease weight strength if available
        config_dict = model_info['config_dict']
        disease_weight_strength = config_dict.get('disease_weight_strength', 2.0)

        # Create data loader
        dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        # Metrics storage
        mse_list = []
        weighted_mse_list = []
        masked_mse_list = []
        ssim_full_list = []
        ssim_masked_list = []
        psnr_full_list = []
        psnr_masked_list = []
        inference_times = []

        # Sample reconstructions for visualization
        sample_reconstructions = []

        print(f"  Processing {len(val_dataset)} validation samples...")

        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"  Evaluating {model_name}")):
                # Move to device
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                masks = batch['mask'].to(self.device)

                # Measure inference time
                start_time = time.time()
                outputs = model(inputs)
                embeddings = model.encode(inputs)
                inference_time = time.time() - start_time
                inference_times.append(inference_time / inputs.size(0))  # Per image

                # Compute metrics for each sample in batch
                for i in range(outputs.size(0)):
                    pred = outputs[i]
                    target = targets[i]
                    mask = masks[i]

                    # 1. Standard MSE (unmasked)
                    mse_val = LossCalculator.masked_mse(
                        pred.unsqueeze(0),
                        target.unsqueeze(0),
                        mask=None,
                        weights=None
                    )
                    mse_list.append(float(mse_val))

                    # 2. Masked MSE (only leaf pixels)
                    masked_mse_val = LossCalculator.masked_mse(
                        pred.unsqueeze(0),
                        target.unsqueeze(0),
                        mask=mask.unsqueeze(0),
                        weights=None
                    )
                    masked_mse_list.append(float(masked_mse_val))

                    # 3. Weighted MSE (disease-based weighting)
                    if model_cfg.uses_weighting:
                        weights = LossCalculator.compute_lab_weights(
                            target.unsqueeze(0),
                            lab_stats,
                            disease_weight_strength=disease_weight_strength
                        )

                        # Apply mask if model uses it
                        if model_cfg.uses_masks:
                            weighted_mse_val = LossCalculator.masked_mse(
                                pred.unsqueeze(0),
                                target.unsqueeze(0),
                                mask=mask.unsqueeze(0),
                                weights=weights
                            )
                        else:
                            weighted_mse_val = LossCalculator.masked_mse(
                                pred.unsqueeze(0),
                                target.unsqueeze(0),
                                mask=None,
                                weights=weights
                            )
                        weighted_mse_list.append(float(weighted_mse_val))
                    else:
                        # For non-weighted models, weighted MSE = masked MSE
                        weighted_mse_list.append(float(masked_mse_val))

                    # 4. SSIM and PSNR (convert to numpy and RGB)
                    # Compute both full-image and masked-region metrics
                    pred_np = pred.cpu().numpy()  # CHW
                    target_np = target.cpu().numpy()  # CHW
                    mask_np = mask.cpu().numpy().squeeze()

                    # Denormalize LAB
                    pred_lab = denormalize_lab(pred_np, lab_stats)  # CHW
                    target_lab = denormalize_lab(target_np, lab_stats)  # CHW

                    # Convert to HWC for LAB to RGB conversion
                    pred_lab_hwc = pred_lab.transpose(1, 2, 0)
                    target_lab_hwc = target_lab.transpose(1, 2, 0)

                    # Convert LAB to RGB using evaluate.py method
                    pred_rgb = lab_to_rgb(pred_lab_hwc)
                    target_rgb = lab_to_rgb(target_lab_hwc)

                    # Compute SSIM (both full and masked)
                    try:
                        ssim_full, ssim_masked = LossCalculator.compute_ssim(
                            pred_rgb, target_rgb, mask_np,
                            compute_full=True, compute_masked=True
                        )
                        # Store full-image SSIM
                        ssim_full_list.append(ssim_full if ssim_full is not None else 0.0)
                        # Store masked-region SSIM (or NaN if not available)
                        ssim_masked_list.append(ssim_masked if ssim_masked is not None else float('nan'))
                    except Exception as e:
                        print(f"\nWarning: SSIM computation failed: {e}")
                        ssim_full_list.append(0.0)
                        ssim_masked_list.append(float('nan'))

                    # Compute PSNR (both full and masked)
                    try:
                        psnr_full, psnr_masked = LossCalculator.compute_psnr(
                            pred_rgb, target_rgb, mask_np,
                            compute_full=True, compute_masked=True
                        )
                        # Store full-image PSNR
                        psnr_full_list.append(psnr_full if psnr_full is not None else 0.0)
                        # Store masked-region PSNR (or NaN if not available)
                        psnr_masked_list.append(psnr_masked if psnr_masked is not None else float('nan'))
                    except Exception as e:
                        print(f"\nWarning: PSNR computation failed: {e}")
                        psnr_full_list.append(0.0)
                        psnr_masked_list.append(float('nan'))

                    # Store sample reconstructions (first batch only)
                    if batch_idx == 0 and len(sample_reconstructions) < self.config.num_sample_images:
                        sample_reconstructions.append({
                            'original_rgb': batch['original_rgb'][i].cpu().numpy(),
                            'target_lab': target_lab_hwc,
                            'target_rgb': target_rgb,
                            'prediction_lab': pred_lab_hwc,
                            'prediction_rgb': pred_rgb,
                            'mask': mask_np,
                            'metadata': batch['metadata']
                        })

        # Store sample reconstructions
        self.sample_reconstructions[model_name] = sample_reconstructions

        # Compute model statistics
        num_params = sum(p.numel() for p in model.parameters())
        model_size = self.get_model_size(model_name)
        latent_dim = config_dict.get('embedding_dim', 256)

        # Compute statistics for full and masked metrics
        # Use nanmean/nanstd to handle NaN values in masked metrics
        ssim_full_mean = float(np.mean(ssim_full_list))
        ssim_full_std = float(np.std(ssim_full_list))
        ssim_masked_mean = float(np.nanmean(ssim_masked_list)) if ssim_masked_list else 0.0
        ssim_masked_std = float(np.nanstd(ssim_masked_list)) if ssim_masked_list else 0.0

        psnr_full_mean = float(np.mean(psnr_full_list))
        psnr_full_std = float(np.std(psnr_full_list))
        psnr_masked_mean = float(np.nanmean(psnr_masked_list)) if psnr_masked_list else 0.0
        psnr_masked_std = float(np.nanstd(psnr_masked_list)) if psnr_masked_list else 0.0

        # Create metrics
        metrics = ModelMetrics(
            model_name=model_name,
            mse_mean=float(np.mean(mse_list)),
            mse_std=float(np.std(mse_list)),
            weighted_mse_mean=float(np.mean(weighted_mse_list)),
            weighted_mse_std=float(np.std(weighted_mse_list)),
            # Full-image metrics (entire image, ignoring mask)
            ssim_full_mean=ssim_full_mean,
            ssim_full_std=ssim_full_std,
            psnr_full_mean=psnr_full_mean,
            psnr_full_std=psnr_full_std,
            # Masked-region metrics (only masked pixels)
            ssim_masked_mean=ssim_masked_mean,
            ssim_masked_std=ssim_masked_std,
            psnr_masked_mean=psnr_masked_mean,
            psnr_masked_std=psnr_masked_std,
            masked_mse_mean=float(np.mean(masked_mse_list)),
            masked_mse_std=float(np.std(masked_mse_list)),
            inference_time_mean=float(np.mean(inference_times)),
            inference_time_std=float(np.std(inference_times)),
            num_parameters=num_params,
            model_size_mb=model_size,
            latent_dim=latent_dim,
            per_image_mse=mse_list,
            per_image_weighted_mse=weighted_mse_list,
            per_image_ssim_full=ssim_full_list,
            per_image_ssim_masked=ssim_masked_list,
            per_image_psnr_full=psnr_full_list,
            per_image_psnr_masked=psnr_masked_list
        )

        return metrics

    def extract_latent_embeddings(self):
        """Extract latent embeddings for all models."""
        print("\n" + "="*80)
        print("EXTRACTING LATENT EMBEDDINGS")
        print("="*80 + "\n")

        # Check if embeddings already exist
        for model_name in self.models.keys():
            model_dir = next(cfg.directory for cfg in self.config.models if cfg.name == model_name)
            embeddings_path = model_dir / "embeddings" / "latent_embeddings.npy"

            if embeddings_path.exists():
                print(f"Loading existing embeddings for {model_name}...")
                embeddings = np.load(embeddings_path)
                self.latent_embeddings[model_name] = embeddings
                print(f"  ✓ Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
            else:
                print(f"  ✗ No pre-computed embeddings found for {model_name}")
                print(f"    Expected: {embeddings_path}")
            print()


# ==============================================================================
# Latent Space Analysis
# ==============================================================================

class LatentSpaceAnalyzer:
    """Analyzer for latent space comparisons."""

    @staticmethod
    def compute_tsne(
        embeddings: np.ndarray,
        perplexity: int = 30,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Compute t-SNE dimensionality reduction.

        Args:
            embeddings: Embeddings array (N, D)
            perplexity: t-SNE perplexity parameter
            random_state: Random seed

        Returns:
            2D coordinates (N, 2)
        """
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            max_iter=1000
        )
        coords_2d = tsne.fit_transform(embeddings)
        return coords_2d

    @staticmethod
    def compute_umap(
        embeddings: np.ndarray,
        n_neighbors: int = 15,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Compute UMAP dimensionality reduction.

        Args:
            embeddings: Embeddings array (N, D)
            n_neighbors: UMAP n_neighbors parameter
            random_state: Random seed

        Returns:
            2D coordinates (N, 2)
        """
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                random_state=random_state
            )
            coords_2d = reducer.fit_transform(embeddings)
            return coords_2d
        except ImportError:
            print("  ✗ UMAP not installed. Install with: pip install umap-learn")
            return None


# ==============================================================================
# Visualization
# ==============================================================================

class Visualizer:
    """Visualization utilities for model comparison."""

    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.

        Args:
            output_dir: Output directory for plots
        """
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_curves(self, configs: Dict[str, Dict]):
        """
        Plot training/validation loss curves for all models.

        Args:
            configs: Dictionary of model configurations
        """
        print("\nPlotting training curves...")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for model_name, config_info in configs.items():
            history = config_info['training_history']

            if 'history' in history and len(history['history']) > 0:
                epochs = [h['epoch'] for h in history['history']]
                train_loss = [h['train_loss'] for h in history['history']]
                val_loss = [h['val_loss'] for h in history['history']]

                # Plot 1: Training loss
                axes[0].plot(epochs, train_loss, label=model_name, marker='o', markersize=3)

                # Plot 2: Validation loss
                axes[1].plot(epochs, val_loss, label=model_name, marker='o', markersize=3)

                # Plot 3: Both
                axes[2].plot(epochs, train_loss, label=f'{model_name} (train)',
                           linestyle='--', alpha=0.7)
                axes[2].plot(epochs, val_loss, label=f'{model_name} (val)',
                           linestyle='-', linewidth=2)

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Validation Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Training vs Validation Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.viz_dir / "loss_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved to {save_path}")

    def plot_latent_space_comparison(
        self,
        embeddings: Dict[str, np.ndarray],
        method: str = "t-SNE"
    ):
        """
        Plot latent space comparison for all models.

        Args:
            embeddings: Dictionary mapping model names to 2D coordinates
            method: Dimensionality reduction method name
        """
        print(f"\nPlotting {method} latent space comparison...")

        num_models = len(embeddings)
        fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))

        if num_models == 1:
            axes = [axes]

        for idx, (model_name, coords) in enumerate(embeddings.items()):
            if coords is not None:
                axes[idx].scatter(
                    coords[:, 0],
                    coords[:, 1],
                    s=10,
                    alpha=0.5,
                    c=np.arange(len(coords)),
                    cmap='viridis'
                )
                axes[idx].set_title(f'{model_name}\n{method} Projection')
                axes[idx].set_xlabel(f'{method} 1')
                axes[idx].set_ylabel(f'{method} 2')
                axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.viz_dir / f"latent_space_comparison_{method.lower()}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved to {save_path}")

    def plot_metrics_comparison(self, metrics: Dict[str, ModelMetrics]):
        """
        Plot comparison of key metrics across models.
        Shows both full-image and masked-region metrics.

        Args:
            metrics: Dictionary of model metrics
        """
        print("\nPlotting metrics comparison...")

        model_names = list(metrics.keys())

        # Extract metrics
        mse_means = [metrics[name].mse_mean for name in model_names]
        mse_stds = [metrics[name].mse_std for name in model_names]

        # Full and masked SSIM
        ssim_full_means = [metrics[name].ssim_full_mean for name in model_names]
        ssim_full_stds = [metrics[name].ssim_full_std for name in model_names]
        ssim_masked_means = [metrics[name].ssim_masked_mean for name in model_names]
        ssim_masked_stds = [metrics[name].ssim_masked_std for name in model_names]

        # Full and masked PSNR
        psnr_full_means = [metrics[name].psnr_full_mean for name in model_names]
        psnr_full_stds = [metrics[name].psnr_full_std for name in model_names]
        psnr_masked_means = [metrics[name].psnr_masked_mean for name in model_names]
        psnr_masked_stds = [metrics[name].psnr_masked_std for name in model_names]

        # Create figure with 2 rows, 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Row 1: MSE and full-image metrics
        # MSE
        axes[0, 0].bar(model_names, mse_means, yerr=mse_stds, capsize=5)
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].set_title('Mean Squared Error')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # SSIM (Full Image)
        axes[0, 1].bar(model_names, ssim_full_means, yerr=ssim_full_stds, capsize=5, color='orange')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].set_title('SSIM (Full Image)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].set_ylim([0, 1])

        # PSNR (Full Image)
        axes[0, 2].bar(model_names, psnr_full_means, yerr=psnr_full_stds, capsize=5, color='green')
        axes[0, 2].set_ylabel('PSNR (dB)')
        axes[0, 2].set_title('PSNR (Full Image)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3, axis='y')

        # Row 2: Masked-region metrics
        # Masked MSE (reuse for comparison)
        masked_mse_means = [metrics[name].masked_mse_mean for name in model_names]
        masked_mse_stds = [metrics[name].masked_mse_std for name in model_names]
        axes[1, 0].bar(model_names, masked_mse_means, yerr=masked_mse_stds, capsize=5, color='purple')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].set_title('MSE (Masked Region)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # SSIM (Masked Region)
        axes[1, 1].bar(model_names, ssim_masked_means, yerr=ssim_masked_stds, capsize=5, color='darkorange')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].set_title('SSIM (Masked Region)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].set_ylim([0, 1])

        # PSNR (Masked Region)
        axes[1, 2].bar(model_names, psnr_masked_means, yerr=psnr_masked_stds, capsize=5, color='darkgreen')
        axes[1, 2].set_ylabel('PSNR (dB)')
        axes[1, 2].set_title('PSNR (Masked Region)')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = self.viz_dir / "metrics_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved to {save_path}")

    def plot_error_distributions(self, metrics: Dict[str, ModelMetrics]):
        """
        Plot distribution of reconstruction errors.
        Shows both full-image and masked-region metrics.

        Args:
            metrics: Dictionary of model metrics
        """
        print("\nPlotting error distributions...")

        fig, axes = plt.subplots(3, 2, figsize=(14, 16))

        for model_name, model_metrics in metrics.items():
            if model_metrics.per_image_mse and len(model_metrics.per_image_mse) > 0:
                # MSE distribution
                axes[0, 0].hist(
                    model_metrics.per_image_mse,
                    bins=50,
                    alpha=0.5,
                    label=model_name
                )

                # Weighted MSE distribution
                axes[0, 1].hist(
                    model_metrics.per_image_weighted_mse,
                    bins=50,
                    alpha=0.5,
                    label=model_name
                )

                # SSIM (Full) distribution
                if model_metrics.per_image_ssim_full:
                    axes[1, 0].hist(
                        model_metrics.per_image_ssim_full,
                        bins=50,
                        alpha=0.5,
                        label=model_name
                    )

                # SSIM (Masked) distribution
                if model_metrics.per_image_ssim_masked:
                    # Filter out NaN values
                    ssim_masked_clean = [x for x in model_metrics.per_image_ssim_masked if not np.isnan(x)]
                    if ssim_masked_clean:
                        axes[1, 1].hist(
                            ssim_masked_clean,
                            bins=50,
                            alpha=0.5,
                            label=model_name
                        )

                # PSNR (Full) distribution
                if model_metrics.per_image_psnr_full:
                    axes[2, 0].hist(
                        model_metrics.per_image_psnr_full,
                        bins=50,
                        alpha=0.5,
                        label=model_name
                    )

                # PSNR (Masked) distribution
                if model_metrics.per_image_psnr_masked:
                    # Filter out NaN values
                    psnr_masked_clean = [x for x in model_metrics.per_image_psnr_masked if not np.isnan(x)]
                    if psnr_masked_clean:
                        axes[2, 1].hist(
                            psnr_masked_clean,
                            bins=50,
                            alpha=0.5,
                            label=model_name
                        )

        axes[0, 0].set_xlabel('MSE')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('MSE Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_xlabel('Weighted MSE')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Weighted MSE Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_xlabel('SSIM')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('SSIM Distribution (Full Image)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_xlabel('SSIM')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('SSIM Distribution (Masked Region)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        axes[2, 0].set_xlabel('PSNR (dB)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title('PSNR Distribution (Full Image)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].set_xlabel('PSNR (dB)')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].set_title('PSNR Distribution (Masked Region)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.viz_dir / "error_distributions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved to {save_path}")

    def plot_reconstruction_grid(
        self,
        sample_reconstructions: Dict[str, List[Dict]],
        num_samples: int = 5
    ):
        """
        Plot reconstruction grid showing original, predictions, and error maps.

        Args:
            sample_reconstructions: Dictionary mapping model names to sample reconstructions
            num_samples: Number of samples to show
        """
        print("\nPlotting reconstruction grid...")

        model_names = list(sample_reconstructions.keys())
        num_models = len(model_names)

        # Each sample gets a row, each model gets columns: original, reconstruction, difference
        num_cols = 1 + num_models * 2  # 1 for original, then (recon + diff) per model

        fig, axes = plt.subplots(num_samples, num_cols, figsize=(4 * num_cols, 4 * num_samples))

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for sample_idx in range(num_samples):
            # Column 0: Original image (same for all models)
            first_model = model_names[0]
            if sample_idx < len(sample_reconstructions[first_model]):
                original_rgb = sample_reconstructions[first_model][sample_idx]['original_rgb']
                axes[sample_idx, 0].imshow(original_rgb)
                axes[sample_idx, 0].set_title('Original' if sample_idx == 0 else '')
                axes[sample_idx, 0].axis('off')

                col_idx = 1
                for model_name in model_names:
                    if sample_idx < len(sample_reconstructions[model_name]):
                        sample = sample_reconstructions[model_name][sample_idx]

                        pred_rgb = sample['prediction_rgb']
                        target_rgb = sample['target_rgb']

                        # Reconstruction
                        axes[sample_idx, col_idx].imshow(pred_rgb)
                        if sample_idx == 0:
                            axes[sample_idx, col_idx].set_title(f'{model_name}\nReconstruction')
                        axes[sample_idx, col_idx].axis('off')

                        # Difference map
                        diff = np.abs(target_rgb.astype(float) - pred_rgb.astype(float))
                        diff_normalized = diff / diff.max() if diff.max() > 0 else diff

                        im = axes[sample_idx, col_idx + 1].imshow(diff_normalized, cmap='hot')
                        if sample_idx == 0:
                            axes[sample_idx, col_idx + 1].set_title(f'{model_name}\nError Map')
                        axes[sample_idx, col_idx + 1].axis('off')

                        col_idx += 2

        plt.tight_layout()
        save_path = self.viz_dir / "reconstruction_grid.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved to {save_path}")


# ==============================================================================
# Report Generation
# ==============================================================================

class ReportGenerator:
    """Generate comparison reports."""

    def __init__(self, output_dir: Path):
        """
        Initialize report generator.

        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_summary_table(self, metrics: Dict[str, ModelMetrics]):
        """
        Generate summary table in CSV format.

        Args:
            metrics: Dictionary of model metrics
        """
        print("\nGenerating summary table...")

        # Convert to DataFrame
        rows = []
        for model_name, model_metrics in metrics.items():
            rows.append(model_metrics.to_dict())

        df = pd.DataFrame(rows)

        # Save to CSV
        csv_path = self.output_dir / "summary_table.csv"
        df.to_csv(csv_path, index=False)

        print(f"  ✓ Saved to {csv_path}")

        # Also print to console
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        print(df.to_string(index=False))
        print()

    def generate_detailed_metrics(
        self,
        metrics: Dict[str, ModelMetrics],
        configs: Dict[str, Dict]
    ):
        """
        Generate detailed metrics JSON file.

        Args:
            metrics: Dictionary of model metrics
            configs: Dictionary of model configurations
        """
        print("Generating detailed metrics...")

        def convert_paths(obj):
            """Recursively convert Path and device objects to strings."""
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, torch.device):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            else:
                return obj

        detailed = {}

        for model_name in metrics.keys():
            detailed[model_name] = {
                'metrics': metrics[model_name].to_dict(),
                'configuration': convert_paths(configs[model_name]['config_dict']),
                'training_history': configs[model_name]['training_history']
            }

        json_path = self.output_dir / "detailed_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(detailed, f, indent=2)

        print(f"  ✓ Saved to {json_path}")

    def generate_interpretation(self, metrics: Dict[str, ModelMetrics]):
        """
        Generate interpretation of results.

        Args:
            metrics: Dictionary of model metrics
        """
        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)

        print("\nModel Characteristics:")
        print("-" * 80)

        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name}:")
            print(f"  Architecture: {model_metrics.num_parameters:,} parameters")
            print(f"  Model size: {model_metrics.model_size_mb:.2f} MB")
            print(f"  Latent dimension: {model_metrics.latent_dim}")

        print("\n" + "-" * 80)
        print("\nBest Model for Different Aspects:")
        print("-" * 80)

        # Find best models for different metrics
        best_mse = min(metrics.items(), key=lambda x: x[1].mse_mean)
        best_ssim_full = max(metrics.items(), key=lambda x: x[1].ssim_full_mean)
        best_ssim_masked = max(metrics.items(), key=lambda x: x[1].ssim_masked_mean)
        best_psnr_full = max(metrics.items(), key=lambda x: x[1].psnr_full_mean)
        best_psnr_masked = max(metrics.items(), key=lambda x: x[1].psnr_masked_mean)
        best_speed = min(metrics.items(), key=lambda x: x[1].inference_time_mean)

        print(f"\n  Reconstruction Accuracy (lowest MSE): {best_mse[0]}")
        print(f"    MSE: {best_mse[1].mse_mean:.6f}")

        print(f"\n  Full-Image Structural Similarity (highest SSIM): {best_ssim_full[0]}")
        print(f"    SSIM (full): {best_ssim_full[1].ssim_full_mean:.4f}")

        print(f"\n  Masked-Region Structural Similarity (highest SSIM): {best_ssim_masked[0]}")
        print(f"    SSIM (masked): {best_ssim_masked[1].ssim_masked_mean:.4f}")

        print(f"\n  Full-Image Peak Signal-to-Noise Ratio (highest PSNR): {best_psnr_full[0]}")
        print(f"    PSNR (full): {best_psnr_full[1].psnr_full_mean:.2f} dB")

        print(f"\n  Masked-Region Peak Signal-to-Noise Ratio (highest PSNR): {best_psnr_masked[0]}")
        print(f"    PSNR (masked): {best_psnr_masked[1].psnr_masked_mean:.2f} dB")

        print(f"\n  Inference Speed (fastest): {best_speed[0]}")
        print(f"    Time: {best_speed[1].inference_time_mean:.4f} seconds per image")

        print("\n" + "="*80)


# ==============================================================================
# Main Comparison Pipeline
# ==============================================================================

def run_comprehensive_comparison(
    output_dir: str = "comparison_results",
    num_eval_samples: Optional[int] = None,
    skip_latent: bool = False
):
    """
    Run comprehensive model comparison.

    Args:
        output_dir: Output directory for results
        num_eval_samples: Number of samples to evaluate (None for all)
        skip_latent: Skip latent space analysis
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE AUTOENCODER MODEL COMPARISON")
    print("="*80)

    # Define models to compare
    base_dir = Path(__file__).parent

    models = [
        ModelConfig(
            name="disease_autoencoder",
            directory=base_dir / "disease_autoencoder",
            description="Weighted MSE loss with SAM2.1 masks during training",
            uses_masks=True,
            uses_weighting=True
        ),
        ModelConfig(
            name="disease_autoencoder_cropped",
            directory=base_dir / "disease_autoencoder_cropped",
            description="Weighted MSE loss on pre-cropped images",
            uses_masks=False,
            uses_weighting=True
        ),
        ModelConfig(
            name="autoencoder_no_weighting",
            directory=base_dir / "autoencoder_no_weighting",
            description="Standard MSE loss on pre-cropped images",
            uses_masks=False,
            uses_weighting=False
        )
    ]

    # Create configuration
    config = ComparisonConfig(
        models=models,
        output_dir=Path(output_dir),
        batch_size=16,
        num_sample_images=10,
        random_seed=42
    )

    # Create output directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "visualizations").mkdir(exist_ok=True)
    (config.output_dir / "model_specific").mkdir(exist_ok=True)

    for model_cfg in models:
        (config.output_dir / "model_specific" / model_cfg.name).mkdir(exist_ok=True)

    # Initialize evaluator
    evaluator = ModelEvaluator(config)

    # Load models
    evaluator.load_models()

    # Evaluate models
    evaluator.evaluate_all_models(num_samples=num_eval_samples)

    # Extract latent embeddings
    if not skip_latent:
        evaluator.extract_latent_embeddings()

        # Perform dimensionality reduction
        tsne_coords = {}
        umap_coords = {}

        for model_name, embeddings in evaluator.latent_embeddings.items():
            if embeddings is not None:
                print(f"Computing t-SNE for {model_name}...")
                tsne_coords[model_name] = LatentSpaceAnalyzer.compute_tsne(
                    embeddings,
                    perplexity=config.tsne_perplexity,
                    random_state=config.random_seed
                )

                if config.use_umap:
                    print(f"Computing UMAP for {model_name}...")
                    umap_result = LatentSpaceAnalyzer.compute_umap(
                        embeddings,
                        n_neighbors=config.umap_n_neighbors,
                        random_state=config.random_seed
                    )
                    if umap_result is not None:
                        umap_coords[model_name] = umap_result

    # Generate visualizations
    visualizer = Visualizer(config.output_dir)

    visualizer.plot_training_curves(evaluator.configs)

    if not skip_latent:
        if tsne_coords:
            visualizer.plot_latent_space_comparison(tsne_coords, method="t-SNE")
        if umap_coords:
            visualizer.plot_latent_space_comparison(umap_coords, method="UMAP")

    visualizer.plot_metrics_comparison(evaluator.metrics)
    visualizer.plot_error_distributions(evaluator.metrics)
    visualizer.plot_reconstruction_grid(evaluator.sample_reconstructions, num_samples=min(5, config.num_sample_images))

    # Generate reports
    report_gen = ReportGenerator(config.output_dir)
    report_gen.generate_summary_table(evaluator.metrics)
    report_gen.generate_detailed_metrics(evaluator.metrics, evaluator.configs)
    report_gen.generate_interpretation(evaluator.metrics)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {config.output_dir}")
    print(f"  - Summary table: {config.output_dir / 'summary_table.csv'}")
    print(f"  - Detailed metrics: {config.output_dir / 'detailed_metrics.json'}")
    print(f"  - Visualizations: {config.output_dir / 'visualizations'}")
    print()


# ==============================================================================
# Command Line Interface
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive comparison of autoencoder models"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of validation samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--skip-latent",
        action="store_true",
        help="Skip latent space analysis"
    )

    args = parser.parse_args()

    run_comprehensive_comparison(
        output_dir=args.output_dir,
        num_eval_samples=args.num_samples,
        skip_latent=args.skip_latent
    )
