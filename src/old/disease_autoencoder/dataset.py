"""
Dataset for disease-aware autoencoder with LAB color space preprocessing.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path


class DiseaseLeafDataset(Dataset):
    """
    Dataset for disease-aware autoencoder.

    Preprocessing:
    1. Load image and mask
    2. Crop to leaf bounding box with padding
    3. Resize to 224x224
    4. Convert RGB to LAB color space
    5. Normalize LAB using masked pixel statistics
    6. Set background pixels to 0
    """

    def __init__(self, image_metadata_list, config, transform=None, compute_stats=False):
        """
        Args:
            image_metadata_list: List of dicts with image info
            config: DiseaseConfig object
            transform: Optional transforms (for training augmentation)
            compute_stats: If True, compute LAB statistics from this dataset
        """
        self.image_metadata = image_metadata_list
        self.config = config
        self.transform = transform

        # LAB normalization statistics (computed from training set)
        self.lab_mean = None
        self.lab_std = None

        if compute_stats:
            self._compute_lab_statistics()

    def __len__(self):
        return len(self.image_metadata)

    def _compute_lab_statistics(self):
        """Compute mean and std of LAB channels from masked pixels only using online computation."""
        print("Computing LAB statistics from masked pixels...")

        # Online statistics computation (Welford's algorithm)
        n = 0
        mean_L, mean_a, mean_b = 0.0, 0.0, 0.0
        M2_L, M2_a, M2_b = 0.0, 0.0, 0.0

        for idx in range(len(self.image_metadata)):
            metadata = self.image_metadata[idx]

            # Load image and masks
            image = cv2.imread(metadata['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask0 = cv2.imread(metadata['mask0_path'], cv2.IMREAD_GRAYSCALE)
            mask1 = cv2.imread(metadata['mask1_path'], cv2.IMREAD_GRAYSCALE)
            combined_mask = np.logical_or(mask0 > 0, mask1 > 0)

            # Convert to LAB
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

            # Extract masked pixels
            masked_pixels = lab_image[combined_mask]

            if len(masked_pixels) > 0:
                # Update statistics using Welford's online algorithm
                for pixel in masked_pixels:
                    n += 1

                    # L channel
                    delta_L = pixel[0] - mean_L
                    mean_L += delta_L / n
                    M2_L += delta_L * (pixel[0] - mean_L)

                    # a channel
                    delta_a = pixel[1] - mean_a
                    mean_a += delta_a / n
                    M2_a += delta_a * (pixel[1] - mean_a)

                    # b channel
                    delta_b = pixel[2] - mean_b
                    mean_b += delta_b / n
                    M2_b += delta_b * (pixel[2] - mean_b)

        # Finalize statistics
        std_L = np.sqrt(M2_L / n) if n > 1 else 0.0
        std_a = np.sqrt(M2_a / n) if n > 1 else 0.0
        std_b = np.sqrt(M2_b / n) if n > 1 else 0.0

        self.lab_mean = np.array([mean_L, mean_a, mean_b])
        self.lab_std = np.array([std_L, std_a, std_b])

        print(f"LAB mean: {self.lab_mean}")
        print(f"LAB std: {self.lab_std}")

    def set_lab_statistics(self, mean, std):
        """Set LAB normalization statistics (from training set)."""
        self.lab_mean = np.array(mean)
        self.lab_std = np.array(std)

    def _get_leaf_bbox(self, mask):
        """Get bounding box of leaf from mask with padding."""
        coords = np.column_stack(np.where(mask > 0))

        if len(coords) == 0:
            # Return full image if no mask
            return 0, mask.shape[0], 0, mask.shape[1]

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Add padding
        pad = self.config.crop_padding
        y_min = max(0, y_min - pad)
        y_max = min(mask.shape[0], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(mask.shape[1], x_max + pad)

        return y_min, y_max, x_min, x_max

    def __getitem__(self, idx):
        """
        Load and preprocess image.

        Returns:
            dict with:
                'input': torch.Tensor (4, H, W) - LAB + mask
                'target': torch.Tensor (3, H, W) - LAB image
                'mask': torch.Tensor (1, H, W) - binary mask
                'metadata': dict
        """
        metadata = self.image_metadata[idx]

        # Load image
        image = cv2.imread(metadata['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load masks and combine
        mask0 = cv2.imread(metadata['mask0_path'], cv2.IMREAD_GRAYSCALE)
        mask1 = cv2.imread(metadata['mask1_path'], cv2.IMREAD_GRAYSCALE)
        combined_mask = np.logical_or(mask0 > 0, mask1 > 0).astype(np.uint8)

        # Get bounding box and crop
        y_min, y_max, x_min, x_max = self._get_leaf_bbox(combined_mask)
        image = image[y_min:y_max, x_min:x_max]
        combined_mask = combined_mask[y_min:y_max, x_min:x_max]

        # Resize
        image = cv2.resize(image, (self.config.image_size, self.config.image_size),
                          interpolation=cv2.INTER_LINEAR)
        combined_mask = cv2.resize(combined_mask, (self.config.image_size, self.config.image_size),
                                   interpolation=cv2.INTER_NEAREST)

        # Convert to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Normalize LAB channels using masked pixel statistics
        if self.lab_mean is not None and self.lab_std is not None:
            lab_image = (lab_image - self.lab_mean) / (self.lab_std + 1e-8)

        # Set background pixels to 0
        mask_3ch = np.stack([combined_mask] * 3, axis=2)
        lab_image = lab_image * mask_3ch

        # Convert to torch tensors (HWC -> CHW)
        lab_tensor = torch.from_numpy(lab_image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(combined_mask).unsqueeze(0).float()

        # Apply transforms (augmentation) if provided
        if self.transform:
            # Stack for synchronized transforms
            stacked = torch.cat([lab_tensor, mask_tensor], dim=0)
            stacked = self.transform(stacked)
            lab_tensor = stacked[:3]
            mask_tensor = stacked[3:4]

        # Create input (LAB + mask)
        input_tensor = torch.cat([lab_tensor, mask_tensor], dim=0)

        return {
            'input': input_tensor,  # (4, H, W)
            'target': lab_tensor,   # (3, H, W)
            'mask': mask_tensor,    # (1, H, W)
            'metadata': metadata
        }


class SyncedTransform:
    """Apply transforms to image and mask synchronously."""

    def __init__(self, config):
        self.config = config

    def __call__(self, tensor):
        """
        Apply random augmentation to stacked tensor (LAB + mask).

        Safe augmentations for disease analysis:
        - 90-degree rotations
        - Horizontal/vertical flips
        - Minimal brightness adjustment (±5% on L channel only)
        """
        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            tensor = torch.rot90(tensor, k=k, dims=[1, 2])

        # Random horizontal flip
        if np.random.rand() < 0.5:
            tensor = torch.flip(tensor, dims=[2])

        # Random vertical flip
        if np.random.rand() < 0.5:
            tensor = torch.flip(tensor, dims=[1])

        # Minimal brightness adjustment (only L channel, not mask)
        if np.random.rand() < 0.5:
            brightness_factor = 1.0 + np.random.uniform(-0.05, 0.05)
            tensor[0] = tensor[0] * brightness_factor  # Only L channel

        return tensor


def get_dataloaders(splits, config):
    """
    Create DataLoaders for train, validation, and test sets.

    Args:
        splits: Dict with 'train', 'val', 'test' image metadata lists
        config: DiseaseConfig object

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
        dict: {'lab_mean': array, 'lab_std': array}
    """
    # Create training dataset and compute LAB statistics
    train_dataset = DiseaseLeafDataset(
        image_metadata_list=splits['train'],
        config=config,
        transform=None,
        compute_stats=True
    )

    # Get LAB statistics
    lab_stats = {
        'lab_mean': train_dataset.lab_mean.tolist(),
        'lab_std': train_dataset.lab_std.tolist()
    }

    # Save statistics
    import json
    stats_path = config.logs_dir / 'lab_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(lab_stats, f, indent=2)
    print(f"LAB statistics saved to {stats_path}")

    dataloaders = {}

    # Training dataset with augmentation
    train_dataset_aug = DiseaseLeafDataset(
        image_metadata_list=splits['train'],
        config=config,
        transform=SyncedTransform(config)
    )
    train_dataset_aug.set_lab_statistics(train_dataset.lab_mean, train_dataset.lab_std)

    dataloaders['train'] = DataLoader(
        train_dataset_aug,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device.type == 'cuda' else False,
        drop_last=True
    )

    # Validation and test datasets without augmentation
    for split_name in ['val', 'test']:
        dataset = DiseaseLeafDataset(
            image_metadata_list=splits[split_name],
            config=config,
            transform=None
        )
        dataset.set_lab_statistics(train_dataset.lab_mean, train_dataset.lab_std)

        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True if config.device.type == 'cuda' else False,
            drop_last=False
        )

    print(f"\nDataLoader Summary:")
    for split_name in ['train', 'val', 'test']:
        dataset_size = len(dataloaders[split_name].dataset)
        num_batches = len(dataloaders[split_name])
        print(f"  {split_name.upper()}: {dataset_size} images, {num_batches} batches")

    return dataloaders, lab_stats
