"""
Dataset for VAE with LAB color space preprocessing.

This dataset uses the exact same preprocessing pipeline as the standard autoencoder
to ensure consistent comparisons. It loads images, applies LAB color space conversion,
cropping, and normalization.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class VAELeafDataset(Dataset):
    """
    Dataset for VAE (reuses preprocessing from standard autoencoder).

    Preprocessing:
    1. Load image and mask
    2. Crop to leaf bounding box with padding
    3. Resize to 224x224
    4. Convert RGB to LAB color space
    5. Normalize LAB using pre-computed statistics
    """

    def __init__(self, image_metadata_list, config, lab_stats, transform=None):
        """
        Args:
            image_metadata_list: List of dicts with image info
            config: VAEConfig object
            lab_stats: Dict with 'lab_mean' and 'lab_std'
            transform: Optional transforms (for training augmentation)
        """
        self.image_metadata = image_metadata_list
        self.config = config
        self.transform = transform

        # LAB normalization statistics (from training set)
        self.lab_mean = np.array(lab_stats['lab_mean'])
        self.lab_std = np.array(lab_stats['lab_std'])

    def __len__(self):
        return len(self.image_metadata)

    def get_leaf_bbox(self, mask):
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
        y_min, y_max, x_min, x_max = self.get_leaf_bbox(combined_mask)
        image = image[y_min:y_max, x_min:x_max]
        combined_mask = combined_mask[y_min:y_max, x_min:x_max]

        # Resize
        image = cv2.resize(image, (self.config.image_size, self.config.image_size),
                          interpolation=cv2.INTER_LINEAR)
        combined_mask = cv2.resize(combined_mask, (self.config.image_size, self.config.image_size),
                                   interpolation=cv2.INTER_NEAREST)

        # Convert to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Normalize LAB channels
        lab_image = (lab_image - self.lab_mean) / (self.lab_std + 1e-8)

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


def get_dataloaders(splits, lab_stats, config):
    """
    Create DataLoaders for train, validation, and test sets.

    Args:
        splits: Dict with 'train', 'val', 'test' image metadata lists
        lab_stats: Dict with 'lab_mean' and 'lab_std'
        config: VAEConfig object

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    dataloaders = {}

    # Training dataset with augmentation
    train_dataset = VAELeafDataset(
        image_metadata_list=splits['train'],
        config=config,
        lab_stats=lab_stats,
        transform=SyncedTransform(config)
    )

    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device.type == 'cuda' else False,
        drop_last=True
    )

    # Validation and test datasets without augmentation
    for split_name in ['val', 'test']:
        dataset = VAELeafDataset(
            image_metadata_list=splits[split_name],
            config=config,
            lab_stats=lab_stats,
            transform=None
        )

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

    return dataloaders
