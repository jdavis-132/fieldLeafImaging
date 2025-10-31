"""
PyTorch Dataset for masked leaf images.

Loads normalized images, combines two masks, and applies masking.
"""

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path


class MaskedLeafDataset(Dataset):
    """
    PyTorch Dataset for masked leaf images.

    Each sample consists of:
    - Normalized RGB image
    - Two segmentation masks (combined with logical OR)
    - Masked image (pixels outside mask set to black)
    """

    def __init__(self, image_metadata_list, config, transform=None):
        """
        Args:
            image_metadata_list: List of dicts with image info (from prepare_splits.py)
            config: Config object with paths and parameters
            transform: Optional torchvision transforms
        """
        self.image_metadata = image_metadata_list
        self.config = config
        self.transform = transform

        # Default transform if none provided
        if self.transform is None:
            self.transform = self.get_default_transform()

    def __len__(self):
        return len(self.image_metadata)

    def __getitem__(self, idx):
        """
        Load and return a masked image.

        Returns:
            dict: {
                'masked_image': torch.Tensor (3, H, W),
                'filename': str,
                'plot': int,
                'genotype': str,
                'device': int
            }
        """
        metadata = self.image_metadata[idx]

        # Load normalized image
        image = cv2.imread(metadata['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Load both masks
        mask0 = cv2.imread(metadata['mask0_path'], cv2.IMREAD_GRAYSCALE)
        mask1 = cv2.imread(metadata['mask1_path'], cv2.IMREAD_GRAYSCALE)

        # Combine masks with logical OR
        combined_mask = np.logical_or(mask0 > 0, mask1 > 0).astype(np.uint8)

        # Apply mask to image
        # Create 3-channel mask
        mask_3ch = np.stack([combined_mask] * 3, axis=2)

        # Set pixels OUTSIDE mask to black [0, 0, 0]
        masked_image = image * mask_3ch

        # Resize to target size
        masked_image = cv2.resize(
            masked_image,
            self.config.image_size,  # (width, height)
            interpolation=cv2.INTER_LINEAR
        )

        # Convert to float [0, 1]
        masked_image = masked_image.astype(np.float32) / 255.0

        # Apply transforms (ToTensor + Normalize)
        if self.transform:
            masked_image = self.transform(masked_image)

        return {
            'masked_image': masked_image,
            'filename': metadata['filename'],
            'plot': metadata['plot'],
            'genotype': metadata['genotype'],
            'device': metadata['device'],
            'split': metadata.get('split', 'unknown')
        }

    def get_default_transform(self):
        """Get default transform: Normalize with ImageNet stats."""
        return transforms.Compose([
            transforms.ToTensor(),  # Converts to (C, H, W) and scales to [0, 1]
            transforms.Normalize(
                mean=self.config.mean,
                std=self.config.std
            )
        ])

    def get_raw_item(self, idx):
        """
        Get item without normalization (for visualization).

        Returns image in [0, 255] range as numpy array.
        """
        metadata = self.image_metadata[idx]

        # Load normalized image
        image = cv2.imread(metadata['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load both masks
        mask0 = cv2.imread(metadata['mask0_path'], cv2.IMREAD_GRAYSCALE)
        mask1 = cv2.imread(metadata['mask1_path'], cv2.IMREAD_GRAYSCALE)

        # Combine masks
        combined_mask = np.logical_or(mask0 > 0, mask1 > 0).astype(np.uint8)

        # Apply mask
        mask_3ch = np.stack([combined_mask] * 3, axis=2)
        masked_image = image * mask_3ch

        # Resize
        masked_image = cv2.resize(
            masked_image,
            self.config.image_size,
            interpolation=cv2.INTER_LINEAR
        )

        return {
            'masked_image': masked_image,  # (H, W, 3) in [0, 255]
            'original_image': cv2.resize(image, self.config.image_size),
            'mask0': cv2.resize(mask0, self.config.image_size),
            'mask1': cv2.resize(mask1, self.config.image_size),
            'combined_mask': cv2.resize(combined_mask * 255, self.config.image_size),
            'metadata': metadata
        }


def get_dataloaders(splits, config):
    """
    Create DataLoaders for train, validation, and test sets.

    Args:
        splits: Dict with 'train', 'val', 'test' image metadata lists
        config: Config object

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    dataloaders = {}

    for split_name in ['train', 'val', 'test']:
        dataset = MaskedLeafDataset(
            image_metadata_list=splits[split_name],
            config=config
        )

        # Use shuffle and data augmentation for training only
        is_train = (split_name == 'train')

        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=is_train,
            num_workers=config.num_workers,
            pin_memory=True if config.device.type == 'cuda' else False,
            drop_last=is_train  # Drop last incomplete batch for training
        )

        dataloaders[split_name] = dataloader

        print(f"{split_name.upper()} DataLoader: {len(dataset)} images, "
              f"{len(dataloader)} batches")

    return dataloaders


def denormalize(tensor, config):
    """
    Denormalize a tensor using config mean and std.

    Args:
        tensor: Tensor of shape (C, H, W) or (B, C, H, W)
        config: Config object with mean and std

    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor(config.mean).view(-1, 1, 1)
    std = torch.tensor(config.std).view(-1, 1, 1)

    if tensor.dim() == 4:  # Batch
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    # Move to same device as tensor
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)

    return tensor * std + mean
