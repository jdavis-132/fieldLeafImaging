#!/usr/bin/env python3
"""
Extract image features using DINOv2 backbone model (ViT-S/14 distilled with registers).

This script:
1. Loads DINOv2 ViT-S/14 distilled model with registers
2. Extracts global image features (CLS token) for each image
3. Saves results to CSV with image paths and feature vectors

Author: Generated for fieldLeafImaging project
Date: 2025
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class DINOv2FeatureExtractor:
    """Extract features from images using DINOv2 model."""

    def __init__(self, model_name='dinov2_vits14_reg', device='cuda'):
        """
        Initialize DINOv2 model for feature extraction.

        Args:
            model_name (str): DINOv2 model variant
                Options: 'dinov2_vits14_reg', 'dinov2_vitb14_reg',
                        'dinov2_vitl14_reg', 'dinov2_vitg14_reg'
            device (str): Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device

        # Check CUDA availability
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            self.device = 'cpu'

        print(f"🔧 Loading DINOv2 model: {model_name}")
        print(f"   Device: {self.device.upper()}")

        try:
            # Load model from torch hub
            self.model = torch.hub.load(
                'facebookresearch/dinov2',
                model_name,
                pretrained=True
            )
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            print(f"✅ Model loaded successfully!")

            # Print GPU info if using CUDA
            if self.device == 'cuda':
                gpu_name = torch.cuda.get_device_name(0)
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"🚀 GPU: {gpu_name}")
                print(f"📊 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

            # Get feature dimension
            self.feature_dim = self._get_feature_dimension()
            print(f"📐 Feature dimension: {self.feature_dim}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            if self.device == 'cuda':
                print("🔄 Trying CPU fallback...")
                self.device = 'cpu'
                self.model = torch.hub.load(
                    'facebookresearch/dinov2',
                    model_name,
                    pretrained=True
                )
                self.model.to(self.device)
                self.model.eval()
                self.feature_dim = self._get_feature_dimension()
                print(f"✅ Model loaded successfully on CPU!")
                print(f"📐 Feature dimension: {self.feature_dim}")
            else:
                raise e

        # Define image preprocessing transform
        # DINOv2 uses 224x224 images with ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _get_feature_dimension(self):
        """
        Determine the feature dimension of the model.

        Returns:
            int: Feature dimension
        """
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            features = self.model(dummy_input)
        return features.shape[-1]

    def load_image(self, image_path):
        """
        Load and preprocess a single image.

        Args:
            image_path (str): Path to image file

        Returns:
            torch.Tensor or None: Preprocessed image tensor, or None if loading fails
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor
        except Exception as e:
            print(f"⚠️  Error loading {image_path}: {e}")
            return None

    def extract_features_single(self, image_tensor):
        """
        Extract features from a single image tensor.

        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor

        Returns:
            numpy.ndarray: Feature vector
        """
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            features = self.model(image_tensor)
            # features is shape [1, feature_dim], extract and convert to numpy
            features_np = features.squeeze(0).cpu().numpy()
        return features_np

    def extract_features_batch(self, image_paths, batch_size=32):
        """
        Extract features for a batch of images.

        Args:
            image_paths (list): List of image file paths
            batch_size (int): Number of images to process at once

        Returns:
            tuple: (features_list, metadata_list, failed_images)
        """
        features_list = []
        metadata_list = []
        failed_images = []

        print(f"\n🔍 Extracting features from {len(image_paths)} images...")
        print(f"   Batch size: {batch_size}")

        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches", unit="batch"):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            batch_valid_paths = []

            # Load batch
            for image_path in batch_paths:
                tensor = self.load_image(image_path)
                if tensor is not None:
                    batch_tensors.append(tensor)
                    batch_valid_paths.append(image_path)
                else:
                    failed_images.append(image_path)

            if not batch_tensors:
                continue

            # Stack tensors into batch
            try:
                batch_tensor = torch.stack(batch_tensors).to(self.device)

                # Extract features for batch
                with torch.no_grad():
                    batch_features = self.model(batch_tensor)
                    batch_features_np = batch_features.cpu().numpy()

                # Store results
                for j, path in enumerate(batch_valid_paths):
                    features_list.append(batch_features_np[j])
                    metadata_list.append(path)

                # Clear GPU cache periodically
                if self.device == 'cuda' and (i // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n⚠️  Error processing batch starting at index {i}: {e}")
                # Try processing images individually
                for path in batch_valid_paths:
                    try:
                        tensor = self.load_image(path)
                        if tensor is not None:
                            features = self.extract_features_single(tensor)
                            features_list.append(features)
                            metadata_list.append(path)
                    except Exception as e2:
                        print(f"\n⚠️  Failed to process {path}: {e2}")
                        failed_images.append(path)

        return features_list, metadata_list, failed_images


def find_images(base_pattern="data/ne2025/device*"):
    """
    Find all .jpg images matching the pattern.

    Args:
        base_pattern (str): Glob pattern for finding image directories

    Returns:
        list: List of image file paths
    """
    print(f"\n📂 Searching for images in pattern: {base_pattern}")

    # Find all directories matching the pattern
    dirs = glob(base_pattern)

    if not dirs:
        print(f"⚠️  No directories found matching pattern: {base_pattern}")
        return []

    # Find all .jpg files in these directories
    image_paths = []
    for dir_path in dirs:
        if os.path.isdir(dir_path):
            jpg_files = glob(os.path.join(dir_path, "*.jpg"))
            image_paths.extend(jpg_files)

    print(f"✅ Found {len(image_paths)} images in {len(dirs)} directories")

    return sorted(image_paths)


def save_features_to_csv(features_list, metadata_list, output_path):
    """
    Save features and metadata to CSV file.

    Args:
        features_list (list): List of feature vectors
        metadata_list (list): List of image paths
        output_path (str): Output CSV file path
    """
    print(f"\n💾 Saving features to CSV: {output_path}")

    # Convert features to numpy array
    features_array = np.array(features_list)

    print(f"   Features shape: {features_array.shape}")
    print(f"   Number of images: {len(metadata_list)}")
    print(f"   Feature dimensions: {features_array.shape[1]}")

    # Create column names
    feature_columns = [f"feature_{i}" for i in range(features_array.shape[1])]

    # Create DataFrame
    df = pd.DataFrame(features_array, columns=feature_columns)
    df.insert(0, 'image_path', metadata_list)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"✅ Saved {len(df)} features to {output_path}")
    print(f"   CSV shape: {df.shape}")
    print(f"   Columns: image_path + {len(feature_columns)} feature dimensions")


def print_summary(features_list, failed_images, processing_time):
    """
    Print summary statistics.

    Args:
        features_list (list): List of features
        failed_images (list): List of failed image paths
        processing_time (float): Total processing time in seconds
    """
    print("\n" + "="*70)
    print("📊 SUMMARY STATISTICS")
    print("="*70)

    total_images = len(features_list) + len(failed_images)
    success_rate = (len(features_list) / total_images * 100) if total_images > 0 else 0

    print(f"Total images found: {total_images}")
    print(f"Successfully processed: {len(features_list)}")
    print(f"Failed extractions: {len(failed_images)}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Processing time: {processing_time:.2f} seconds")

    if features_list:
        features_array = np.array(features_list)
        images_per_second = len(features_list) / processing_time if processing_time > 0 else 0
        print(f"Processing speed: {images_per_second:.2f} images/second")

        print(f"\nFeature statistics:")
        print(f"  Dimensions: {features_array.shape[1]}")
        print(f"  Mean: {features_array.mean():.4f}")
        print(f"  Std: {features_array.std():.4f}")
        print(f"  Min: {features_array.min():.4f}")
        print(f"  Max: {features_array.max():.4f}")

    if failed_images:
        print(f"\n⚠️  Failed images:")
        for img_path in failed_images[:10]:  # Show first 10
            print(f"  - {img_path}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")

    print("="*70)


def main():
    """Main execution function."""

    print("="*70)
    print("🎯 DINOv2 Feature Extraction")
    print("="*70)

    # Configuration
    MODEL_NAME = 'dinov2_vits14_reg'  # ViT-S/14 distilled with registers
    OUTPUT_CSV = 'dinov2_features.csv'
    IMAGE_PATTERN = 'data/ne2025/device*'
    DEVICE = 'cuda'  # Use CUDA for GPU acceleration
    BATCH_SIZE = 32  # Adjust based on GPU memory (32 should be safe for RTX 3060)

    # Find images
    image_paths = find_images(IMAGE_PATTERN)

    if not image_paths:
        print("❌ No images found! Exiting.")
        sys.exit(1)

    # Initialize extractor
    try:
        extractor = DINOv2FeatureExtractor(model_name=MODEL_NAME, device=DEVICE)
    except Exception as e:
        print(f"❌ Failed to initialize extractor: {e}")
        sys.exit(1)

    # Extract features
    start_time = time.time()
    features_list, metadata_list, failed_images = extractor.extract_features_batch(
        image_paths,
        batch_size=BATCH_SIZE
    )
    processing_time = time.time() - start_time

    if not features_list:
        print("❌ No features extracted! Exiting.")
        sys.exit(1)

    # Save to CSV
    save_features_to_csv(features_list, metadata_list, OUTPUT_CSV)

    # Print summary
    print_summary(features_list, failed_images, processing_time)

    print(f"\n✅ Feature extraction complete!")
    print(f"📁 Output saved to: {os.path.abspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    main()
