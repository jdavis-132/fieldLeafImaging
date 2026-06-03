#!/usr/bin/env python3
"""
Extract pooled image embeddings from SAM2.1 for all images in the dataset.

This script:
1. Loads SAM2.1 model and uses its image encoder
2. Extracts embeddings for each image
3. Applies global average pooling to get a single vector per image
4. Saves results to a CSV file

Author: Generated for fieldLeafImaging project
Date: 2025
"""

import torch
import numpy as np
import cv2
import pandas as pd
import os
import sys
from pathlib import Path
from glob import glob
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import warnings
warnings.filterwarnings('ignore')


class SAM2EmbeddingExtractor:
    """Extract pooled embeddings from SAM2.1 image encoder."""

    def __init__(self, model_path="models/sam2.1_hiera_tiny.pt", device="cuda"):
        """
        Initialize SAM 2.1 model for embedding extraction.

        Args:
            model_path (str): Path to the model checkpoint
            device (str): Device to run the model on ('cuda', 'cpu')
        """
        self.device = device
        self.model_path = model_path

        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            self.device = "cpu"

        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Expected at: {os.path.abspath(model_path)}"
            )

        print(f"🔧 Loading SAM 2.1 model on {self.device.upper()}...")

        # Build the model - using SAM2.1 hiera tiny config
        try:
            config_path = "configs/sam2.1/sam2.1_hiera_t.yaml"  # SAM2.1 config
            self.sam2_model = build_sam2(config_path, model_path, device=self.device)
            self.sam2_model = self.sam2_model.to(self.device)
            self.sam2_model.eval()  # Set to evaluation mode

            # Create predictor for image preprocessing
            self.predictor = SAM2ImagePredictor(self.sam2_model)

            print(f"✅ Model loaded successfully on {self.device.upper()}!")

            # Print GPU memory usage if on CUDA
            if self.device == "cuda":
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🚀 GPU: {gpu_name}")
                print(f"📊 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            if self.device == "cuda":
                print("🔄 Trying CPU fallback...")
                self.device = "cpu"
                self.sam2_model = build_sam2(config_path, model_path, device=self.device)
                self.sam2_model = self.sam2_model.to(self.device)
                self.sam2_model.eval()
                self.predictor = SAM2ImagePredictor(self.sam2_model)
                print("✅ Model loaded successfully on CPU!")
            else:
                raise e

    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess image for SAM2.1.

        Args:
            image_path (str): Path to the image

        Returns:
            numpy.ndarray: RGB image array, or None if loading fails
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"⚠️  Error loading {image_path}: {e}")
            return None

    def extract_embedding(self, image):
        """
        Extract image embedding using SAM2.1 image encoder.

        Args:
            image (numpy.ndarray): RGB image array

        Returns:
            numpy.ndarray: Pooled embedding vector (1D)
        """
        with torch.no_grad():
            # Set the image (this preprocesses and encodes it)
            self.predictor.set_image(image)

            # Get the image embedding from the predictor
            # This returns the full feature map from the image encoder
            features = self.predictor.get_image_embedding()

            # Apply global average pooling across spatial dimensions
            # Features shape is typically [B, C, H, W] or [C, H, W]
            if features.dim() == 4:
                # Shape: [B, C, H, W] -> [B, C]
                pooled = features.mean(dim=[2, 3])
            elif features.dim() == 3:
                # Shape: [C, H, W] -> [C]
                pooled = features.mean(dim=[1, 2])
            elif features.dim() == 2:
                # Shape: [H, W] -> scalar (unlikely but handle it)
                pooled = features.mean()
            else:
                # Already 1D or unexpected shape
                pooled = features

            # Convert to numpy and ensure 1D
            embedding = pooled.squeeze().cpu().numpy()

            # Ensure it's 1D
            if embedding.ndim > 1:
                embedding = embedding.flatten()

            return embedding

    def extract_embeddings_batch(self, image_paths, batch_size=1):
        """
        Extract embeddings for a batch of images.

        Note: SAM2.1's predictor processes images individually,
        so we process them one at a time but with progress tracking.

        Args:
            image_paths (list): List of image file paths
            batch_size (int): Batch size (kept at 1 for SAM2.1 compatibility)

        Returns:
            tuple: (embeddings_list, metadata_list, failed_images)
        """
        embeddings_list = []
        metadata_list = []
        failed_images = []

        print(f"\n🔍 Extracting embeddings from {len(image_paths)} images...")

        for image_path in tqdm(image_paths, desc="Processing images", unit="img"):
            # Load image
            image = self.load_and_preprocess_image(image_path)

            if image is None:
                failed_images.append(image_path)
                continue

            try:
                # Extract embedding
                embedding = self.extract_embedding(image)

                # Store results
                embeddings_list.append(embedding)
                metadata_list.append(image_path)

                # Clear CUDA cache periodically to avoid memory issues
                if self.device == "cuda" and len(embeddings_list) % 100 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n⚠️  Failed to extract embedding for {image_path}: {e}")
                failed_images.append(image_path)
                continue

        return embeddings_list, metadata_list, failed_images


def find_images(base_pattern="data/ne2020/device*"):
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
        print(f"   Checking if data/ne2025/device* exists instead...")
        dirs = glob("data/ne2025/device*")
        if dirs:
            print(f"✅ Found data/ne2025/ instead, using that pattern")
            base_pattern = "data/ne2025/device*"
        else:
            return []

    # Find all .jpg files in these directories
    image_paths = []
    for dir_path in dirs:
        if os.path.isdir(dir_path):
            jpg_files = glob(os.path.join(dir_path, "*.jpg"))
            image_paths.extend(jpg_files)

    print(f"✅ Found {len(image_paths)} images in {len(dirs)} directories")

    return sorted(image_paths)


def save_embeddings_to_csv(embeddings_list, metadata_list, output_path):
    """
    Save embeddings and metadata to CSV file.

    Args:
        embeddings_list (list): List of embedding vectors
        metadata_list (list): List of image paths
        output_path (str): Output CSV file path
    """
    print(f"\n💾 Saving embeddings to CSV: {output_path}")

    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings_list)

    print(f"   Embeddings shape: {embeddings_array.shape}")
    print(f"   Number of images: {len(metadata_list)}")
    print(f"   Embedding dimensions: {embeddings_array.shape[1]}")

    # Create column names
    embedding_columns = [f"embedding_{i}" for i in range(embeddings_array.shape[1])]

    # Create DataFrame
    df = pd.DataFrame(embeddings_array, columns=embedding_columns)
    df.insert(0, 'image_path', metadata_list)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"✅ Saved {len(df)} embeddings to {output_path}")
    print(f"   CSV shape: {df.shape}")
    print(f"   Columns: image_path + {len(embedding_columns)} embedding dimensions")


def print_summary(embeddings_list, failed_images, embedding_dim=None):
    """
    Print summary statistics.

    Args:
        embeddings_list (list): List of embeddings
        failed_images (list): List of failed image paths
        embedding_dim (int, optional): Embedding dimension
    """
    print("\n" + "="*70)
    print("📊 SUMMARY STATISTICS")
    print("="*70)

    total_images = len(embeddings_list) + len(failed_images)
    success_rate = (len(embeddings_list) / total_images * 100) if total_images > 0 else 0

    print(f"Total images processed: {total_images}")
    print(f"Successfully extracted: {len(embeddings_list)}")
    print(f"Failed extractions: {len(failed_images)}")
    print(f"Success rate: {success_rate:.2f}%")

    if embeddings_list:
        embeddings_array = np.array(embeddings_list)
        print(f"\nEmbedding dimensions: {embeddings_array.shape[1]}")
        print(f"Embedding statistics:")
        print(f"  Mean: {embeddings_array.mean():.4f}")
        print(f"  Std: {embeddings_array.std():.4f}")
        print(f"  Min: {embeddings_array.min():.4f}")
        print(f"  Max: {embeddings_array.max():.4f}")

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
    print("🎯 SAM2.1 Image Embedding Extraction")
    print("="*70)

    # Configuration
    MODEL_PATH = "src/sam2/models/sam2.1_hiera_tiny.pt"
    OUTPUT_CSV = "src/sam2/sam2_image_embeddings.csv"
    IMAGE_PATTERN = "data/ne2020/device*"
    DEVICE = "cuda"  # Use CUDA for GPU acceleration
    BATCH_SIZE = 1  # SAM2.1 processes images individually

    # Create output directory if needed
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Find images
    image_paths = find_images(IMAGE_PATTERN)

    if not image_paths:
        print("❌ No images found! Exiting.")
        sys.exit(1)

    # Initialize extractor
    try:
        extractor = SAM2EmbeddingExtractor(model_path=MODEL_PATH, device=DEVICE)
    except Exception as e:
        print(f"❌ Failed to initialize extractor: {e}")
        sys.exit(1)

    # Extract embeddings
    embeddings_list, metadata_list, failed_images = extractor.extract_embeddings_batch(
        image_paths,
        batch_size=BATCH_SIZE
    )

    if not embeddings_list:
        print("❌ No embeddings extracted! Exiting.")
        sys.exit(1)

    # Save to CSV
    save_embeddings_to_csv(embeddings_list, metadata_list, OUTPUT_CSV)

    # Print summary
    print_summary(embeddings_list, failed_images)

    print("\n✅ Embedding extraction complete!")
    print(f"📁 Output saved to: {os.path.abspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    main()
