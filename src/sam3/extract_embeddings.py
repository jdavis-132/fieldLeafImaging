#!/usr/bin/env python3
"""
Extract pooled image embeddings from SAM3 for all images in the dataset.

This script:
1. Loads SAM3 model and uses its image encoder
2. Extracts embeddings for each image
3. Applies global average pooling (mean) and standard deviation across spatial dimensions
4. Saves results to a CSV file with both mean and std embeddings

Author: Generated for fieldLeafImaging project
Date: 2025
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm
from transformers import Sam3Processor, Sam3Model
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class SAM3EmbeddingExtractor:
    """Extract pooled embeddings (mean and std) from SAM3 image encoder."""

    def __init__(self, model_path="src/sam3", device="cuda"):
        """
        Initialize SAM3 model for embedding extraction.

        Args:
            model_path (str): Path to the model directory
            device (str): Device to run the model on ('cuda', 'cpu')
        """
        self.device = device
        self.model_path = model_path

        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            self.device = "cpu"

        # Verify model directory exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model directory not found: {model_path}\n"
                f"Expected at: {os.path.abspath(model_path)}"
            )

        print(f"🔧 Loading SAM3 model on {self.device.upper()}...")

        # Load the model using transformers
        try:
            self.model = Sam3Model.from_pretrained(model_path).to(self.device)
            self.model.eval()  # Set to evaluation mode

            # Load processor for image preprocessing
            self.processor = Sam3Processor.from_pretrained(model_path)

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
                self.model = Sam3Model.from_pretrained(model_path).to(self.device)
                self.model.eval()
                self.processor = Sam3Processor.from_pretrained(model_path)
                print("✅ Model loaded successfully on CPU!")
            else:
                raise e

    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess image for SAM3.

        Args:
            image_path (str): Path to the image

        Returns:
            PIL.Image: RGB image, or None if loading fails
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            print(f"⚠️  Error loading {image_path}: {e}")
            return None

    def extract_embedding(self, image):
        """
        Extract image embedding using SAM3 image encoder.

        Args:
            image (PIL.Image): RGB image

        Returns:
            tuple: (mean_embedding, std_embedding) - both are 1D numpy arrays
                   mean_embedding: Global average pooling across spatial dimensions
                   std_embedding: Standard deviation across spatial dimensions
        """
        with torch.no_grad():
            # Process image - SAM3 uses text prompts, but we can use empty/dummy text
            # for pure image encoding
            inputs = self.processor(images=image, text="", return_tensors="pt").to(self.device)

            # Get the model outputs
            outputs = self.model(**inputs, output_hidden_states=True)

            # Extract image embeddings from the vision encoder
            # SAM3 model has a vision_encoder that produces image features
            # We'll use the last hidden state from the vision model
            if hasattr(outputs, 'vision_hidden_states') and outputs.vision_hidden_states is not None:
                # Use the last layer's hidden states
                features = outputs.vision_hidden_states[-1]
            elif hasattr(outputs, 'image_embeddings') and outputs.image_embeddings is not None:
                features = outputs.image_embeddings
            else:
                # Fallback: try to extract from the model's vision encoder directly
                # Re-process through vision encoder only
                pixel_values = inputs.pixel_values
                vision_outputs = self.model.vision_encoder(pixel_values, output_hidden_states=True)
                features = vision_outputs.last_hidden_state

            # Apply global average pooling and std across spatial dimensions
            # Features shape is typically [B, H*W, C] or [B, C, H, W]
            if features.dim() == 4:
                # Shape: [B, C, H, W] -> [B, C]
                pooled_mean = features.mean(dim=[2, 3])
                pooled_std = features.std(dim=[2, 3])
            elif features.dim() == 3:
                # Shape: [B, N, C] -> [B, C] (average over sequence length N)
                pooled_mean = features.mean(dim=1)
                pooled_std = features.std(dim=1)
            elif features.dim() == 2:
                # Already [B, C] - no spatial dimensions to compute std over
                pooled_mean = features
                pooled_std = torch.zeros_like(features)
            else:
                # Unexpected shape, flatten and hope for the best
                pooled_mean = features.flatten(1)
                pooled_std = torch.zeros_like(pooled_mean)

            # Convert to numpy and ensure 1D
            embedding_mean = pooled_mean.squeeze().cpu().numpy()
            embedding_std = pooled_std.squeeze().cpu().numpy()

            # Ensure they're 1D
            if embedding_mean.ndim > 1:
                embedding_mean = embedding_mean.flatten()
            if embedding_std.ndim > 1:
                embedding_std = embedding_std.flatten()

            return embedding_mean, embedding_std

    def extract_embeddings_batch(self, image_paths, batch_size=1):
        """
        Extract embeddings for a batch of images.

        Note: SAM3's processor processes images individually,
        so we process them one at a time but with progress tracking.

        Args:
            image_paths (list): List of image file paths
            batch_size (int): Batch size (kept at 1 for SAM3 compatibility)

        Returns:
            tuple: (embeddings_mean_list, embeddings_std_list, metadata_list, failed_images)
        """
        embeddings_mean_list = []
        embeddings_std_list = []
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
                # Extract embedding (mean and std)
                embedding_mean, embedding_std = self.extract_embedding(image)

                # Store results
                embeddings_mean_list.append(embedding_mean)
                embeddings_std_list.append(embedding_std)
                metadata_list.append(image_path)

                # Clear CUDA cache periodically to avoid memory issues
                if self.device == "cuda" and len(embeddings_mean_list) % 100 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n⚠️  Failed to extract embedding for {image_path}: {e}")
                failed_images.append(image_path)
                continue

        return embeddings_mean_list, embeddings_std_list, metadata_list, failed_images


def find_images(base_pattern):
    """
    Find all image files matching the glob pattern.

    Args:
        base_pattern (str): Glob pattern for finding images

    Returns:
        list: List of image file paths
    """
    print(f"\n📂 Searching for images matching pattern: {base_pattern}")

    # Find all files matching the pattern
    image_paths = glob(base_pattern, recursive=True)

    # Filter for common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = [p for p in image_paths if Path(p).suffix.lower() in image_extensions]

    if not image_paths:
        print(f"⚠️  No images found matching pattern: {base_pattern}")
        return []

    print(f"✅ Found {len(image_paths)} images")

    return sorted(image_paths)


def save_embeddings_to_csv(embeddings_mean_list, embeddings_std_list, metadata_list, output_path):
    """
    Save embeddings (mean and std) and metadata to CSV file.

    Args:
        embeddings_mean_list (list): List of mean embedding vectors
        embeddings_std_list (list): List of std embedding vectors
        metadata_list (list): List of image paths
        output_path (str): Output CSV file path
    """
    print(f"\n💾 Saving embeddings to CSV: {output_path}")

    # Convert embeddings to numpy arrays
    embeddings_mean_array = np.array(embeddings_mean_list)
    embeddings_std_array = np.array(embeddings_std_list)

    embedding_dim = embeddings_mean_array.shape[1]

    print(f"   Mean embeddings shape: {embeddings_mean_array.shape}")
    print(f"   Std embeddings shape: {embeddings_std_array.shape}")
    print(f"   Number of images: {len(metadata_list)}")
    print(f"   Embedding dimensions: {embedding_dim}")

    # Create column names for mean and std
    mean_columns = [f"embedding_mean_{i}" for i in range(embedding_dim)]
    std_columns = [f"embedding_std_{i}" for i in range(embedding_dim)]

    # Create DataFrame with mean embeddings
    df = pd.DataFrame(embeddings_mean_array, columns=mean_columns)

    # Add std embeddings
    df_std = pd.DataFrame(embeddings_std_array, columns=std_columns)
    df = pd.concat([df, df_std], axis=1)

    # Insert image path as first column
    df.insert(0, 'image_path', metadata_list)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"✅ Saved {len(df)} embeddings to {output_path}")
    print(f"   CSV shape: {df.shape}")
    print(f"   Columns: image_path + {embedding_dim} mean dimensions + {embedding_dim} std dimensions")


def print_summary(embeddings_mean_list, embeddings_std_list, failed_images, embedding_dim=None):
    """
    Print summary statistics.

    Args:
        embeddings_mean_list (list): List of mean embeddings
        embeddings_std_list (list): List of std embeddings
        failed_images (list): List of failed image paths
        embedding_dim (int, optional): Embedding dimension
    """
    print("\n" + "="*70)
    print("📊 SUMMARY STATISTICS")
    print("="*70)

    total_images = len(embeddings_mean_list) + len(failed_images)
    success_rate = (len(embeddings_mean_list) / total_images * 100) if total_images > 0 else 0

    print(f"Total images processed: {total_images}")
    print(f"Successfully extracted: {len(embeddings_mean_list)}")
    print(f"Failed extractions: {len(failed_images)}")
    print(f"Success rate: {success_rate:.2f}%")

    if embeddings_mean_list:
        embeddings_mean_array = np.array(embeddings_mean_list)
        embeddings_std_array = np.array(embeddings_std_list)
        print(f"\nEmbedding dimensions: {embeddings_mean_array.shape[1]}")
        print(f"Mean embedding statistics:")
        print(f"  Mean: {embeddings_mean_array.mean():.4f}")
        print(f"  Std: {embeddings_mean_array.std():.4f}")
        print(f"  Min: {embeddings_mean_array.min():.4f}")
        print(f"  Max: {embeddings_mean_array.max():.4f}")
        print(f"Std embedding statistics:")
        print(f"  Mean: {embeddings_std_array.mean():.4f}")
        print(f"  Std: {embeddings_std_array.std():.4f}")
        print(f"  Min: {embeddings_std_array.min():.4f}")
        print(f"  Max: {embeddings_std_array.max():.4f}")

    if failed_images:
        print(f"\n⚠️  Failed images:")
        for img_path in failed_images[:10]:  # Show first 10
            print(f"  - {img_path}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")

    print("="*70)


def main():
    """Main execution function."""

    parser = argparse.ArgumentParser(
        description="Extract SAM3 image embeddings and save to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract embeddings from all JPG images in a directory
  python extract_embeddings.py "data/ne2025/device1/*.jpg" -o embeddings.csv

  # Extract from multiple directories using wildcard
  python extract_embeddings.py "data/ne2025/device*/*.jpg" -o embeddings.csv

  # Extract from nested directories
  python extract_embeddings.py "data/**/*.jpg" -o embeddings.csv

  # Use CPU instead of GPU
  python extract_embeddings.py "data/*.jpg" -o embeddings.csv --device cpu

  # Specify custom model path
  python extract_embeddings.py "data/*.jpg" -o embeddings.csv --model /path/to/sam3
        """
    )

    parser.add_argument(
        "image_pattern",
        type=str,
        help="Glob pattern for finding images (e.g., 'data/ne2025/device*/*.jpg')"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="src/sam3",
        help="Path to SAM3 model directory (default: src/sam3)"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference (default: cuda)"
    )

    args = parser.parse_args()

    print("="*70)
    print("🎯 SAM3 Image Embedding Extraction")
    print("="*70)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Find images
    image_paths = find_images(args.image_pattern)

    if not image_paths:
        print("❌ No images found! Exiting.")
        sys.exit(1)

    # Initialize extractor
    try:
        extractor = SAM3EmbeddingExtractor(model_path=args.model, device=args.device)
    except Exception as e:
        print(f"❌ Failed to initialize extractor: {e}")
        sys.exit(1)

    # Extract embeddings
    embeddings_mean_list, embeddings_std_list, metadata_list, failed_images = extractor.extract_embeddings_batch(
        image_paths,
        batch_size=1
    )

    if not embeddings_mean_list:
        print("❌ No embeddings extracted! Exiting.")
        sys.exit(1)

    # Save to CSV
    save_embeddings_to_csv(embeddings_mean_list, embeddings_std_list, metadata_list, args.output)

    # Print summary
    print_summary(embeddings_mean_list, embeddings_std_list, failed_images)

    print("\n✅ Embedding extraction complete!")
    print(f"📁 Output saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
