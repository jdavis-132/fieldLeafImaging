#!/usr/bin/env python3
"""
Extract image features using DINOv2 backbone model (ViT-L/14 distilled with registers).

For each image, computes the mean and standard deviation of each DINOv2 feature
across all spatial patch tokens. This yields 2 * feature_dim values per image
(mean_0..mean_N, std_0..std_N) and captures both the average activation and
the spatial variability of each feature across the image.
"""

import torch
from torchvision.transforms import v2
import cv2
import pandas as pd
import numpy as np
import os
import sys
import time
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class DINOv2PatchStatsExtractor:
    """Extract per-patch mean and std features from images using DINOv2."""

    def __init__(self, model_name='dinov2_vitl14_reg', device='cuda'):
        self.model_name = model_name
        self.device = device

        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = 'cpu'

        print(f"Loading DINOv2 model: {model_name} on {self.device.upper()}")

        try:
            self.model = torch.hub.load(
                'facebookresearch/dinov2',
                model_name,
                pretrained=True
            )
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully")

            if self.device == 'cuda':
                print(f"GPU: {torch.cuda.get_device_name(0)}")

            self.feature_dim = self._get_feature_dimension()
            print(f"Feature dimension: {self.feature_dim} (output columns: {self.feature_dim * 2})")

        except Exception as e:
            print(f"Error loading model: {e}")
            if self.device == 'cuda':
                print("Trying CPU fallback...")
                self.device = 'cpu'
                self.model = torch.hub.load(
                    'facebookresearch/dinov2',
                    model_name,
                    pretrained=True
                )
                self.model.to(self.device)
                self.model.eval()
                self.feature_dim = self._get_feature_dimension()
                print(f"Model loaded on CPU. Feature dimension: {self.feature_dim}")
            else:
                raise
        # uses the image transforms from https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py self.transform; 
        self.transform = v2.Compose([
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(1008, 1008)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _get_feature_dimension(self):
        dummy = torch.randn(1, 3, 1008, 1008).to(self.device)
        with torch.no_grad():
            out = self.model.forward_features(dummy)
        return out['x_norm_patchtokens'].shape[-1]

    def load_image(self, image_path):
        try:
            bgr = cv2.imread(image_path)
            if bgr is None:
                raise ValueError(f"cv2.imread returned None for {image_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            return self.transform(tensor)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def _patch_stats(self, patch_tokens):
        """Compute mean and std across patches. Input: [batch, num_patches, feature_dim]."""
        mean = patch_tokens.mean(dim=1)   # [batch, feature_dim]
        std = patch_tokens.std(dim=1)     # [batch, feature_dim]
        return torch.cat([mean, std], dim=1)  # [batch, feature_dim * 2]

    def extract_features_single(self, image_tensor):
        with torch.no_grad():
            x = image_tensor.unsqueeze(0).to(self.device)
            out = self.model.forward_features(x)
            combined = self._patch_stats(out['x_norm_patchtokens'])
            return combined.squeeze(0).cpu().numpy()

    def extract_features_batch(self, image_paths, batch_size=32):
        features_list = []
        metadata_list = []
        failed_images = []

        print(f"\nExtracting patch-stats features from {len(image_paths)} images (batch_size={batch_size})...")

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches", unit="batch"):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            batch_valid_paths = []

            for path in batch_paths:
                tensor = self.load_image(path)
                if tensor is not None:
                    batch_tensors.append(tensor)
                    batch_valid_paths.append(path)
                else:
                    failed_images.append(path)

            if not batch_tensors:
                continue

            try:
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                with torch.no_grad():
                    out = self.model.forward_features(batch_tensor)
                    combined = self._patch_stats(out['x_norm_patchtokens'])
                    batch_features_np = combined.cpu().numpy()

                for j, path in enumerate(batch_valid_paths):
                    features_list.append(batch_features_np[j])
                    metadata_list.append(path)

                if self.device == 'cuda' and (i // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\nError processing batch at index {i}: {e}")
                for path in batch_valid_paths:
                    try:
                        tensor = self.load_image(path)
                        if tensor is not None:
                            features = self.extract_features_single(tensor)
                            features_list.append(features)
                            metadata_list.append(path)
                    except Exception as e2:
                        print(f"\nFailed to process {path}: {e2}")
                        failed_images.append(path)

        return features_list, metadata_list, failed_images


def find_images(base_pattern="data/processed/ne2025/device*"):
    print(f"\nSearching for images: {base_pattern}")
    dirs = glob(base_pattern)
    if not dirs:
        print(f"No directories found matching: {base_pattern}")
        return []
    image_paths = []
    for dir_path in dirs:
        if os.path.isdir(dir_path):
            image_paths.extend(glob(os.path.join(dir_path, "*.png")))
    print(f"Found {len(image_paths)} images in {len(dirs)} directories")
    return sorted(image_paths)


def save_features_to_csv(features_list, metadata_list, output_path):
    print(f"\nSaving features to: {output_path}")
    features_array = np.array(features_list)
    n_features = features_array.shape[1] // 2
    mean_cols = [f"mean_{i}" for i in range(n_features)]
    std_cols = [f"std_{i}" for i in range(n_features)]
    df = pd.DataFrame(features_array, columns=mean_cols + std_cols)
    df.insert(0, 'image_path', metadata_list)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows x {features_array.shape[1]} feature columns to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract DINOv2 patch-stats features from images.")
    parser.add_argument('image_dir', help='Directory (or glob pattern) of images to process')
    parser.add_argument('output_csv', help='Path to write the output CSV')
    args = parser.parse_args()

    MODEL_NAME = 'dinov2_vitl14_reg'
    OUTPUT_CSV = args.output_csv
    IMAGE_PATTERN = args.image_dir
    DEVICE = 'cuda'
    BATCH_SIZE = 16

    image_paths = find_images(IMAGE_PATTERN)
    if not image_paths:
        print("No images found. Exiting.")
        sys.exit(1)

    try:
        extractor = DINOv2PatchStatsExtractor(model_name=MODEL_NAME, device=DEVICE)
    except Exception as e:
        print(f"Failed to initialize extractor: {e}")
        sys.exit(1)

    start_time = time.time()
    features_list, metadata_list, failed_images = extractor.extract_features_batch(
        image_paths, batch_size=BATCH_SIZE
    )
    elapsed = time.time() - start_time

    if not features_list:
        print("No features extracted. Exiting.")
        sys.exit(1)

    save_features_to_csv(features_list, metadata_list, OUTPUT_CSV)

    total = len(features_list) + len(failed_images)
    print(f"\nDone: {len(features_list)}/{total} images in {elapsed:.1f}s "
          f"({len(features_list)/elapsed:.1f} img/s)")
    if failed_images:
        print(f"Failed: {len(failed_images)} images")


if __name__ == "__main__":
    main()
