#!/usr/bin/env python3
"""
Test script for DINOv2 feature extraction.
Processes only 10 images to verify everything works.
"""

import sys
import time
from extract_dinov2_features import (
    DINOv2FeatureExtractor,
    find_images,
    save_features_to_csv,
    print_summary
)


def main():
    print("="*70)
    print("🧪 Testing DINOv2 Feature Extraction (10 images)")
    print("="*70)

    # Configuration
    MODEL_NAME = 'dinov2_vits14_reg'
    OUTPUT_CSV = 'test_dinov2_features.csv'
    DEVICE = 'cuda'
    BATCH_SIZE = 4  # Small batch for testing

    # Find images and take only first 10
    all_images = find_images("data/ne2025/device*")
    if not all_images:
        print("❌ No images found!")
        sys.exit(1)

    test_images = all_images[:10]

    print(f"\n🧪 Testing with {len(test_images)} images:")
    for img in test_images[:5]:  # Show first 5
        print(f"  - {img}")
    if len(test_images) > 5:
        print(f"  ... and {len(test_images) - 5} more")

    # Initialize extractor
    print("\n" + "="*70)
    try:
        extractor = DINOv2FeatureExtractor(model_name=MODEL_NAME, device=DEVICE)
    except Exception as e:
        print(f"❌ Failed to initialize extractor: {e}")
        sys.exit(1)

    # Extract features
    start_time = time.time()
    features_list, metadata_list, failed_images = extractor.extract_features_batch(
        test_images,
        batch_size=BATCH_SIZE
    )
    processing_time = time.time() - start_time

    if features_list:
        # Save to CSV
        save_features_to_csv(features_list, metadata_list, OUTPUT_CSV)

        # Print summary
        print_summary(features_list, failed_images, processing_time)

        print("\n✅ Test completed successfully!")
        print("   You can now run the full extraction with: python extract_dinov2_features.py")
    else:
        print("\n❌ Test failed - no features extracted")
        sys.exit(1)


if __name__ == "__main__":
    main()
