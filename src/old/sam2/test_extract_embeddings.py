#!/usr/bin/env python3
"""
Test script for SAM2.1 embedding extraction.
Processes only 5 images to verify everything works.
"""

import sys
from extract_embeddings import (
    SAM2EmbeddingExtractor,
    find_images,
    save_embeddings_to_csv,
    print_summary
)

def main():
    print("="*70)
    print("🧪 Testing SAM2.1 Embedding Extraction (5 images)")
    print("="*70)

    # Configuration
    MODEL_PATH = "src/sam2/models/sam2.1_hiera_tiny.pt"
    OUTPUT_CSV = "src/sam2/test_embeddings.csv"
    DEVICE = "cuda"

    # Find images and take only first 5
    all_images = find_images("data/ne2020/device*")
    test_images = all_images[:5]

    print(f"\n🧪 Testing with {len(test_images)} images:")
    for img in test_images:
        print(f"  - {img}")

    # Initialize extractor
    print("\n" + "="*70)
    extractor = SAM2EmbeddingExtractor(model_path=MODEL_PATH, device=DEVICE)

    # Extract embeddings
    embeddings_list, metadata_list, failed_images = extractor.extract_embeddings_batch(test_images)

    if embeddings_list:
        # Save to CSV
        save_embeddings_to_csv(embeddings_list, metadata_list, OUTPUT_CSV)

        # Print summary
        print_summary(embeddings_list, failed_images)

        print("\n✅ Test completed successfully!")
        print("   You can now run the full extraction with: python src/sam2/extract_embeddings.py")
    else:
        print("\n❌ Test failed - no embeddings extracted")
        sys.exit(1)

if __name__ == "__main__":
    main()
