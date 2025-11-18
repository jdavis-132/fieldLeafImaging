import json
import numpy as np
import os

# Load image splits
with open('/home/schnable/Documents/fieldLeafImaging/src/autoencoder_no_weighting/logs/image_splits.json', 'r') as f:
    splits = json.load(f)

# Extract image paths from splits
train_images = [item['image_path'] for item in splits['train']]
test_images = [item['image_path'] for item in splits['test']]
val_images = [item['image_path'] for item in splits['val']]

# Count images in each split
train_count = len(train_images)
test_count = len(test_images)
val_count = len(val_images)
total_count = train_count + test_count + val_count

print("=" * 60)
print("IMAGE SPLITS SUMMARY")
print("=" * 60)
print(f"Train:      {train_count:5d} images")
print(f"Test:       {test_count:5d} images")
print(f"Validation: {val_count:5d} images")
print(f"Total:      {total_count:5d} images")
print()

# Load embeddings metadata
with open('/home/schnable/Documents/fieldLeafImaging/src/autoencoder_no_weighting/embeddings/embeddings_metadata.json', 'r') as f:
    metadata = json.load(f)

# Create set of embedded image paths
embedded_paths = {item['image_path'] for item in metadata}

# Count embeddings per split by checking if image paths are in embedded set
embeddings_train = sum(1 for img in train_images if img in embedded_paths)
embeddings_test = sum(1 for img in test_images if img in embedded_paths)
embeddings_val = sum(1 for img in val_images if img in embedded_paths)
embeddings_total = len(metadata)

print("=" * 60)
print("EMBEDDINGS SUMMARY")
print("=" * 60)
print(f"Train:      {embeddings_train:5d} embeddings")
print(f"Test:       {embeddings_test:5d} embeddings")
print(f"Validation: {embeddings_val:5d} embeddings")
print(f"Total:      {embeddings_total:5d} embeddings")
print()

# Check for missing embeddings
print("=" * 60)
print("COVERAGE CHECK")
print("=" * 60)
train_missing = train_count - embeddings_train
test_missing = test_count - embeddings_test
val_missing = val_count - embeddings_val
total_missing = total_count - embeddings_total

print(f"Train:      {train_missing:5d} missing ({embeddings_train/train_count*100:.1f}% coverage)")
print(f"Test:       {test_missing:5d} missing ({embeddings_test/test_count*100:.1f}% coverage)")
print(f"Validation: {val_missing:5d} missing ({embeddings_val/val_count*100:.1f}% coverage)")
print(f"Total:      {total_missing:5d} missing ({embeddings_total/total_count*100:.1f}% coverage)")
print()

# Load actual embeddings array to verify shape
embeddings = np.load('/home/schnable/Documents/fieldLeafImaging/src/autoencoder_no_weighting/embeddings/latent_embeddings.npy')
print("=" * 60)
print("EMBEDDINGS ARRAY INFO")
print("=" * 60)
print(f"Shape: {embeddings.shape}")
print(f"Expected: ({embeddings_total}, 256)")
print()

# Check if all images are accounted for
if total_missing == 0:
    print("✓ ALL EMBEDDINGS PRESENT - All images have embeddings saved!")
else:
    print(f"✗ MISSING EMBEDDINGS - {total_missing} images do not have embeddings")

    # Find which images are missing
    print("\nChecking for missing images...")

    # Check each split
    all_split_images = train_images + test_images + val_images
    missing_images = [img for img in all_split_images if img not in embedded_paths]

    if missing_images:
        print(f"\nMissing images ({len(missing_images)}):")
        for img in missing_images[:10]:  # Show first 10
            print(f"  - {img}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
