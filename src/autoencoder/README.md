# Leaf Image Dataset for Autoencoder

This directory contains a PyTorch Dataset implementation for loading preprocessed leaf images with genotype-based splitting to prevent data leakage.

## Overview

The dataset infrastructure provides:
- **Genotype-based splitting**: Each genotype appears in only ONE split (train/val/test)
- **Flexible configuration**: YAML-based configuration for easy experimentation
- **Metadata support**: Automatically matches images with metadata from CSV files
- **Preprocessing**: Automatic resizing to target dimensions
- **Memory caching**: Optional image caching for faster training

## Files

- `config.yaml`: Configuration file for dataset parameters
- `dataset.py`: PyTorch Dataset class implementation
- `data_utils.py`: Utility functions for creating splits and DataLoaders
- `example_dataloader.py`: Example script demonstrating usage
- `README.md`: This documentation file

## Quick Start

### 1. Configure your dataset

Edit `config.yaml` to specify your data paths and parameters:

```yaml
data:
  environment: "test_output"  # or "aamu2025", "fvsu2025", etc.
  colorspace: "RGB"           # or "LAB", "HSV", etc.
  use_masked: false           # true to use masked images
  metadata_csv: "data/test_output/brightness_statistics.csv"

image:
  target_size: [256, 256]     # (height, width)

split:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  random_seed: 42
```

### 2. Run the example script

```bash
cd /Users/jensinadavis/Documents/fieldLeafImaging
python src/autoencoder/example_dataloader.py
```

This will:
- Load the configuration
- Create train/val/test splits
- Verify no genotype leakage
- Display batch statistics
- Save visualizations of sample batches

### 3. Use in your training code

```python
from src.autoencoder.data_utils import create_dataloaders_from_config

# Create DataLoaders
train_loader, val_loader, test_loader = create_dataloaders_from_config()

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        images = batch['image']          # Shape: (B, 3, H, W)
        genotypes = batch['genotype']    # List of genotype IDs
        metadata = batch['metadata']     # Dictionary of metadata fields

        # Your training code here
        outputs = model(images)
        loss = criterion(outputs, images)
        # ...
```

## Data Structure

### Expected Directory Structure

Images should be organized as:
```
data/
├── {environment}/
│   └── {subdir}/  (optional)
│       ├── cropped_{colorspace}_normalized/
│       │   ├── 1201_LeafPhotoA_2025-09-08 10_44_12.793-05_00_0.png
│       │   ├── 1201_LeafPhotoA_2025-09-08 10_44_12.793-05_00_1.png
│       │   └── ...
│       └── cropped_{colorspace}_normalized_masked/  (if use_masked=true)
│           └── ...
└── {environment}_metadata.csv
```

### Image Naming Convention

Images must follow this naming convention:
```
{genotype}_{description}_{timestamp}_{leaf_number}.png
```

Example: `1201_LeafPhotoA_2025-09-08 10_44_12.793-05_00_0.png`
- Genotype: `1201`
- Description: `LeafPhotoA`
- Timestamp: `2025-09-08 10_44_12.793-05_00`
- Leaf number: `0`

The genotype (first part before underscore) is used for splitting data.

### Metadata CSV

The metadata CSV should include at minimum:
- `image_name`: Base name of the image (without leaf number suffix)
- Other columns: Any additional metadata (brightness stats, vegetation indices, etc.)

Example:
```csv
image_name,image_width,image_height,hsv_v_mean,hsv_v_std,...
1201_LeafPhotoA_2025-09-08 10_44_12.793-05_00,3060,4080,128.85,...
2752_LeafPhotoA_2025-09-09 16_37_44.997-05_00,3060,4080,142.87,...
```

## API Reference

### Dataset Class

```python
from src.autoencoder.dataset import LeafImageDataset

dataset = LeafImageDataset(
    image_paths=list_of_paths,
    metadata_df=pandas_dataframe,
    target_size=(256, 256),
    colorspace='RGB',
    use_masked=False,
    cache_images=False
)

# Get a sample
sample = dataset[0]
# Returns: {
#     'image': torch.Tensor,        # Shape: (3, H, W), values in [0, 1]
#     'metadata': dict,              # Metadata from CSV
#     'image_path': str,             # Path to image file
#     'genotype': str                # Genotype identifier
# }

# Get genotype information
genotypes = dataset.get_genotypes()  # List of unique genotypes
indices = dataset.get_samples_by_genotype('1201')  # Indices for genotype
```

### Data Utilities

```python
from src.autoencoder.data_utils import (
    load_config,
    create_datasets_from_config,
    create_dataloaders_from_config,
    verify_no_genotype_leakage
)

# Load configuration
config = load_config('src/autoencoder/config.yaml')

# Create datasets
train_ds, val_ds, test_ds = create_datasets_from_config(config)

# Create DataLoaders
train_loader, val_loader, test_loader = create_dataloaders_from_config(config)

# Verify no data leakage
verify_no_genotype_leakage(train_ds, val_ds, test_ds)
```

## Configuration Options

### Data Section
- `base_dir`: Base directory for data
- `environment`: Environment subdirectory (e.g., 'test_output', 'aamu2025')
- `subdir`: Optional subdirectory within environment
- `metadata_csv`: Path to metadata CSV file
- `genotype_file`: Optional genotype reference file

### Image Section
- `colorspace`: Colorspace name ('RGB', 'LAB', 'HSV', etc.)
- `use_masked`: Whether to use masked images
- `target_size`: [height, width] for resizing

### Split Section
- `train_ratio`: Fraction for training (default: 0.70)
- `val_ratio`: Fraction for validation (default: 0.15)
- `test_ratio`: Fraction for testing (default: 0.15)
- `random_seed`: Seed for reproducibility (default: 42)

### DataLoader Section
- `batch_size`: Number of samples per batch (default: 32)
- `num_workers`: Number of data loading workers (default: 4)
- `shuffle_train`: Shuffle training data (default: true)
- `shuffle_val`: Shuffle validation data (default: false)
- `shuffle_test`: Shuffle test data (default: false)
- `drop_last`: Drop incomplete last batch (default: false)
- `pin_memory`: Pin memory for faster GPU transfer (default: true)

### Settings Section
- `verbose`: Print detailed information (default: true)
- `cache_images`: Cache images in memory (default: false)

## Genotype-Based Splitting

The splitting strategy ensures no data leakage:

1. **Extract genotypes**: Parse genotype ID from image filenames
2. **Group by genotype**: Collect all images for each genotype
3. **Split genotypes**: Randomly assign genotypes to train/val/test
4. **Assign samples**: Place all images of a genotype in the same split

This ensures that if a model sees images from genotype "1201" during training, it will NEVER see any other images from "1201" during validation or testing.

### Verification

Always verify no leakage after creating datasets:

```python
verify_no_genotype_leakage(train_dataset, val_dataset, test_dataset)
```

Output:
```
✓ No genotype leakage detected!
  Train genotypes: 28
  Val genotypes:   6
  Test genotypes:  6
```

## Image Preprocessing

Images are already preprocessed and stored as PNG files with pixel values normalized to [0, 1]. The dataset:

1. Loads PNG images using OpenCV
2. Converts from BGR to RGB colorspace
3. Resizes to target dimensions using bilinear interpolation
4. Converts to PyTorch tensor with shape (C, H, W)
5. Ensures values are in [0, 1] range

**No additional normalization or augmentation is applied** unless you add custom transforms.

## Memory Considerations

For large datasets:
- Set `cache_images: false` in config
- Reduce `num_workers` if running out of memory
- Use smaller `batch_size`

For small datasets that fit in RAM:
- Set `cache_images: true` for faster training
- Images will be cached after first access

## Example Output

Running `example_dataloader.py`:

```
================================================================================
Leaf Image Dataset Example
================================================================================

1. Loading configuration...
   Config loaded from: src/autoencoder/config.yaml
   Environment: test_output
   Colorspace: RGB
   Target size: [256, 256]

2. Creating datasets...
Loading images from: data/test_output/cropped_RGB_normalized
Loading metadata from: data/test_output/brightness_statistics.csv
Total images: 240
Unique genotypes: 40

Genotype split:
  Train: 28 genotypes (70.0%)
  Val:   6 genotypes (15.0%)
  Test:  6 genotypes (15.0%)

Image split:
  Train: 168 images (70.0%)
  Val:   36 images (15.0%)
  Test:  36 images (15.0%)

3. Verifying no genotype leakage...
✓ No genotype leakage detected!
  Train genotypes: 28
  Val genotypes:   6
  Test genotypes:  6

...
```

## Troubleshooting

### "Image directory does not exist"
- Check `environment` and `colorspace` in config.yaml
- Verify directory naming: `cropped_{colorspace}_normalized`
- Ensure colorspace is uppercase in directory name (e.g., `RGB` not `rgb`)

### "No images found in directory"
- Check that `.png` files exist in the directory
- Verify file extension is `.png` (not `.jpg` or other)

### "Missing metadata"
- Ensure `image_name` column exists in metadata CSV
- Check that base names (without leaf number) match metadata entries
- Verify CSV is properly formatted

### "Split ratios must sum to 1.0"
- Ensure train_ratio + val_ratio + test_ratio = 1.0
- Use ratios like 0.70, 0.15, 0.15 (not percentages like 70, 15, 15)

## Advanced Usage

### Custom Transforms

To add data augmentation:

```python
from torchvision import transforms

# Define transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# Note: Current implementation doesn't support transforms parameter
# You would need to modify the Dataset class to accept and apply transforms
```

### Accessing Specific Genotypes

```python
# Get all samples from a specific genotype
genotype_indices = dataset.get_samples_by_genotype('1201')
genotype_subset = torch.utils.data.Subset(dataset, genotype_indices)

# Create a DataLoader for just this genotype
genotype_loader = DataLoader(genotype_subset, batch_size=4)
```

### Changing Splits After Creation

```python
from src.autoencoder.data_utils import create_genotype_splits

# Create new splits with different ratios
train_idx, val_idx, test_idx = create_genotype_splits(
    image_paths,
    train_ratio=0.80,
    val_ratio=0.10,
    test_ratio=0.10,
    random_seed=123  # Different seed for different split
)
```

## Citation

If you use this dataset infrastructure in your research, please cite:

```bibtex
@software{leaf_image_dataset,
  title={Leaf Image Dataset for Autoencoder Training},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/fieldLeafImaging}
}
```

## License

[Specify your license here]

## Contact

For questions or issues, please contact [your contact information].
