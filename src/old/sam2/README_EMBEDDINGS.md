# SAM2.1 Image Embedding Extraction

## Overview

Scripts to extract pooled image embeddings from SAM2.1 (Segment Anything Model 2.1) for all images in the dataset.

## Files

- **`extract_embeddings.py`**: Main script to extract embeddings from all images
- **`test_extract_embeddings.py`**: Test script to verify extraction works (processes only 5 images)
- **`sam2_image_embeddings.csv`**: Output CSV file with embeddings (created after running)

## Requirements

- SAM2.1 installed (`sam2` package)
- Model checkpoint: `src/sam2/models/sam2.1_hiera_tiny.pt`
- CUDA-capable GPU (NVIDIA GeForce RTX 3060 or similar)
- Python packages: `torch`, `numpy`, `pandas`, `cv2`, `tqdm`

## Quick Start

### 1. Test the extraction (recommended first)

```bash
python src/sam2/test_extract_embeddings.py
```

This processes only 5 images to verify everything works correctly. It should complete in a few seconds and create `src/sam2/test_embeddings.csv`.

**Expected output:**
- ✅ Model loaded successfully on CUDA
- 🔍 Extracted 5 embeddings
- 📊 Success rate: 100%
- 💾 CSV with 256-dimensional embeddings

### 2. Run full extraction

```bash
python src/sam2/extract_embeddings.py
```

This processes all images in `data/ne2025/device*/` directories.

**Expected processing:**
- ~6,147 images total
- ~256-dimensional embeddings per image
- Processing speed: ~4-5 images/second on RTX 3060
- Estimated time: ~20-25 minutes
- Output: `src/sam2/sam2_image_embeddings.csv`

## Output Format

The CSV file has the following structure:

| image_path | embedding_0 | embedding_1 | ... | embedding_255 |
|------------|-------------|-------------|-----|---------------|
| data/ne2025/device1/1201_LeafPhotoA_2025-09-08_10_44_12.jpg | 0.0291 | -0.2576 | ... | 0.1234 |
| ... | ... | ... | ... | ... |

- **Column 1**: `image_path` - Path to the source image
- **Columns 2-257**: `embedding_0` to `embedding_255` - 256-dimensional embedding vector

## How It Works

1. **Model Loading**: Loads SAM2.1 Hiera Tiny model with config `configs/sam2.1/sam2.1_hiera_t.yaml`
2. **Image Processing**:
   - Loads each `.jpg` image from the dataset
   - Converts to RGB format
   - Preprocesses using SAM2.1's standard preprocessing (typically resizes to 1024x1024)
3. **Embedding Extraction**:
   - Passes image through SAM2.1's image encoder
   - Extracts feature maps from the encoder
   - Applies global average pooling across spatial dimensions (H, W)
   - Results in a single 256-dimensional vector per image
4. **CSV Export**: Saves image paths and embeddings to CSV

## Embeddings Details

- **Dimensions**: 256 (from Hiera Tiny encoder)
- **Pooling**: Global average pooling over spatial dimensions
- **Normalization**: Embeddings are raw encoder outputs (not normalized)
- **Data type**: float32
- **Typical value range**: [-1.5, 1.0]

## GPU Memory Usage

- **Model loading**: ~0.28 GB
- **Peak usage**: ~0.5-1.0 GB (includes processing overhead)
- **Recommended**: 4GB+ VRAM

The script automatically clears CUDA cache every 100 images to prevent memory issues.

## Troubleshooting

### CUDA Out of Memory
If you encounter CUDA OOM errors:
1. The script will automatically fall back to CPU
2. Processing will be slower but will complete successfully

### No images found
If the script reports no images:
- Check that images exist in `data/ne2025/device*/` directories
- Verify the IMAGE_PATTERN in the script matches your directory structure
- The script automatically checks both `data/ne2020/` and `data/ne2025/`

### Model loading errors
If model fails to load:
- Verify checkpoint exists: `src/sam2/models/sam2.1_hiera_tiny.pt`
- Check SAM2.1 is installed: `python -c "import sam2; print(sam2.__file__)"`
- Ensure config path is correct: `configs/sam2.1/sam2.1_hiera_t.yaml`

### Corrupted images
The script handles corrupted images gracefully:
- Prints warning for each failed image
- Continues processing remaining images
- Reports failed images in summary statistics

## Performance Tips

1. **GPU acceleration**: Always use CUDA if available (~5x faster than CPU)
2. **Batch processing**: Currently processes 1 image at a time (SAM2.1 design)
3. **Memory management**: Cache is cleared every 100 images automatically
4. **Progress tracking**: tqdm shows real-time progress and estimated completion time

## Example Usage

### Extract embeddings and check results
```bash
# Run extraction
python src/sam2/extract_embeddings.py

# Check CSV size
wc -l src/sam2/sam2_image_embeddings.csv

# View first few rows
head -5 src/sam2/sam2_image_embeddings.csv

# Check embedding statistics
python -c "
import pandas as pd
df = pd.read_csv('src/sam2/sam2_image_embeddings.csv')
print(f'Total images: {len(df)}')
print(f'Embedding dimensions: {len(df.columns) - 1}')
print(f'Mean embedding value: {df.iloc[:, 1:].mean().mean():.4f}')
"
```

### Load embeddings for downstream tasks
```python
import pandas as pd
import numpy as np

# Load embeddings
df = pd.read_csv('src/sam2/sam2_image_embeddings.csv')

# Extract paths and embeddings
image_paths = df['image_path'].values
embeddings = df.iloc[:, 1:].values  # Shape: (N, 256)

print(f"Loaded {len(embeddings)} embeddings")
print(f"Embedding shape: {embeddings.shape}")

# Example: Compute similarity between first two images
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(embeddings[0:1], embeddings[1:2])
print(f"Similarity: {sim[0, 0]:.4f}")
```

## Citation

If you use SAM2.1 embeddings in your research, please cite:

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review error messages in terminal output
3. Verify test script works: `python src/sam2/test_extract_embeddings.py`
