# DINOv2 Feature Extraction

## Overview

Scripts to extract image features using the DINOv2 (Self-Distillation with No Labels) vision transformer backbone from Facebook Research. This extracts powerful visual representations suitable for downstream tasks like classification, clustering, retrieval, and analysis.

## Files

- **`extract_dinov2_features.py`**: Main script to extract features from all images
- **`test_extract_dinov2_features.py`**: Test script to verify extraction works (processes 10 images)
- **`dinov2_features.csv`**: Output CSV file with features (created after running)
- **`README_DINOV2.md`**: This documentation file

## Model Details

**DINOv2 ViT-S/14 with Registers (distilled)**
- **Architecture**: Vision Transformer Small (ViT-S/14)
- **Patch size**: 14×14 pixels
- **Training**: Self-supervised learning on 142M images
- **Registers**: Additional learned tokens that improve feature quality
- **Feature dimension**: 384 (CLS token output)
- **Input size**: 224×224 pixels
- **Source**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

## Requirements

### Python Packages
```bash
pip install torch torchvision tqdm pandas numpy pillow
```

### GPU Requirements
- **Recommended**: NVIDIA GPU with CUDA support
- **Minimum VRAM**: 4GB (for batch_size=32)
- **Tested on**: NVIDIA GeForce RTX 3060 (12GB)

### Model Download
The model is automatically downloaded from PyTorch Hub on first run (~84MB).

## Quick Start

### 1. Test the extraction (recommended first)

```bash
python test_extract_dinov2_features.py
```

**What this does:**
- Downloads DINOv2 model (first run only, ~84MB)
- Processes 10 images from your dataset
- Creates `test_dinov2_features.csv`
- Takes ~1-2 seconds after model download

**Expected output:**
```
✅ Model loaded successfully!
🚀 GPU: NVIDIA GeForce RTX 3060
📐 Feature dimension: 384
🔍 Extracting features from 10 images...
✅ Test completed successfully!
```

### 2. Run full extraction

```bash
python extract_dinov2_features.py
```

**What this does:**
- Processes all ~6,147 images in `data/ne2025/device*/`
- Extracts 384-dimensional features per image
- Saves to `dinov2_features.csv`
- Takes ~10-15 minutes on RTX 3060

**Expected processing:**
- Processing speed: ~10-15 images/second on GPU
- Batch size: 32 images (adjustable)
- Memory usage: ~2-3GB GPU memory
- Output size: ~10MB CSV file

## Output Format

CSV structure:

| image_path | feature_0 | feature_1 | ... | feature_383 |
|------------|-----------|-----------|-----|-------------|
| data/ne2025/device1/image1.jpg | -0.8178 | -0.0271 | ... | 0.3456 |
| data/ne2025/device1/image2.jpg | 0.1234 | -0.5678 | ... | -0.2109 |

- **Column 1**: `image_path` - Full path to source image
- **Columns 2-385**: `feature_0` to `feature_383` - 384-dimensional feature vector from CLS token

## How It Works

### 1. Model Loading
```python
# Loads DINOv2 ViT-S/14 with registers from PyTorch Hub
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
```

### 2. Image Preprocessing
```python
# Standard DINOv2 preprocessing pipeline:
transforms.Compose([
    Resize(256),              # Resize shorter edge to 256
    CenterCrop(224),          # Center crop to 224×224
    ToTensor(),               # Convert to tensor [0, 1]
    Normalize(                # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 3. Feature Extraction
- Image passes through Vision Transformer
- Extracts the **CLS token** (classification token) from final layer
- CLS token is a 384-dimensional vector representing the entire image
- This global feature is suitable for:
  - Image classification
  - Similarity search
  - Clustering
  - Retrieval
  - Transfer learning

### 4. Batch Processing
- Images processed in batches of 32 (configurable)
- Efficient GPU utilization
- Automatic error handling for corrupted images
- Progress tracking with tqdm

## Feature Characteristics

### Statistics from Test Run
```
Dimensions: 384
Mean: -0.0071
Std: 1.3157
Min: -4.3170
Max: 4.5360
```

### Properties
- **Normalized**: Features are roughly zero-centered
- **Scale**: Typical values range from -5 to +5
- **Distribution**: Approximately Gaussian
- **No post-processing**: Raw CLS token output (no normalization applied)

## Usage Examples

### Basic Usage
```bash
# Test with 10 images
python test_extract_dinov2_features.py

# Extract all features
python extract_dinov2_features.py
```

### Load Features in Python
```python
import pandas as pd
import numpy as np

# Load features
df = pd.read_csv('dinov2_features.csv')

# Extract paths and features
image_paths = df['image_path'].values
features = df.iloc[:, 1:].values  # Shape: (N, 384)

print(f"Loaded {len(features)} features")
print(f"Feature shape: {features.shape}")
```

### Compute Image Similarity
```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity between first two images
sim = cosine_similarity(features[0:1], features[1:2])
print(f"Similarity: {sim[0, 0]:.4f}")

# Find most similar images to first image
similarities = cosine_similarity(features[0:1], features)[0]
most_similar_idx = np.argsort(similarities)[::-1][1:6]  # Top 5 (excluding self)
print("Most similar images:")
for idx in most_similar_idx:
    print(f"  {image_paths[idx]}: {similarities[idx]:.4f}")
```

### Clustering
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cluster images into 10 groups
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(features)

# Visualize cluster sizes
plt.hist(clusters, bins=10)
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Image Cluster Distribution')
plt.savefig('cluster_distribution.png')
```

### Dimensionality Reduction (UMAP/t-SNE)
```python
from umap import UMAP
import matplotlib.pyplot as plt

# Reduce to 2D for visualization
reducer = UMAP(n_components=2, random_state=42)
features_2d = reducer.fit_transform(features)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.5, s=1)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('DINOv2 Feature Space (UMAP)')
plt.savefig('dinov2_umap.png')
```

## Configuration

### Adjust Batch Size
Edit `extract_dinov2_features.py`:
```python
BATCH_SIZE = 32  # Reduce if you encounter OOM errors
```

**Recommended batch sizes:**
- 4GB VRAM: batch_size=8
- 6GB VRAM: batch_size=16
- 8GB VRAM: batch_size=24
- 12GB VRAM: batch_size=32-48

### Use Different Model Variant
Edit `extract_dinov2_features.py`:
```python
MODEL_NAME = 'dinov2_vits14_reg'  # Current (384 dims)
# MODEL_NAME = 'dinov2_vitb14_reg'  # Base model (768 dims)
# MODEL_NAME = 'dinov2_vitl14_reg'  # Large model (1024 dims)
# MODEL_NAME = 'dinov2_vitg14_reg'  # Giant model (1536 dims)
```

**Model comparison:**

| Model | Params | Feature Dim | Speed | Quality |
|-------|--------|-------------|-------|---------|
| ViT-S/14 | 21M | 384 | Fast | Good |
| ViT-B/14 | 86M | 768 | Medium | Better |
| ViT-L/14 | 300M | 1024 | Slow | Excellent |
| ViT-G/14 | 1.1B | 1536 | Very Slow | Best |

### Change Input Directory
Edit `extract_dinov2_features.py`:
```python
IMAGE_PATTERN = 'data/ne2025/device*'  # Current
# IMAGE_PATTERN = 'path/to/your/images/*'
```

## Performance Optimization

### GPU Memory Tips
1. **Reduce batch size** if you get OOM errors
2. **Use smaller model** (ViT-S instead of ViT-B/L/G)
3. **Process in chunks** if dataset is very large

### Speed Tips
1. **Use GPU** (10-15x faster than CPU)
2. **Increase batch size** (if GPU memory allows)
3. **Use ViT-S/14** for fastest processing

### Storage Tips
- CSV file size: ~10MB for 6,000 images with 384 features
- Consider saving as `.npy` for larger datasets:
  ```python
  np.save('dinov2_features.npy', features_array)
  ```

## Troubleshooting

### CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```python
# Reduce batch size in extract_dinov2_features.py
BATCH_SIZE = 16  # or 8
```

### Model Download Fails
**Error**: `ConnectionError` or download timeout

**Solution**:
1. Check internet connection
2. Try again (resume download)
3. Manual download:
   ```bash
   wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth
   # Place in ~/.cache/torch/hub/checkpoints/
   ```

### Corrupted Images
**Symptom**: Some images fail to load

**Behavior**:
- Script continues processing
- Failed images reported in summary
- Does not stop execution

### CPU Fallback
**Symptom**: "CUDA not available, falling back to CPU"

**Impact**:
- Still works but slower (~10x)
- Processing time: ~1-2 hours for 6,000 images

## Comparison with Other Models

### DINOv2 vs. SAM2.1 Embeddings

| Aspect | DINOv2 ViT-S/14 | SAM2.1 Tiny |
|--------|-----------------|-------------|
| Feature dim | 384 | 256 |
| Training | Self-supervised (142M images) | Segmentation-focused |
| Speed | 10-15 img/s | 4-6 img/s |
| Use case | General features | Segmentation + features |
| Quality | Excellent for classification/retrieval | Good for spatial tasks |

### When to Use DINOv2
- ✅ Image classification
- ✅ Image retrieval/search
- ✅ Clustering and grouping
- ✅ Transfer learning
- ✅ Zero-shot applications
- ✅ General-purpose features

### When to Use SAM2.1
- ✅ Segmentation tasks
- ✅ Object detection
- ✅ Spatial relationships
- ✅ Region-based features

## Citation

If you use DINOv2 features in your research, please cite:

```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

## Support

For issues or questions:

1. **Test first**: Run `python test_extract_dinov2_features.py`
2. **Check requirements**: Ensure all packages installed
3. **Verify GPU**: Check CUDA with `torch.cuda.is_available()`
4. **Review errors**: Read error messages in terminal output

## Additional Resources

- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [GitHub Repository](https://github.com/facebookresearch/dinov2)
- [Interactive Demo](https://dinov2.metademolab.com/)
- [Blog Post](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/)
