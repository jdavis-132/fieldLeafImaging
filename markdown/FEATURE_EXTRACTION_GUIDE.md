# Complete Feature Extraction Guide

## Overview

This project now includes **two powerful feature extraction pipelines** for extracting visual representations from images:

1. **DINOv2** (Self-supervised Vision Transformer) - General-purpose features
2. **SAM2.1** (Segment Anything Model) - Segmentation-focused features

Both models use state-of-the-art deep learning architectures and provide complementary feature representations suitable for different downstream tasks.

---

## Quick Start

### 1. DINOv2 Feature Extraction (Recommended for most tasks)

```bash
# Test (10 images, ~2 seconds)
python test_extract_dinov2_features.py

# Full extraction (6,147 images, ~10-15 minutes)
python extract_dinov2_features.py
```

**Output**: `dinov2_features.csv` with 384-dimensional features

### 2. SAM2.1 Embedding Extraction

```bash
# Test (5 images, ~2 seconds)
python src/sam2/test_extract_embeddings.py

# Full extraction (6,147 images, ~20-25 minutes)
python src/sam2/extract_embeddings.py
```

**Output**: `src/sam2/sam2_image_embeddings.csv` with 256-dimensional features

### 3. Analyze and Compare

```bash
# Analyze DINOv2 features
python example_dinov2_usage.py

# Compare both models
python compare_embeddings.py
```

---

## Model Comparison

| Aspect | DINOv2 ViT-S/14 | SAM2.1 Tiny |
|--------|-----------------|-------------|
| **Architecture** | Vision Transformer Small | Hiera Tiny |
| **Feature Dimension** | 384 | 256 |
| **Training Objective** | Self-supervised (image-level) | Segmentation (pixel-level) |
| **Training Data** | 142M images | Segmentation datasets |
| **Processing Speed** | 10-15 img/s | 4-6 img/s |
| **Model Size** | 84MB | 149MB |
| **GPU Memory** | ~2-3GB | ~0.5-1GB |
| **Best For** | Classification, retrieval, clustering | Segmentation, spatial tasks |

---

## When to Use Each Model

### Use DINOv2 for:

✅ **Image classification**
- Leaf disease classification
- Plant variety identification
- Quality assessment

✅ **Image retrieval/similarity search**
- Find similar leaf patterns
- Duplicate detection
- Visual search

✅ **Clustering and grouping**
- Organize images by visual similarity
- Discover natural groupings
- Unsupervised analysis

✅ **Transfer learning**
- Fine-tuning for specific tasks
- Few-shot learning
- Domain adaptation

✅ **General-purpose features**
- Baseline for any vision task
- Exploratory data analysis
- Feature engineering

### Use SAM2.1 for:

✅ **Segmentation tasks**
- Leaf boundary detection
- Disease spot segmentation
- Background removal

✅ **Object detection**
- Multi-leaf scenes
- Plant part identification
- Spatial relationships

✅ **Region-based analysis**
- Local feature extraction
- Part-based models
- Spatial pooling strategies

✅ **Complementary to DINOv2**
- Ensemble models
- Multi-view learning
- Fusion strategies

---

## File Structure

```
fieldLeafImaging/
├── extract_dinov2_features.py          # DINOv2 extraction script
├── test_extract_dinov2_features.py     # DINOv2 test script
├── dinov2_features.csv                 # DINOv2 output (after running)
├── example_dinov2_usage.py             # Usage examples
├── compare_embeddings.py               # Model comparison
├── requirements_dinov2.txt             # Python dependencies
├── README_DINOV2.md                    # DINOv2 documentation
│
├── src/sam2/
│   ├── extract_embeddings.py           # SAM2 extraction script
│   ├── test_extract_embeddings.py      # SAM2 test script
│   ├── sam2_image_embeddings.csv       # SAM2 output (after running)
│   └── README_EMBEDDINGS.md            # SAM2 documentation
│
├── data/ne2025/device*/                # Input images (~6,147 images)
│   ├── *.jpg
│   └── ...
│
└── FEATURE_EXTRACTION_GUIDE.md         # This file
```

---

## Installation

### Prerequisites

```bash
# Minimum requirements
pip install torch torchvision numpy pandas pillow tqdm

# Full installation (includes analysis tools)
pip install -r requirements_dinov2.txt
```

### GPU Setup

**Check CUDA availability:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected output:**
```
CUDA available: True
```

---

## Detailed Usage

### DINOv2 Extraction

```bash
# Basic usage
python extract_dinov2_features.py

# With custom configuration
python extract_dinov2_features.py  # Edit script to change:
# - MODEL_NAME: 'dinov2_vits14_reg' (default)
# - BATCH_SIZE: 32 (adjust for your GPU)
# - IMAGE_PATTERN: 'data/ne2025/device*'
```

**Expected output:**
```
✅ Model loaded successfully!
🚀 GPU: NVIDIA GeForce RTX 3060
📐 Feature dimension: 384
🔍 Extracting features from 6147 images...
Processing speed: 12.5 images/second
✅ Feature extraction complete!
```

### SAM2 Extraction

```bash
# Basic usage
python src/sam2/extract_embeddings.py

# Output location
# Creates: src/sam2/sam2_image_embeddings.csv
```

**Expected output:**
```
✅ Model loaded successfully on CUDA!
🚀 GPU: NVIDIA GeForce RTX 3060
🔍 Extracting embeddings from 6147 images...
Processing speed: 4.6 images/second
✅ Embedding extraction complete!
```

---

## Working with Features

### Load Features in Python

```python
import pandas as pd
import numpy as np

# Load DINOv2 features
df = pd.read_csv('dinov2_features.csv')
image_paths = df['image_path'].values
features = df.iloc[:, 1:].values  # Shape: (N, 384)

print(f"Loaded {len(features)} features")
print(f"Feature shape: {features.shape}")
```

### Compute Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

# Similarity between two images
sim = cosine_similarity(features[0:1], features[1:2])
print(f"Similarity: {sim[0, 0]:.4f}")

# Find most similar images to query
query_idx = 0
similarities = cosine_similarity(features[query_idx:query_idx+1], features)[0]
top_5 = np.argsort(similarities)[::-1][1:6]

print("Most similar images:")
for idx in top_5:
    print(f"  {image_paths[idx]}: {similarities[idx]:.4f}")
```

### Clustering

```python
from sklearn.cluster import KMeans

# K-means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(features)

# Save cluster assignments
pd.DataFrame({
    'image_path': image_paths,
    'cluster': clusters
}).to_csv('image_clusters.csv', index=False)
```

### Dimensionality Reduction

```python
from umap import UMAP
import matplotlib.pyplot as plt

# Reduce to 2D
reducer = UMAP(n_components=2, random_state=42)
features_2d = reducer.fit_transform(features)

# Visualize
plt.figure(figsize=(10, 8))
plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.5, s=1)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.savefig('feature_space.png')
```

### Classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Assuming you have labels
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

---

## Performance Optimization

### GPU Memory Management

**If you get OOM (Out of Memory) errors:**

```python
# Reduce batch size in extract_dinov2_features.py
BATCH_SIZE = 16  # or 8

# For SAM2, it processes 1 image at a time (no batch adjustment needed)
```

**Recommended batch sizes by GPU:**
- 4GB VRAM: `BATCH_SIZE = 8`
- 6GB VRAM: `BATCH_SIZE = 16`
- 8GB VRAM: `BATCH_SIZE = 24`
- 12GB VRAM: `BATCH_SIZE = 32-48`

### Speed Optimization

1. **Use GPU** (10-15x faster than CPU)
2. **Increase batch size** (if memory allows)
3. **Use DINOv2 ViT-S** (fastest model)

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size (see above)

#### 2. Model Download Fails
```
ConnectionError or download timeout
```
**Solution**: Check internet, try again (downloads resume automatically)

#### 3. No images found
```
❌ No images found!
```
**Solution**: Check `IMAGE_PATTERN` in script matches your directory structure

#### 4. Slow processing on CPU
```
Processing speed: 1.2 images/second
```
**Solution**: Install CUDA-enabled PyTorch for GPU acceleration

---

## Best Practices

### 1. Always test first
```bash
# Test with small sample before full extraction
python test_extract_dinov2_features.py  # 10 images
python src/sam2/test_extract_embeddings.py  # 5 images
```

### 2. Monitor GPU usage
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### 3. Save intermediate results
```python
# For very large datasets, save in chunks
if i % 1000 == 0:
    np.save(f'features_chunk_{i}.npy', features_array)
```

### 4. Normalize features if needed
```python
from sklearn.preprocessing import normalize

# L2 normalization for cosine similarity
features_normalized = normalize(features, norm='l2')
```

---

## Citation

### DINOv2
```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

### SAM2
```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

---

## Additional Resources

### Documentation
- [DINOv2 README](README_DINOV2.md) - Detailed DINOv2 documentation
- [SAM2 README](src/sam2/README_EMBEDDINGS.md) - Detailed SAM2 documentation

### External Links
- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [DINOv2 GitHub](https://github.com/facebookresearch/dinov2)
- [SAM2 Paper](https://arxiv.org/abs/2408.00714)
- [SAM2 GitHub](https://github.com/facebookresearch/segment-anything-2)

### Example Notebooks
- `example_dinov2_usage.py` - Clustering, similarity, visualization
- `compare_embeddings.py` - Compare DINOv2 vs SAM2

---

## Summary

You now have two powerful feature extraction pipelines:

1. **DINOv2** - Best for classification, retrieval, clustering
   - 384 dimensions
   - 10-15 images/second
   - General-purpose features

2. **SAM2** - Best for segmentation, spatial tasks
   - 256 dimensions
   - 4-6 images/second
   - Segmentation-focused features

**Recommendation**: Start with DINOv2 for most tasks. Use SAM2 if you specifically need segmentation capabilities or want to ensemble both models.

---

## Support

For questions or issues:
1. Run test scripts first
2. Check GPU with `nvidia-smi`
3. Review error messages
4. Consult model-specific READMEs

**Happy feature extraction! 🚀**
