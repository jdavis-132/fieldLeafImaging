# Disease-Aware Autoencoder for Maize Leaf Clustering

A U-Net style autoencoder with attention mechanism for learning disease-aware representations of maize leaves. The model uses LAB color space and disease-weighted loss to prioritize diseased regions during training.

## Overview

This implementation provides:

- **U-Net Architecture**: Encoder-decoder with skip connections for better reconstruction
- **Attention Mechanism**: Spatial and channel attention in the bottleneck
- **LAB Color Space**: Better representation of disease symptoms than RGB
- **Disease-Weighted Loss**: Automatically prioritizes diseased regions based on color deviation from healthy tissue
- **UMAP Visualization**: 2D projection of latent space for clustering analysis

## Architecture

### Input Processing
1. Load RGB image and corresponding masks
2. Crop to leaf bounding box with padding
3. Resize to 224×224
4. Convert RGB → LAB color space
5. Normalize using masked pixel statistics
6. Set background pixels to 0

### Model
- **Input**: LAB image (3 channels) + binary mask (1 channel) = 4 channels
- **Encoder**: 4 downsampling blocks (64 → 128 → 256 → 512 channels)
- **Bottleneck**: 1024 channels with spatial + channel attention
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: Reconstructed LAB image (3 channels)
- **Embedding**: 256-dimensional latent representation

### Disease-Weighted Loss

The loss function automatically weights pixels based on disease severity:

**In LAB space:**
- Healthy tissue: Green (a* ≈ -30), moderate yellow (b* ≈ 30)
- Diseased tissue: Shifts toward red (a* > -30) and yellow (b* > 30)

**Weight calculation:**
```
weight = 1.0 + disease_score * (strength - 1.0)
```

where disease_score is based on color deviation from healthy tissue.

**Final loss:**
```
loss = weighted_MSE + λ * L1(embeddings)
```

## Installation

1. Install dependencies:
```bash
cd disease_autoencoder
pip install -r requirements.txt
```

2. For GPU support (optional but recommended):
   - **CUDA**: Install PyTorch with CUDA from [pytorch.org](https://pytorch.org)
   - **Mac (MPS)**: Already supported in PyTorch ≥2.0

## Usage

### 1. Train the Model

```bash
python -m disease_autoencoder.train
```

This will:
- Use the existing `prepare_splits` function to split data by genotype (prevents data leakage)
- Compute LAB normalization statistics from training set
- Train the model with disease-weighted loss
- Save checkpoints to `disease_autoencoder/models/`
- Log training history to `disease_autoencoder/logs/`

**Training features:**
- Early stopping (patience: 15 epochs)
- Learning rate scheduling (ReduceLROnPlateau)
- Checkpoint saving (best + every 10 epochs)
- Progress bars with loss components

**Expected training time:**
- CPU: ~10-20 hours
- GPU: ~2-4 hours
- MPS (Mac): ~4-8 hours

### 2. Extract Embeddings and Visualize

```bash
python -m disease_autoencoder.evaluate
```

This will:
- Load the best trained model
- Extract latent embeddings for all images
- Create UMAP visualization of latent space
- Generate reconstruction examples
- Save embeddings for clustering

**Output files:**
- `disease_autoencoder/embeddings/latent_embeddings.npy` - Embeddings (N × 256)
- `disease_autoencoder/embeddings/embeddings_metadata.json` - Image metadata
- `disease_autoencoder/embeddings/umap_coordinates.npy` - UMAP 2D projection
- `disease_autoencoder/visualizations/umap_latent_space.png` - UMAP plot
- `disease_autoencoder/visualizations/reconstructions.png` - Reconstruction examples

### 3. Use Embeddings for Clustering

```python
import numpy as np
import json
from sklearn.cluster import KMeans, DBSCAN

# Load embeddings
embeddings = np.load('disease_autoencoder/embeddings/latent_embeddings.npy')
with open('disease_autoencoder/embeddings/embeddings_metadata.json', 'r') as f:
    metadata = json.load(f)

# Cluster using K-means
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Or use DBSCAN for density-based clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(embeddings)

# Analyze clusters by genotype
for i in range(clusters.max() + 1):
    cluster_genotypes = [metadata[j]['genotype']
                         for j in np.where(clusters == i)[0]]
    print(f"Cluster {i}: {len(cluster_genotypes)} samples")
```

## Configuration

Edit `disease_autoencoder/config.py` to customize:

### Data Parameters
- `image_size`: Input size (default: 224)
- `crop_padding`: Padding around leaf bbox (default: 10)

### Model Architecture
- `embedding_dim`: Latent dimension (default: 256)
- `unet_features`: Feature channels [64, 128, 256, 512]
- `use_attention`: Use attention mechanism (default: True)

### Training Hyperparameters
- `batch_size`: Batch size (default: 16)
- `num_epochs`: Maximum epochs (default: 100)
- `learning_rate`: Initial LR (default: 1e-4)
- `patience`: Early stopping patience (default: 15)

### Loss Function
- `disease_weight_strength`: Max weight for diseased pixels (default: 2.0)
- `l1_regularization`: L1 weight on embeddings (default: 1e-5)

### UMAP Parameters
- `umap_n_neighbors`: Neighborhood size (default: 15)
- `umap_min_dist`: Minimum distance (default: 0.1)
- `umap_metric`: Distance metric (default: 'cosine')

## Project Structure

```
disease_autoencoder/
├── __init__.py              # Package initialization
├── config.py                # Configuration management
├── dataset.py               # Data loading and preprocessing
├── model.py                 # U-Net autoencoder architecture
├── loss.py                  # Disease-weighted loss function
├── train.py                 # Training script
├── evaluate.py              # Evaluation and visualization
├── requirements.txt         # Dependencies
└── README.md               # This file

# Generated during training/evaluation:
disease_autoencoder/
├── models/                  # Model checkpoints
│   ├── checkpoint_best.pth
│   ├── checkpoint_latest.pth
│   └── checkpoint_epoch_*.pth
├── embeddings/              # Extracted embeddings
│   ├── latent_embeddings.npy
│   ├── embeddings_metadata.json
│   └── umap_coordinates.npy
├── visualizations/          # Plots and figures
│   ├── umap_latent_space.png
│   └── reconstructions.png
└── logs/                    # Training logs and statistics
    ├── image_splits.json
    ├── lab_statistics.json
    └── training_history.json
```

## Key Features

### 1. Genotype-Based Data Splitting
Uses the existing `prepare_splits` function to ensure all images from the same genotype are in the same split, preventing data leakage.

### 2. LAB Color Space Preprocessing
- More perceptually uniform than RGB
- Better separation of luminance (L) and color (a, b)
- Disease symptoms are more distinct in a* and b* channels

### 3. Disease-Aware Loss Weighting
Automatically identifies and prioritizes diseased regions based on:
- a* channel: Shift from green (healthy) to red (diseased)
- b* channel: Excessive yellow (disease symptom)

No manual annotation needed!

### 4. Safe Augmentation for Disease Analysis
- 90° rotations (preserves disease patterns)
- Horizontal/vertical flips
- Minimal brightness adjustment (±5% L channel only)
- **NO color jittering** (preserves disease color signatures)

### 5. Attention Mechanism
- **Spatial Attention**: Focus on important regions
- **Channel Attention**: Weight important feature channels
- Applied in bottleneck for better feature learning

## Troubleshooting

### Memory Issues
- Reduce `batch_size` in config
- Reduce `image_size` (try 192 or 160)
- Reduce `num_workers` for DataLoader

### Training Too Slow
- Use GPU if available
- Reduce `num_epochs` (50 may be sufficient)
- Increase `batch_size` if memory allows

### Poor Reconstruction Quality
- Increase `num_epochs`
- Adjust `disease_weight_strength` (try 1.5 or 3.0)
- Check LAB normalization statistics in logs

### Clustering Not Working Well
- Try different `embedding_dim` (128, 512)
- Adjust UMAP parameters
- Use different clustering algorithm (HDBSCAN, Gaussian Mixture)

## Citation

If you use this code, please cite:

```bibtex
@software{disease_autoencoder,
  title = {Disease-Aware Autoencoder for Maize Leaf Clustering},
  year = {2025},
  note = {U-Net architecture with attention mechanism and disease-weighted loss}
}
```

## License

This code is provided for research purposes. See the main project repository for license information.
