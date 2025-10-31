# Disease-Aware Autoencoder - Quick Start Guide

## Setup

### 1. Install Dependencies

First, install the required packages:

```bash
cd disease_autoencoder
pip install -r requirements.txt
```

**For GPU support:**
- **NVIDIA GPU**: Install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/)
- **Mac (Apple Silicon)**: PyTorch 2.0+ includes MPS support automatically
- **CPU only**: The default installation works

### 2. Verify Setup

Run the test script to verify everything is working:

```bash
python -m disease_autoencoder.test_setup
```

If all tests pass, you're ready to train!

## Training

### Basic Training

Run the training script:

```bash
python -m disease_autoencoder.train
```

**What happens:**
1. ✓ Uses existing `prepare_splits` to split data by genotype (prevents data leakage)
2. ✓ Computes LAB normalization statistics from training set
3. ✓ Trains U-Net autoencoder with disease-weighted loss
4. ✓ Saves checkpoints and training logs
5. ✓ Automatic early stopping when validation loss plateaus

**Expected runtime:**
- GPU: 2-4 hours
- CPU: 10-20 hours
- Mac MPS: 4-8 hours

**Output files:**
- `disease_autoencoder/models/checkpoint_best.pth` - Best model
- `disease_autoencoder/logs/training_history.json` - Loss curves
- `disease_autoencoder/logs/lab_statistics.json` - Normalization stats

### Monitor Training

The training script shows:
- Progress bar with current loss
- Epoch summary with train/val losses
- Loss components (reconstruction, L1, disease weights)
- Learning rate updates
- Early stopping status

Example output:
```
Epoch 25/100: 100%|████████| 234/234 [02:15<00:00]
  Train Loss: 0.003456 (rec: 0.003200, l1: 0.000256, weight: 1.45)
  Val Loss:   0.003892 (rec: 0.003650, l1: 0.000242, weight: 1.48)
  Saved best model (val_loss: 0.003892)
```

## Evaluation

After training completes, extract embeddings and create visualizations:

```bash
python -m disease_autoencoder.evaluate
```

**What happens:**
1. ✓ Loads best trained model
2. ✓ Extracts 256-dim embeddings for all images
3. ✓ Creates UMAP 2D visualization
4. ✓ Generates reconstruction examples
5. ✓ Saves everything for clustering

**Output files:**
- `disease_autoencoder/embeddings/latent_embeddings.npy` - Shape: (N, 256)
- `disease_autoencoder/embeddings/embeddings_metadata.json` - Image info
- `disease_autoencoder/embeddings/umap_coordinates.npy` - 2D UMAP coords
- `disease_autoencoder/visualizations/umap_latent_space.png` - UMAP plot
- `disease_autoencoder/visualizations/reconstructions.png` - Examples

## Clustering Analysis

Use the extracted embeddings for clustering:

```python
import numpy as np
import json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load embeddings
embeddings = np.load('disease_autoencoder/embeddings/latent_embeddings.npy')
umap_coords = np.load('disease_autoencoder/embeddings/umap_coordinates.npy')

with open('disease_autoencoder/embeddings/embeddings_metadata.json', 'r') as f:
    metadata = json.load(f)

# Cluster in latent space
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Visualize clusters in UMAP space
plt.figure(figsize=(10, 8))
scatter = plt.scatter(umap_coords[:, 0], umap_coords[:, 1],
                     c=clusters, cmap='tab10', alpha=0.6, s=20)
plt.colorbar(scatter, label='Cluster')
plt.title('Clusters in UMAP Space')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.savefig('disease_autoencoder/visualizations/clusters.png', dpi=300)
plt.show()

# Analyze cluster composition
for i in range(n_clusters):
    cluster_mask = clusters == i
    cluster_genotypes = [metadata[j]['genotype']
                        for j in np.where(cluster_mask)[0]]

    print(f"\nCluster {i}:")
    print(f"  Size: {sum(cluster_mask)} images")
    print(f"  Unique genotypes: {len(set(cluster_genotypes))}")

    # Top genotypes in this cluster
    from collections import Counter
    top_genotypes = Counter(cluster_genotypes).most_common(5)
    print(f"  Top genotypes:")
    for genotype, count in top_genotypes:
        print(f"    - {genotype}: {count} images")
```

## Troubleshooting

### Out of Memory (OOM)

If you get memory errors during training:

```python
# Edit disease_autoencoder/config.py
self.batch_size = 8        # Reduce from 16
self.image_size = 192      # Reduce from 224
self.num_workers = 2       # Reduce from 4
```

### Training Too Slow

```python
# Edit disease_autoencoder/config.py
self.num_epochs = 50       # Reduce from 100
self.patience = 10         # Reduce from 15
```

### Poor Reconstruction Quality

1. Check training logs: `cat disease_autoencoder/logs/training_history.json`
2. Visualize loss curves
3. Adjust disease weighting:

```python
# Edit disease_autoencoder/config.py
self.disease_weight_strength = 3.0  # Increase from 2.0
```

### Model Not Converging

- Increase learning rate: `self.learning_rate = 5e-4`
- Disable L1 regularization: `self.l1_regularization = 0`
- Train longer: `self.num_epochs = 150`

## What Makes This Different?

### 1. Disease-Aware Loss
Unlike standard autoencoders, this implementation:
- Automatically detects diseased regions based on LAB color
- Prioritizes reconstruction of disease symptoms
- No manual annotation needed!

### 2. LAB Color Space
- Better for disease: Separates luminance from color
- Healthy tissue: Green (a* < 0)
- Diseased tissue: Red/yellow shift (a* → 0, b* ↑)

### 3. U-Net Architecture
- Skip connections preserve spatial details
- Attention mechanism focuses on important features
- Better reconstruction than simple encoder-decoder

### 4. Genotype-Based Splitting
- Uses existing `prepare_splits` function
- All images from same genotype stay together
- Prevents data leakage between train/val/test

## Next Steps

After getting embeddings:

1. **Cluster Analysis**: Try different clustering algorithms (K-means, DBSCAN, HDBSCAN)
2. **Disease Severity**: Analyze cluster properties (average color deviation)
3. **Genotype Patterns**: Check if clusters align with genotypes
4. **Feature Importance**: Use SHAP or attention maps to understand what the model learned

## Advanced Usage

### Custom Loss Function

Disable disease weighting to compare:

```python
# In train.py, modify:
loss_fn = create_loss_function(config, use_disease_weighting=False)
```

### Different Architecture

```python
# Edit disease_autoencoder/config.py
self.unet_features = [32, 64, 128, 256]  # Smaller model
self.embedding_dim = 128                  # Smaller bottleneck
self.use_attention = False                # Disable attention
```

### Resume Training

```python
# In train.py, add checkpoint loading:
checkpoint = torch.load('disease_autoencoder/models/checkpoint_latest.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## File Structure

```
disease_autoencoder/
├── config.py           # Configuration (edit this!)
├── dataset.py          # Data loading (LAB preprocessing)
├── model.py            # U-Net architecture
├── loss.py            # Disease-weighted loss
├── train.py           # Training script (run this)
├── evaluate.py        # Evaluation script (run this)
├── test_setup.py      # Setup verification
├── requirements.txt   # Dependencies
├── README.md          # Full documentation
└── QUICKSTART.md      # This file

# Generated files:
disease_autoencoder/
├── models/            # Checkpoints
├── embeddings/        # Latent codes
├── visualizations/    # Plots
└── logs/             # Training logs
```

## Getting Help

1. Check README.md for detailed documentation
2. Run test_setup.py to diagnose issues
3. Review training logs in `logs/training_history.json`
4. Check UMAP visualization to verify embeddings make sense

## Common Questions

**Q: How long should I train?**
A: Early stopping will handle this automatically. Usually 30-60 epochs is enough.

**Q: How do I know if the model is working?**
A: Check the UMAP visualization - you should see some clustering by genotype or disease severity.

**Q: What's a good reconstruction loss?**
A: After training, values around 0.003-0.01 are typical. Lower is better, but don't overfit!

**Q: Should I use disease weighting?**
A: Yes! It helps the model focus on disease symptoms rather than healthy tissue.

**Q: Can I use this for other crops?**
A: Yes! Just adjust the healthy color ranges in `loss.py` for your crop's healthy tissue color.

Happy clustering! 🌽🔬
