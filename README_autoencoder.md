# Autoencoder Pipeline for Masked Leaf Images

Complete PyTorch-based pipeline for training an autoencoder on masked, normalized leaf images with genotype-based train/test/validation splitting.

## Overview

This pipeline:
1. **Loads normalized images** from ColorChecker normalization
2. **Combines two segmentation masks** (logical OR) for each image
3. **Applies masking** (sets non-mask pixels to black)
4. **Splits by genotype** to prevent data leakage
5. **Trains a convolutional autoencoder** using PyTorch
6. **Extracts embeddings** from the bottleneck layer
7. **Visualizes results** including UMAP/PCA projections

## Requirements

Install additional packages beyond base requirements:

```bash
pip install umap-learn scikit-learn seaborn
```

All other dependencies (PyTorch, OpenCV, etc.) are already in `requirements.txt`.

## Quick Start

### Option 1: Run Complete Pipeline

```bash
python run_pipeline.py
```

This runs all steps in sequence:
- Prepare genotype-based splits
- Visualize dataset
- Train autoencoder
- Extract embeddings
- Generate visualizations

### Option 2: Run Steps Individually

```bash
# Step 1: Prepare splits (CRITICAL: genotype-based to prevent leakage)
python -m autoencoder.prepare_splits

# Step 2: Visualize dataset
python -m autoencoder.visualize

# Step 3: Train model
python -m autoencoder.train

# Step 4: Extract embeddings
python -m autoencoder.extract_embeddings
```

## Data Structure

### Input Requirements

For each image, you need:
1. **Normalized image**: `output/device{X}/{plot}_LeafPhoto{A|B}_timestamp.jpg`
2. **Mask 1**: `output/stripSegmentation/ne2025/device{X}_{plot}_..../strip_0000_mask.png`
3. **Mask 2**: `output/stripSegmentation/ne2025/device{X}_{plot}_..../strip_0001_mask.png`
4. **Genotype metadata**: `data/ne2025/SbDiv_ne2025_fieldindex.csv` (with 'plot' and 'genotype' columns)

### Current Dataset

- **Total usable images**: 5,950
- **Unique genotypes**: 961
- **Devices**: 8 (device1-device8)
- **Image dimensions**: 4080 × 3060 (resized to 512 × 384 for training)

## Key Features

### 1. Genotype-Based Splitting

**CRITICAL**: All images with the same genotype are kept in the same split (train, val, or test).

- **Why?** Prevents data leakage where the model could learn genotype-specific features and appear to generalize when it's actually memorizing genotypes seen in training.
- **Strategy**:
  - Genotypes with ≥3 images: randomly split 70/15/15 (train/val/test)
  - Genotypes with <3 images: assign to training set
  - Random seed: 42 (for reproducibility)

### 2. Dual Mask Combination

Each image has **two** segmentation masks that are combined:

```python
combined_mask = logical_or(mask_0, mask_1)
masked_image = original_image * combined_mask  # Pixels outside mask → [0,0,0]
```

This captures both leaf strips in the image.

### 3. Autoencoder Architecture

```
Encoder:
  Conv2d(3→64) + BN + ReLU → /2
  Conv2d(64→128) + BN + ReLU → /2
  Conv2d(128→256) + BN + ReLU → /2
  Conv2d(256→512) + BN + ReLU → /2
  FC(512×24×32 → embedding_dim)

Bottleneck: 256-dimensional embedding (configurable)

Decoder:
  FC(embedding_dim → 512×24×32)
  ConvTranspose2d(512→256) + BN + ReLU → ×2
  ConvTranspose2d(256→128) + BN + ReLU → ×2
  ConvTranspose2d(128→64) + BN + ReLU → ×2
  ConvTranspose2d(64→3) + Sigmoid → ×2
```

**Total parameters**: ~64M trainable parameters

### 4. Training Features

- **Loss**: MSE (Mean Squared Error) reconstruction loss
- **Optimizer**: AdamW with weight decay
- **LR Scheduler**: ReduceLROnPlateau (reduces LR when val loss plateaus)
- **Early Stopping**: Stops if val loss doesn't improve for 15 epochs
- **Checkpointing**: Saves best model + per-epoch checkpoints
- **Visualization**: Saves reconstructions every 5 epochs

## Configuration

Edit `autoencoder/config.py` to adjust hyperparameters:

```python
# Model
embedding_dim = 256  # Bottleneck dimension

# Training
batch_size = 16
num_epochs = 100
learning_rate = 1e-4
weight_decay = 1e-5

# Data splits
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Image processing
image_size = (512, 384)  # (width, height)
```

## Outputs

### Directory Structure

```
models/
├── best_model.pth              # Best model (lowest val loss)
└── checkpoint_epoch_*.pth      # Per-epoch checkpoints

embeddings/
├── embeddings.csv              # Embeddings + metadata (CSV)
├── embeddings.npz              # Numpy arrays (train/val/test)
├── embeddings_metadata.json    # Metadata only
└── embeddings_summary.txt      # Statistics

visualizations/
├── SUMMARY_REPORT.txt          # Comprehensive summary
├── training_curves.png         # Train/val loss over epochs
├── reconstructions_*.png       # Original vs reconstructed images
├── masking_process_examples.png  # Shows mask combination
├── split_distribution.png      # Images and genotypes per split
├── genotype_distribution.png   # Top genotypes per split
├── embeddings_umap.png         # UMAP projection of embeddings
└── embeddings_pca.png          # PCA projection of embeddings

logs/
├── image_splits.json           # Image assignments to splits
├── genotype_splits.json        # Genotype assignments to splits
├── training_log.csv            # Per-epoch metrics
├── training_config.json        # Saved configuration
└── dataset_summary.txt         # Dataset statistics
```

### Embeddings CSV Format

```csv
filename,plot,genotype,device,split,emb_0,emb_1,...,emb_255
1201_LeafPhotoA_...,1201,PI 533750,1,train,-0.234,0.567,...,0.123
...
```

## Usage Examples

### Training with Custom Config

```python
from autoencoder.config import Config
from autoencoder.train import train

config = Config()
config.embedding_dim = 512  # Larger embedding
config.batch_size = 32      # Larger batch
config.num_epochs = 50      # Fewer epochs

train(config)
```

### Loading Trained Model for Inference

```python
import torch
from autoencoder.config import Config
from autoencoder.model import create_model
from autoencoder.utils import load_checkpoint

config = Config()
model = create_model(config)
load_checkpoint('models/best_model.pth', model, device=config.device)

# Extract embedding for a single image
model.eval()
with torch.no_grad():
    embedding = model.encode(image_tensor)
```

### Analyzing Embeddings

```python
import pandas as pd
import numpy as np

# Load embeddings
df = pd.read_csv('embeddings/embeddings.csv')

# Get embeddings for a specific genotype
genotype_embeddings = df[df['genotype'] == 'Check'][[f'emb_{i}' for i in range(256)]]

# Calculate average embedding per genotype
genotype_means = df.groupby('genotype')[[f'emb_{i}' for i in range(256)]].mean()
```

## Monitoring Training

Track training progress:

```bash
# View training log
cat logs/training_log.csv

# Plot in real-time (requires installation)
# pip install pandas matplotlib
python << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('logs/training_log.csv')
plt.plot(df['epoch'], df['train_loss'], label='Train')
plt.plot(df['epoch'], df['val_loss'], label='Val')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
EOF
```

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size in `config.py`:
```python
config.batch_size = 8  # or 4
```

### Training Too Slow

- Check GPU is being used: `nvidia-smi`
- Increase `num_workers` in config
- Reduce image size: `config.image_size = (256, 192)`

### Poor Reconstructions

- Train longer (increase `num_epochs`)
- Reduce embedding dimension to force better compression
- Check that masks are being applied correctly (see visualizations)

### Data Leakage Concerns

- Verify genotype splits: `logs/genotype_splits.json`
- Ensure no genotype appears in multiple splits
- Check summary report: `visualizations/SUMMARY_REPORT.txt`

## Advanced Usage

### Custom Data Augmentation

Modify `dataset.py` to add augmentations:

```python
def get_default_transform(self):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),  # Add this
        transforms.Normalize(mean=self.config.mean, std=self.config.std)
    ])
```

### Different Loss Function

Try perceptual loss or other variants in `train.py`:

```python
# Instead of MSELoss
criterion = nn.L1Loss()  # L1/MAE loss
```

### Embedding Dimensionality Analysis

```python
from sklearn.decomposition import PCA

# Load embeddings
data = np.load('embeddings/embeddings.npz')
embeddings = data['embeddings']

# Explained variance
pca = PCA()
pca.fit(embeddings)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()
```

## Citation

If you use this pipeline in your research, please cite:
- PyTorch: Paszke et al. (2019)
- UMAP: McInnes et al. (2018)

## Next Steps

1. **Review outputs**: Check `visualizations/SUMMARY_REPORT.txt`
2. **Analyze embeddings**: Use for:
   - Genotype classification
   - Phenotype prediction
   - Clustering analysis
   - Outlier detection
3. **Iterate**: Adjust hyperparameters and retrain
4. **Deploy**: Use trained encoder for feature extraction on new images
