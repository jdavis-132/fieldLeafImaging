# Standard Autoencoder (No Disease Weighting)

This is a modified version of the disease-aware autoencoder that removes disease-based pixel weighting from the loss function. Instead, it uses simple Mean Squared Error (MSE) reconstruction loss that treats all pixels equally.

## What Changed from the Original Model

### Key Difference: Loss Function

**Original (Disease-Weighted) Version:**
```python
# loss.py - DiseaseWeightedLoss
def compute_disease_weights(self, lab_target, mask, lab_mean, lab_std):
    """
    Compute per-pixel weights based on LAB color deviation from healthy tissue.
    - Healthy tissue: Green (negative a*), moderate b*
    - Diseased tissue: Red shift (less negative a*), yellow shift (higher b*)
    - Weights range from 1.0 (healthy) to disease_weight_strength (diseased)
    """
    # ... computes weights based on color deviation ...
    weights = 1.0 + disease_score * (disease_weight_strength - 1.0)
    return weights
```

**Modified (No Weighting) Version:**
```python
# loss.py - SimpleMSELoss
def forward(self, predictions, targets, masks, embeddings=None, lab_stats=None):
    """
    Simple MSE loss on ALL pixels - no disease weighting.
    """
    squared_error = (predictions - targets) ** 2
    reconstruction_loss = squared_error.mean()
    return reconstruction_loss
```

### What Stayed the Same

The following components are **identical** to the disease-weighted version:

- ✅ **Model architecture**: U-Net with attention mechanism
- ✅ **Number of parameters**: Exact same network
- ✅ **Data loading**: Same preprocessing and augmentation
- ✅ **Training procedure**: Same optimizer, learning rate, scheduler
- ✅ **Hyperparameters**: Batch size, learning rate, epochs, etc.

### Configuration Changes

| Parameter | Disease-Weighted | No Weighting | Notes |
|-----------|-----------------|--------------|-------|
| `disease_weight_strength` | 2.0 | *(removed)* | Not applicable |
| `loss_type` | Disease-weighted | Simple MSE | Changed |
| Output directory | `disease_autoencoder_cropped/` | `autoencoder_no_weighting/` | Changed |
| All other params | *Same* | *Same* | Identical |

## Why Remove Disease Weighting?

Disease weighting was designed to prioritize reconstruction of diseased regions by giving them higher loss weights. However, this approach has trade-offs:

### Disease-Weighted Model (Original)

**Advantages:**
- Better reconstruction quality for diseased regions
- More sensitive to disease-specific color patterns
- Latent embeddings may better separate disease states

**Disadvantages:**
- Under-represents healthy tissue variation
- May bias the model toward disease features
- Could miss subtle variations in healthy tissue

### Standard Model (No Weighting)

**Advantages:**
- Balanced reconstruction across all regions
- Treats healthy and diseased tissue equally
- Better captures overall leaf morphology
- Unbiased representation in latent space

**Disadvantages:**
- May have slightly worse reconstruction of diseased regions
- Less specialized for disease-specific features

## Directory Structure

```
src/autoencoder_no_weighting/
├── __init__.py           # Package initialization
├── config.py             # Configuration (updated paths, removed disease params)
├── model.py              # U-Net autoencoder (identical to original)
├── dataset.py            # Data loading (identical to original)
├── loss.py               # SimpleMSELoss (removed disease weighting)
├── train.py              # Training script (uses SimpleMSELoss)
├── README.md             # This file
├── models/               # Saved model checkpoints
├── embeddings/           # Latent embeddings
├── visualizations/       # Reconstruction visualizations
└── logs/                 # Training logs and history
```

## How to Train the Model

### 1. Basic Training

```bash
cd /home/schnable/Documents/fieldLeafImaging
python -m src.autoencoder_no_weighting.train
```

### 2. Training Output

The training script will:
1. Load and split data by genotype (70% train, 15% val, 15% test)
2. Compute LAB color space statistics from training set
3. Create U-Net autoencoder model
4. Train using **Simple MSE loss** (no disease weighting)
5. Save checkpoints to `src/autoencoder_no_weighting/models/`
6. Save training history to `src/autoencoder_no_weighting/logs/`

### 3. Monitor Training

Training progress is displayed with:
```
Epoch 50/200
  Train Loss: 0.012345 (rec: 0.012000, l1: 0.000345)
  Val Loss:   0.013456 (rec: 0.013100, l1: 0.000356)
```

Key differences from disease-weighted version:
- No "mean_weight" statistic (always 1.0)
- Simpler loss breakdown

## Model Files

### Checkpoint Format

```python
checkpoint = {
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_loss': best_val_loss,
    'config': {
        'loss_type': 'simple_mse',  # Indicates no disease weighting
        ...
    }
}
```

### Loading a Trained Model

```python
from src.autoencoder_no_weighting.config import AutoencoderConfig
from src.autoencoder_no_weighting.model import create_model
import torch

# Load config and model
config = AutoencoderConfig()
model = create_model(config)

# Load checkpoint
checkpoint = torch.load('src/autoencoder_no_weighting/models/checkpoint_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extract embeddings
with torch.no_grad():
    embeddings = model.encode(input_tensor)
```

## Comparing to Disease-Weighted Version

Run the comparison script to see detailed differences:

```bash
python -m src.compare_models
```

This will show:
- Model architecture comparison (should be identical)
- Parameter count comparison (should be identical)
- Loss function differences (main change)
- Configuration differences
- Expected performance differences

## Expected Performance Differences

### Reconstruction Quality

| Region Type | Disease-Weighted | No Weighting |
|-------------|-----------------|--------------|
| Diseased areas | **Better** | Good |
| Healthy areas | Good | **Better** |
| Overall | Specialized | **Balanced** |

### Latent Embeddings

| Characteristic | Disease-Weighted | No Weighting |
|----------------|-----------------|--------------|
| Disease separation | **Strong** | Moderate |
| Healthy variation | Moderate | **Strong** |
| Overall structure | Biased to disease | **Unbiased** |

### Use Cases

**Use Disease-Weighted Version When:**
- Primary goal is disease detection/classification
- Need maximum sensitivity to disease features
- Healthy tissue variation is less important

**Use No-Weighting Version When:**
- Need balanced representation of all tissue
- Interested in overall leaf morphology
- Want unbiased embeddings for clustering
- Comparing healthy and diseased equally important

## Implementation Details

### Loss Function

```python
class SimpleMSELoss(nn.Module):
    """Simple MSE loss - all pixels weighted equally."""

    def forward(self, predictions, targets, masks, embeddings=None, lab_stats=None):
        # Standard MSE reconstruction loss
        squared_error = (predictions - targets) ** 2
        reconstruction_loss = squared_error.mean()

        # Optional L1 regularization on embeddings
        l1_loss = self.l1_weight * torch.abs(embeddings).mean()

        total_loss = reconstruction_loss + l1_loss
        return total_loss, loss_dict
```

### Code Comments

Throughout the modified code, you'll find comments indicating changes:

```python
# Removed disease weighting - using standard MSE loss
# Original: loss = weighted_mse_loss(output, target, disease_weights)
# Modified: loss = F.mse_loss(output, target)
```

## Training Hyperparameters

All hyperparameters remain identical to the disease-weighted version:

```python
# Model
embedding_dim = 256
unet_features = [64, 128, 256, 512]
use_attention = True

# Training
batch_size = 16
num_epochs = 200
learning_rate = 1e-4
weight_decay = 1e-5
patience = 15  # Early stopping

# Loss
l1_regularization = 1e-5  # On embeddings
# disease_weight_strength removed (not applicable)
```

## Troubleshooting

### Issue: Different Results Than Expected

**Solution:** Ensure you're comparing models trained on the same data splits:
```python
# Both models should use the same random seed
config.random_seed = 42
```

### Issue: Loss Values Seem Different

**Solution:** This is expected! Disease-weighted loss values are not directly comparable to unweighted loss values because of the different weighting schemes. Compare reconstruction quality visually instead.

### Issue: Model Not Improving

**Solution:** Check that:
1. LAB statistics are computed correctly
2. Data augmentation is working
3. Learning rate scheduler is active
4. Early stopping patience is appropriate

## Citation

If you use this model, please cite:

```bibtex
@software{autoencoder_no_weighting,
  title = {Standard Autoencoder for Maize Leaf Reconstruction},
  author = {Your Name},
  year = {2025},
  note = {Modified from disease-aware autoencoder to remove disease weighting}
}
```

## Related Files

- Original disease-weighted version: `src/disease_autoencoder_cropped/`
- Comparison script: `src/compare_models.py`
- Vanilla autoencoder: `src/autoencoder/` (earlier version)

## Contact

For questions or issues, please check:
1. This README
2. The comparison script output (`python -m src.compare_models`)
3. Original disease-weighted version documentation
