# Changes from Disease-Weighted Autoencoder

## Summary

This document details the specific changes made to create the standard autoencoder (no disease weighting) from the original disease-aware autoencoder.

## Files Modified

### 1. `config.py`

**Changes:**
- Renamed class: `DiseaseConfig` → `AutoencoderConfig`
- Updated output directories: `disease_autoencoder_cropped/` → `autoencoder_no_weighting/`
- **Removed parameter:** `disease_weight_strength` (was 2.0)
- Updated docstrings to reflect removal of disease weighting
- Added `loss_type: 'simple_mse'` to saved config

**Unchanged:**
- All training hyperparameters (batch_size, learning_rate, etc.)
- Model architecture parameters (embedding_dim, unet_features, etc.)
- Data split ratios
- Random seed

### 2. `loss.py`

**Changes:**
- **Removed:** `DiseaseWeightedLoss` class with disease weight computation
- **Kept:** `SimpleMSELoss` class (modified to be the default)
- Modified `create_loss_function()` to only create SimpleMSELoss
- Removed `use_disease_weighting` parameter
- Added detailed comments explaining the changes

**Key Code Change:**
```python
# Original (disease_autoencoder_cropped/loss.py):
def create_loss_function(config, use_disease_weighting=True):
    if use_disease_weighting:
        loss_fn = DiseaseWeightedLoss(config)
    else:
        loss_fn = SimpleMSELoss()
    return loss_fn

# Modified (autoencoder_no_weighting/loss.py):
def create_loss_function(config):
    # Removed disease weighting option - always use simple MSE
    loss_fn = SimpleMSELoss(config)
    return loss_fn
```

### 3. `train.py`

**Changes:**
- Updated imports to use `autoencoder_no_weighting` modules
- Modified loss function creation (line 184):
  ```python
  # Original: loss_fn = create_loss_function(self.config, use_disease_weighting=True)
  # Modified: loss_fn = create_loss_function(self.config)
  ```
- Updated class docstring to mention "no disease weighting"
- Added "Loss type: Simple MSE" to training startup message
- Removed "weight" statistic from progress bar
- Updated checkpoint to include `loss_type: 'simple_mse'`

**Unchanged:**
- Training loop structure
- Validation procedure
- Checkpoint saving logic
- Early stopping
- Learning rate scheduling

### 4. `model.py`

**Changes:**
- Updated import in `__main__` section: `DiseaseConfig` → `AutoencoderConfig`

**Unchanged:**
- **Entire model architecture** (U-Net with attention)
- All classes: `ConvBlock`, `SpatialAttention`, `ChannelAttention`, `AttentionBlock`, `Encoder`, `Decoder`, `DiseaseAutoencoder`
- Number of parameters: 31,438,724

### 5. `dataset.py`

**Changes:**
- Updated docstring to mention "standard autoencoder"
- Added note that dataset is identical to disease-weighted version

**Unchanged:**
- **All data loading logic**
- LAB color space conversion
- Cropping and resizing
- Augmentation
- Normalization

### 6. `__init__.py`

**Changes:**
- Updated docstring from "Disease-aware" to "Standard"
- Added note about removal of disease weighting

## Files Created

### 1. `README.md`

Comprehensive documentation including:
- What changed from the original model
- Why remove disease weighting
- How to train the new model
- Comparison to disease-weighted version
- Expected performance differences
- Troubleshooting guide

### 2. `../compare_models.py`

Comparison script that shows:
- Model architecture comparison (identical)
- Parameter count comparison (identical)
- Loss function differences (main change)
- Configuration differences
- Expected performance trade-offs

### 3. `CHANGES.md` (this file)

Detailed change log documenting all modifications.

## What Was NOT Changed

These components are **completely identical** to the disease-weighted version:

✅ **Model Architecture:**
- U-Net encoder-decoder structure
- Attention mechanisms (spatial and channel)
- Skip connections
- Number of parameters: 31,438,724
- Embedding dimension: 256

✅ **Data Processing:**
- Image loading and cropping
- LAB color space conversion
- Normalization procedure
- Data augmentation (rotations, flips, brightness)

✅ **Training Hyperparameters:**
- Batch size: 16
- Learning rate: 1e-4
- Weight decay: 1e-5
- Number of epochs: 200
- Early stopping patience: 15
- Learning rate scheduler settings
- Random seed: 42

✅ **Optimizer and Scheduler:**
- AdamW optimizer
- ReduceLROnPlateau scheduler

## Key Functional Differences

### Loss Computation

| Aspect | Disease-Weighted | No Weighting |
|--------|-----------------|--------------|
| **Per-pixel weights** | 1.0 to 2.0 based on color | Always 1.0 |
| **LAB color analysis** | Computes deviation from healthy tissue | Not used |
| **Mean weight** | Variable (e.g., 1.641) | Always 1.0 |
| **Reconstruction focus** | Biased toward diseased regions | Balanced across all regions |

### Example Loss Values

For the same input:
- **Disease-weighted loss:** 6.022 (total), mean_weight: 1.641
- **Standard loss:** 2.000 (total), mean_weight: 1.000

**Note:** Loss values are not directly comparable due to different weighting schemes.

## Expected Performance Impact

### Reconstruction Quality

**Disease-Weighted Advantages:**
- ✓ Better reconstruction of diseased regions
- ✓ More sensitive to disease-specific features
- ✓ Embeddings may better separate disease states

**Standard (No Weighting) Advantages:**
- ✓ Balanced reconstruction quality
- ✓ Better representation of healthy tissue variation
- ✓ Unbiased latent embeddings
- ✓ Better for morphology analysis

### Use Case Recommendations

**Use Disease-Weighted When:**
- Primary goal is disease detection/classification
- Need maximum sensitivity to disease features
- Diseased regions are most important

**Use No-Weighting When:**
- Need balanced tissue representation
- Interested in overall leaf morphology
- Want unbiased embeddings for clustering
- Comparing healthy and diseased tissue equally

## Testing

All components have been tested and verified:

✅ Loss function test: `python -m src.autoencoder_no_weighting.loss`
```
✓ Loss function test passed!
Mean weight: 1.0 (always 1.0 for unweighted loss)
```

✅ Model test: `python -m src.autoencoder_no_weighting.model`
```
✓ All shape tests passed!
Total parameters: 31,438,724
```

✅ Comparison test: `python -m src.compare_models`
```
✓ Models have identical architectures!
Parameter difference: 0 parameters
```

## Training the New Model

To train the standard autoencoder:

```bash
cd /home/schnable/Documents/fieldLeafImaging
python -m src.autoencoder_no_weighting.train
```

Output files will be saved to:
- Models: `src/autoencoder_no_weighting/models/`
- Embeddings: `src/autoencoder_no_weighting/embeddings/`
- Logs: `src/autoencoder_no_weighting/logs/`
- Visualizations: `src/autoencoder_no_weighting/visualizations/`

## Version History

**v1.0.0** (2025-11-06)
- Initial creation from disease-aware autoencoder
- Removed disease weighting from loss function
- Kept all other components identical
- Added comprehensive documentation
- Created comparison script

## Related Files

- Original: `src/disease_autoencoder_cropped/`
- Comparison: `src/compare_models.py`
- Earlier version: `src/autoencoder/` (vanilla autoencoder)
