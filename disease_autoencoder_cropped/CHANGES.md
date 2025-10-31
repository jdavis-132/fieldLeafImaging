# Changes in disease_autoencoder_cropped

This directory contains a modified version of the disease autoencoder that handles non-leaf pixels differently from the original version.

## Summary of Changes

The original `disease_autoencoder/` code was modified to create this `disease_autoencoder_cropped/` version with the following key differences:

### 1. Non-Leaf Pixel Values (dataset.py)
**Original behavior:**
- Non-leaf pixels (areas outside the segmentation mask) were set to 0 after normalization
- Code: `lab_image = lab_image * mask_3ch` (line 172 in original)

**New behavior:**
- Non-leaf pixels retain their original normalized values
- All pixels within the cropped bounding box are preserved
- **Modified location:** `disease_autoencoder_cropped/dataset.py:170-172`

### 2. Loss Weighting (loss.py)
**Original behavior:**
- Pixel weights were multiplied by the mask, effectively setting weights to 0 for non-leaf pixels
- Code: `weights = weights * mask` (line 90 in original)
- Only masked (leaf) pixels contributed to the loss

**New behavior:**
- All pixels within the cropped region contribute to the loss
- Disease-based weighting is applied uniformly across all pixels
- Fallback uses uniform weights of 1.0 for all pixels (not just masked ones)
- **Modified locations:**
  - `disease_autoencoder_cropped/loss.py:94-96` (DiseaseWeightedLoss.compute_disease_weights)
  - `disease_autoencoder_cropped/loss.py:123` (DiseaseWeightedLoss.forward)
  - `disease_autoencoder_cropped/loss.py:164-169` (SimpleMSELoss.forward)

### 3. Configuration Paths (config.py)
**Original behavior:**
- Output directories pointed to `disease_autoencoder/`

**New behavior:**
- Output directories point to `disease_autoencoder_cropped/`
- Ensures separate models, embeddings, visualizations, and logs
- **Modified location:** `disease_autoencoder_cropped/config.py:23-26`

## Files Modified

1. **dataset.py**
   - Removed line that zeros out background pixels
   - Updated docstring to reflect new behavior
   - Location: Lines 170-172

2. **loss.py**
   - Removed mask multiplication from weight computation
   - Changed fallback weights to uniform 1.0 instead of mask values
   - Updated SimpleMSELoss to compute loss over all pixels
   - Locations: Lines 94-96, 123, 164-169

3. **config.py**
   - Updated output directory paths to use `disease_autoencoder_cropped/`
   - Updated docstrings and config string representation
   - Locations: Lines 1-5, 23-26, 96, 105

## Behavioral Differences

### Image Preprocessing
- **Original:** Crops to bounding box → Resizes → Normalizes → **Zeros background**
- **Cropped:** Crops to bounding box → Resizes → Normalizes → **Keeps all values**

### Loss Computation
- **Original:** Only leaf pixels (within mask) contribute to loss
- **Cropped:** All pixels in cropped region contribute to loss

### Expected Impact
1. The model will learn to reconstruct background areas within the leaf bounding box
2. May improve boundary reconstruction and spatial context understanding
3. Disease weighting still emphasizes diseased regions, but background also matters
4. Potentially more stable training since loss is computed over more pixels

## Compatibility

- The modified code maintains the same API and can be used as a drop-in replacement
- Model architecture remains identical (same input/output shapes)
- Models trained with either version can be loaded by the other (though behavior differs)
- Data loading and preprocessing pipeline is otherwise unchanged

## Usage

Use this version exactly like the original:

```python
from disease_autoencoder_cropped.config import DiseaseConfig
from disease_autoencoder_cropped.dataset import get_dataloaders
from disease_autoencoder_cropped.loss import create_loss_function
from disease_autoencoder_cropped.model import DiseaseAutoencoder

# Rest of code is identical to original
```

## Testing Recommendations

When using this modified version, you should:

1. Compare reconstruction quality on test images with both versions
2. Check if background regions are reconstructed meaningfully
3. Monitor training stability and convergence
4. Evaluate whether disease detection/representation improves
5. Visualize embeddings to see if they capture different features

## Reverting to Original

If you need the original behavior, simply use the files in `disease_autoencoder/` instead of `disease_autoencoder_cropped/`.
