# SSIM and PSNR Dual Metrics Implementation

## Summary

Modified `src/comprehensive_model_comparison.py` to properly compute SSIM and PSNR metrics **both with and without considering masks**. This provides a more comprehensive evaluation of model performance by separating full-image reconstruction quality from masked-region (foreground) reconstruction quality.

## Changes Made

### 1. Updated Metric Computation Functions (Lines 496-653)

#### `compute_ssim()` - Now returns dual metrics
**Before:** Returned single SSIM value, mask parameter was ignored
**After:** Returns tuple `(ssim_full, ssim_masked)`

- **`ssim_full`**: Computed over entire image (ignoring mask)
- **`ssim_masked`**: Computed only over masked regions where mask > 0
- Handles edge cases: empty masks return `None` for masked metric
- Includes detailed error handling with warnings

```python
def compute_ssim(pred, target, mask=None, compute_full=True, compute_masked=True) -> Tuple[Optional[float], Optional[float]]
```

#### `compute_psnr()` - Now returns dual metrics
**Before:** Returned single PSNR value, mask parameter was ignored
**After:** Returns tuple `(psnr_full, psnr_masked)`

- **`psnr_full`**: PSNR over entire image (ignoring mask)
- **`psnr_masked`**: PSNR over masked regions only
  - Extracts only masked pixels for MSE computation
  - Computes data range from masked pixels only
  - Returns `None` for empty masks

```python
def compute_psnr(pred, target, mask=None, compute_full=True, compute_masked=True) -> Tuple[Optional[float], Optional[float]]
```

**Key Implementation Details:**
- For masked SSIM: zeroes out background pixels, computes SSIM over entire image
- For masked PSNR: extracts only masked pixels, computes MSE and data range from those pixels
- Both properly handle multi-channel images
- NaN-safe computation using `np.nanmean()` and `np.nanstd()`

---

### 2. Updated ModelMetrics Dataclass (Lines 660-706)

**Added new metric fields:**
```python
# Full-image metrics (entire image, ignoring mask)
ssim_full_mean: float
ssim_full_std: float
psnr_full_mean: float
psnr_full_std: float

# Masked-region metrics (only masked pixels considered)
ssim_masked_mean: float
ssim_masked_std: float
psnr_masked_mean: float
psnr_masked_std: float
```

**Updated per-image metrics:**
```python
per_image_ssim_full: List[float]
per_image_ssim_masked: List[float]
per_image_psnr_full: List[float]
per_image_psnr_masked: List[float]
```

**Backward Compatibility:**
- All new metrics properly initialized
- `to_dict()` method updated to exclude per-image metrics from serialization
- Maintains existing MSE and weighted MSE metrics

---

### 3. Updated Evaluation Loop (Lines 874-1002)

**Metrics Storage:**
```python
ssim_full_list = []
ssim_masked_list = []
psnr_full_list = []
psnr_masked_list = []
```

**Computation Code (Lines 974-1002):**
```python
# Compute SSIM (both full and masked)
ssim_full, ssim_masked = LossCalculator.compute_ssim(
    pred_rgb, target_rgb, mask_np,
    compute_full=True, compute_masked=True
)
ssim_full_list.append(ssim_full if ssim_full is not None else 0.0)
ssim_masked_list.append(ssim_masked if ssim_masked is not None else float('nan'))

# Compute PSNR (both full and masked)
psnr_full, psnr_masked = LossCalculator.compute_psnr(
    pred_rgb, target_rgb, mask_np,
    compute_full=True, compute_masked=True
)
psnr_full_list.append(psnr_full if psnr_full is not None else 0.0)
psnr_masked_list.append(psnr_masked if psnr_masked is not None else float('nan'))
```

**Error Handling:**
- Try-except blocks for both SSIM and PSNR
- Informative warning messages on failure
- Graceful degradation (stores 0.0 or NaN on error)

---

### 4. Updated ModelMetrics Creation (Lines 1024-1066)

**NaN-safe statistics computation:**
```python
ssim_full_mean = float(np.mean(ssim_full_list))
ssim_full_std = float(np.std(ssim_full_list))
ssim_masked_mean = float(np.nanmean(ssim_masked_list))  # Handles NaN values
ssim_masked_std = float(np.nanstd(ssim_masked_list))

psnr_full_mean = float(np.mean(psnr_full_list))
psnr_full_std = float(np.std(psnr_full_list))
psnr_masked_mean = float(np.nanmean(psnr_masked_list))  # Handles NaN values
psnr_masked_std = float(np.nanstd(psnr_masked_list))
```

Uses `np.nanmean()` and `np.nanstd()` to properly handle NaN values from missing/empty masks.

---

### 5. Updated Console Output (Lines 834-839)

**Before:**
```python
print(f"  ✓ SSIM: {metrics.ssim_mean:.4f}")
print(f"  ✓ PSNR: {metrics.psnr_mean:.2f} dB")
```

**After:**
```python
print(f"  ✓ SSIM (full): {metrics.ssim_full_mean:.4f} ± {metrics.ssim_full_std:.4f}")
print(f"  ✓ SSIM (masked): {metrics.ssim_masked_mean:.4f} ± {metrics.ssim_masked_std:.4f}")
print(f"  ✓ PSNR (full): {metrics.psnr_full_mean:.2f} ± {metrics.psnr_full_std:.2f} dB")
print(f"  ✓ PSNR (masked): {metrics.psnr_masked_mean:.2f} ± {metrics.psnr_masked_std:.2f} dB")
```

Clearly labels full vs masked metrics in output.

---

### 6. Updated Visualization: Metrics Comparison (Lines 1274-1358)

**Before:** Single row with 3 plots (MSE, SSIM, PSNR)
**After:** 2 rows × 3 columns grid:

**Row 1 - Full-Image Metrics:**
- MSE (Full Image)
- SSIM (Full Image)
- PSNR (Full Image)

**Row 2 - Masked-Region Metrics:**
- MSE (Masked Region)
- SSIM (Masked Region)
- PSNR (Masked Region)

**Features:**
- Color-coded bars for easy distinction
- Error bars showing standard deviation
- Clear titles indicating full vs masked
- Consistent y-axis limits for SSIM (0-1)

---

### 7. Updated Visualization: Error Distributions (Lines 1360-1473)

**Before:** 2×2 grid (MSE, Weighted MSE, SSIM, PSNR)
**After:** 3×2 grid showing:

**Row 1:** MSE distributions
- MSE (Full Image)
- Weighted MSE

**Row 2:** SSIM distributions
- SSIM (Full Image)
- SSIM (Masked Region)

**Row 3:** PSNR distributions
- PSNR (Full Image)
- PSNR (Masked Region)

**Features:**
- Filters out NaN values before plotting masked metrics
- Separate histograms for full vs masked
- Clear titles and legends
- Overlaid distributions for all models

---

### 8. Updated Interpretation Report (Lines 1654-1680)

**Added separate best-model detection for:**
- Best SSIM (Full Image)
- Best SSIM (Masked Region)
- Best PSNR (Full Image)
- Best PSNR (Masked Region)

**Output Example:**
```
Full-Image Structural Similarity (highest SSIM): disease_autoencoder_cropped
  SSIM (full): 0.8234

Masked-Region Structural Similarity (highest SSIM): disease_autoencoder
  SSIM (masked): 0.8567

Full-Image Peak Signal-to-Noise Ratio (highest PSNR): autoencoder_no_weighting
  PSNR (full): 28.45 dB

Masked-Region Peak Signal-to-Noise Ratio (highest PSNR): disease_autoencoder
  PSNR (masked): 31.23 dB
```

This reveals which model excels at different aspects of reconstruction.

---

### 9. Updated Docstring (Lines 1-27)

Updated module docstring to reflect:
- Dual metric computation for SSIM and PSNR
- Distinction between full-image and masked-region metrics
- Updated date to 2025-11-12

---

## Benefits of These Changes

### 1. **More Accurate Model Comparison**
- Separates background reconstruction from foreground (leaf) reconstruction
- Background may dominate full-image metrics, masking model differences
- Masked metrics focus on what matters: the leaf region

### 2. **Better Understanding of Model Behavior**
- Can identify if a model excels at foreground or background reconstruction
- Reveals trade-offs between full-image vs masked-region performance
- Helps choose the right model for the specific task

### 3. **Maintains Backward Compatibility**
- All existing metrics still computed
- Full-image metrics preserve previous behavior
- Masked metrics add new information without removing old

### 4. **Robust Error Handling**
- Handles empty masks gracefully
- NaN-safe statistics computation
- Clear warnings for failed computations
- Graceful degradation on errors

### 5. **Comprehensive Visualization**
- Side-by-side comparison of full vs masked
- Easy to spot differences between models
- Publication-ready figures
- Clear labeling prevents confusion

---

## Usage Example

When you run the script now, you'll see output like:

```
Evaluating disease_autoencoder...
  ✓ MSE: 0.001234 ± 0.000123
  ✓ SSIM (full): 0.8234 ± 0.0156
  ✓ SSIM (masked): 0.8567 ± 0.0142
  ✓ PSNR (full): 28.45 ± 2.34 dB
  ✓ PSNR (masked): 31.23 ± 2.67 dB
```

And the CSV output (`summary_table.csv`) will include columns:
- `ssim_full_mean`, `ssim_full_std`
- `ssim_masked_mean`, `ssim_masked_std`
- `psnr_full_mean`, `psnr_full_std`
- `psnr_masked_mean`, `psnr_masked_std`

---

## Technical Notes

### SSIM Computation Details

**Full-Image SSIM:**
- Uses entire image as-is
- Standard scikit-image SSIM computation
- Data range from full image min/max

**Masked-Region SSIM:**
- Zeros out background pixels (mask == 0)
- Computes SSIM over entire image with zeroed background
- Data range computed only from foreground pixels
- This effectively measures SSIM only in the region of interest

### PSNR Computation Details

**Full-Image PSNR:**
- Standard PSNR computation over all pixels
- Data range from full image min/max

**Masked-Region PSNR:**
- Extracts only masked pixels into 1D arrays
- Computes MSE only from these pixels
- Data range from masked pixels min/max
- Formula: `10 * log10(data_range^2 / MSE_masked)`

### Edge Case Handling

1. **Empty Masks:** Return `None` for masked metrics
2. **NaN Values:** Use `np.nanmean()` and `np.nanstd()` for statistics
3. **Zero MSE:** Cap PSNR at 100.0 to avoid inf
4. **Computation Errors:** Catch exceptions, print warnings, store fallback values

---

## Testing Recommendations

1. **Verify Dual Output:** Check that both full and masked metrics are computed
2. **Check Empty Masks:** Ensure script handles images with no foreground gracefully
3. **Compare Values:** Masked metrics should differ from full metrics when background is present
4. **Validate CSV:** Confirm all new metric columns appear in output files
5. **Inspect Plots:** Verify 2x3 grid shows correct full vs masked comparisons

---

## Files Modified

- `src/comprehensive_model_comparison.py` - Main comparison script (1725 lines)

**Key sections modified:**
- Lines 496-653: Metric computation functions
- Lines 660-706: ModelMetrics dataclass
- Lines 874-1002: Evaluation loop
- Lines 1024-1066: Metrics aggregation
- Lines 1274-1358: Metrics comparison plot
- Lines 1360-1473: Error distribution plot
- Lines 1654-1680: Interpretation section

---

## Future Enhancements (Optional)

1. **Weighted Masked Metrics:** Apply disease weighting to masked metrics
2. **Per-Channel Metrics:** Separate metrics for L, a, b channels
3. **Spatial Heatmaps:** Show where models perform best/worst within mask
4. **Mask Coverage Analysis:** Report percentage of image covered by mask
5. **Boundary Metrics:** Evaluate reconstruction quality near mask boundaries

---

## Summary

The script now provides a **dual-metric evaluation** that separates full-image reconstruction quality from foreground-only quality. This gives researchers a more nuanced understanding of model performance and helps identify which model best reconstructs the actual leaf tissue (foreground) versus the entire image including background.

**Key Takeaway:**
Masked metrics focus evaluation on what matters most - the leaf region - while full metrics preserve the ability to assess overall reconstruction including background.
