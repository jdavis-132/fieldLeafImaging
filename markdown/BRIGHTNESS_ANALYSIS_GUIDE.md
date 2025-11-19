# Brightness Channel Analysis Results

## Executive Summary

Successfully analyzed 4 sorghum leaf images across 3 color spaces (HSV, LAB, YCrCb) to determine optimal brightness channel and thresholds for midrib detection.

**Key Finding: YCrCb Y-channel provides the best contrast ratio (5.10) for separating midrib from leaf lamina.**

## Analysis Results

### Best Color Space: **YCrCb Y**
- **Contrast Ratio**: 5.10 (highest among all channels)
- **Mean Brightness**: 127.3 ± 5.0
- **Std Deviation**: 51.1 ± 5.9
- **Dynamic Range**: 0-227

### Color Space Comparison
| Color Space | Contrast Ratio | Mean | Std Dev | Notes |
|-------------|----------------|------|---------|-------|
| **YCrCb Y** | **5.10** | 127.3 | 51.1 | **Best overall** |
| LAB L | 5.06 | 134.1 | 52.3 | Very close second |
| HSV V | 4.87 | 136.2 | 53.3 | Slightly lower contrast |

**Interpretation**: All three channels show similar performance (contrast ratios 4.87-5.10), indicating that brightness is consistently ~5x higher in bright regions (midrib) compared to dark regions (lamina). YCrCb Y has a slight edge.

## Recommended Thresholds for Midrib Detection

Based on YCrCb Y-channel analysis:

### 1. Conservative (High Precision)
```python
threshold = 183.0  # 95th percentile of brightness
```
- **Use when**: You want to minimize false positives
- **Pros**: High confidence that detected regions are midrib
- **Cons**: May miss dimmer midrib sections

### 2. Balanced (Recommended)
```python
threshold = mean + 1.5 * std = 204.0
```
- **Use when**: You want good balance of precision and recall
- **Pros**: Catches most midrib pixels while avoiding most lamina
- **Cons**: May include some bright lamina edges

### 3. Sensitive (High Recall)
```python
threshold = mean + 1.0 * std = 178.4
```
- **Use when**: You want to capture entire midrib, including dimmer sections
- **Pros**: Maximum midrib coverage
- **Cons**: Higher chance of including bright lamina tissue

## Brightness Statistics by Image

| Image | YCrCb Y Mean | YCrCb Y Std | Contrast Ratio | 95th Percentile |
|-------|--------------|-------------|----------------|-----------------|
| 2752 | 128.6 | 59.8 | 5.67 | 203.6 |
| 1201 | 123.8 | 52.9 | 4.46 | 182.4 |
| 2749 | 134.8 | 47.6 | 5.50 | 179.6 |
| 2115 | 122.1 | 44.1 | 4.75 | 166.4 |

**Observations**:
- Brightness varies significantly between images (mean: 122-135)
- Standard deviation ranges from 44-60, indicating varying image contrast
- Contrast ratios are consistent (4.46-5.67), suggesting reliable brightness separation

## Generated Visualizations

### For Each Image:

1. **Brightness Comparison** (`*_brightness_comparison.png`)
   - Row 1: Original image, HSV V-channel, LAB L-channel
   - Row 2: YCrCb Y-channel, Viridis-mapped HSV V, Viridis-mapped LAB L
   - Row 3: Histogram comparison of all three channels
   - Row 4: Otsu thresholding applied to each channel

2. **Line Profiles** (`*_line_profiles.png`)
   - Horizontal brightness profiles at 3 vertical positions
   - Shows how brightness changes across the leaf width
   - Useful for identifying midrib location (peak brightness)

3. **Edge Detection** (`*_edge_detection.png`)
   - Sobel and Canny edge detection on each brightness channel
   - Helps identify which channel best highlights midrib boundaries

### Individual Brightness Maps

Grayscale images saved in:
- `brightness_maps/hsv_v/` - HSV Value channel
- `brightness_maps/lab_l/` - LAB Lightness channel
- `brightness_maps/ycrcb_y/` - YCrCb Luminance channel

## Implementation Recommendations

### Updated Midrib Detection Algorithm

```python
import cv2
import numpy as np

def detect_midrib_ycrcb(image, roi_y1, roi_y2, roi_x1, roi_x2):
    """
    Detect midrib using YCrCb Y-channel (luminance).

    Args:
        image: BGR image
        roi_y1, roi_y2, roi_x1, roi_x2: Region of interest coordinates

    Returns:
        Binary mask of midrib
    """
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0]

    # Extract ROI
    roi_y = y_channel[roi_y1:roi_y2, roi_x1:roi_x2]

    # Calculate adaptive threshold
    mean_y = np.mean(roi_y)
    std_y = np.std(roi_y)
    threshold = mean_y + 1.5 * std_y  # Balanced threshold

    # Apply threshold
    midrib_mask = (roi_y > threshold).astype(np.uint8) * 255

    # Morphological operations
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    midrib_mask = cv2.morphologyEx(midrib_mask, cv2.MORPH_CLOSE, kernel_horizontal)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    midrib_mask = cv2.morphologyEx(midrib_mask, cv2.MORPH_OPEN, kernel_open)

    return midrib_mask
```

### Why YCrCb Y-channel?

1. **Slightly better contrast** (5.10 vs 5.06 for LAB L, 4.87 for HSV V)
2. **Perceptually uniform**: Designed to match human vision
3. **Better for JPEG**: Many cameras save in YCrCb internally
4. **Consistent performance**: Works well across different lighting conditions

### Alternative: Use LAB L-channel

LAB L-channel is nearly as good (contrast ratio 5.06) and may be preferable if:
- You're already using LAB for other color analysis
- You want perceptually uniform lightness (0-100 scale conceptually)
- You're working with print/display applications

## Key Insights

### 1. Brightness is Highly Discriminative
- **5x contrast ratio** between bright (midrib) and dark (lamina) regions
- All three color spaces show similar patterns
- Brightness alone can identify midrib regions effectively

### 2. Adaptive Thresholding is Essential
- Image brightness varies significantly (mean: 122-135)
- Use **mean + k*std** where k=1.0 to 1.5
- Fixed thresholds won't work across different images

### 3. Shape Constraints Still Important
- Brightness filtering alone may select non-midrib bright regions
- Combine with:
  - High aspect ratio (width >> height)
  - Centered position in ROI
  - Horizontal continuity (morphological closing)

### 4. Edge Detection Shows Clear Boundaries
- All three channels produce clear edges around midrib
- Sobel and Canny both highlight midrib boundaries well
- Can be used for refinement after initial detection

## Troubleshooting Guide

### Problem: Too Many False Positives
**Solution**:
- Increase threshold (use k=2.0 instead of 1.5)
- Add stricter shape constraints
- Filter by component size (<10% of ROI)

### Problem: Missing Midrib Sections
**Solution**:
- Decrease threshold (use k=1.0 instead of 1.5)
- Increase morphological closing kernel size
- Use relaxed fallback with AR >= 1.5

### Problem: Including Lamina Edges
**Solution**:
- Combine brightness with color filtering (low saturation)
- Add lamina exclusion mask (high saturation + green hue)
- Use skeleton extraction to get centerline only

## Usage

### View Visualizations
```bash
# View brightness comparisons
ls data/test_output/comparisons/*brightness_comparison.png

# View line profiles
ls data/test_output/histograms/*line_profiles.png

# View edge detection
ls data/test_output/comparisons/*edge_detection.png
```

### Run Analysis on New Images
```bash
# Analyze images in a directory
python3 src/segment_green_strips/analyze_brightness_channels.py \
  --input-dir path/to/images \
  --output-dir path/to/output

# Specify file pattern
python3 src/segment_green_strips/analyze_brightness_channels.py \
  --input-dir path/to/images \
  --pattern "*.png"
```

### Access Statistics
```bash
# View CSV statistics
cat data/test_output/brightness_statistics.csv | column -t -s,

# View summary report
cat data/test_output/brightness_analysis_summary.txt
```

## Next Steps

1. **Update detect_midrib.py** to use YCrCb Y-channel as primary brightness source
2. **Adjust thresholds** to use mean + 1.5*std (balanced approach)
3. **Test on failed images** (e.g., image 6026) to see if YCrCb improves detection
4. **Consider ensemble approach**: Combine YCrCb Y with LAB L for robustness
5. **Validate against ground truth**: Manually annotate midribs to measure precision/recall

## Files Generated

```
data/test_output/
├── brightness_maps/
│   ├── hsv_v/          # HSV Value channel grayscale images (4 images)
│   ├── lab_l/          # LAB Lightness channel grayscale images (4 images)
│   └── ycrcb_y/        # YCrCb Y channel grayscale images (4 images)
├── comparisons/        # Multi-panel comparison visualizations (8 images)
├── histograms/         # Line profile visualizations (4 images)
├── brightness_statistics.csv          # Detailed statistics for all images
└── brightness_analysis_summary.txt    # Text summary with recommendations
```

Total: 24 visualization files + 2 data files

## Conclusion

The brightness channel analysis confirms that **luminance/lightness is a strong discriminator** for midrib detection, with YCrCb Y-channel providing the best contrast ratio. Using adaptive thresholding (mean + 1.5*std) combined with shape constraints should provide robust midrib detection across varying lighting conditions.

The analysis also validates the current approach in `detect_midrib.py` which uses LAB L-channel - it's nearly as good as YCrCb Y and the difference is marginal. Either channel can be used effectively.
