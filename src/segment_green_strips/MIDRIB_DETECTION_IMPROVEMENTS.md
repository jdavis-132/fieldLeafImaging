# Midrib Detection Algorithm Improvements

## Problem Statement
The original midrib detection algorithm was incorrectly selecting leaf lamina tissue as the midrib instead of the actual midrib (the bright central vein between the two leaf strips).

## Root Cause Analysis

### Diagnostic Investigation
I created diagnostic scripts to analyze the actual color properties of midrib vs. lamina tissue in the test images. Key findings:

**Image 2749 Analysis:**
- **Green Lamina**: H=45°, S=55.5, V=51.6, L=52.6, B=134.3
- **Midrib (actual)**: H=15.4°, S=18.6, V=124.7, L=129.5, B=131.1

### Key Insights
1. **Midrib is MUCH brighter** than lamina (L=129.5 vs 52.6, ~2.5x brighter!)
2. **Midrib has MUCH lower saturation** (S=18.6 vs 55.5, ~1/3 the saturation)
3. **Midrib has different hue** (H=15° yellow-orange vs H=45° green)
4. **B channel is NOT discriminative** - midrib can have LOWER B values than lamina (contrary to initial assumption)

### Original Algorithm Problems
1. **Brightness threshold too low**: Used mean + 0.5*std, which was not discriminative enough
2. **Wrong color constraints**:
   - Required B > mean_green_b + 10 (midrib actually has similar or lower B)
   - Saturation threshold calculation used wrong channel (A instead of S)
   - Too narrow hue range (30-75°, missing yellow-orange midribs at 15-30°)
3. **Component selection too strict**: Required aspect ratio >=3.0 AND height <40% of ROI AND width >30%, which was too restrictive
4. **No size filtering**: Allowed selection of million-pixel components that were clearly not midrib

## Improvements Implemented

### 1. Stricter Brightness Filtering
```python
# Changed from mean + 0.5*std to mean + 2.0*std
l_threshold = min(200, mean_green_l + 2.0 * std_green_l)
v_threshold = min(220, mean_green_v + 2.0 * std_green_v)
```
**Rationale**: Midrib is 2-3x brighter than lamina, so we need a much higher threshold to be discriminative.

### 2. Fixed Color Constraints
```python
# Removed incorrect B channel requirement
# Fixed saturation calculation to use actual HSV saturation
mean_green_s = np.mean(hsv[:,:,1][green_mask > 0])
max_saturation = min(70, mean_green_s * 0.55)

# Broadened hue range to include yellow-orange midribs
hue_mask = (h_channel >= 8) & (h_channel <= 75)
```
**Rationale**: Use correct saturation channel, and accommodate variable midrib hues (yellow-orange to yellow-green).

### 3. Adaptive Color Filtering with Fallback
```python
# Adaptive saturation threshold based on lamina saturation
if mean_green_s > 100:
    max_saturation = min(80, mean_green_s * 0.6)
else:
    max_saturation = min(70, mean_green_s * 0.55)

# Multiple color criteria with OR logic
low_sat_mask = (s_channel < max_saturation) & (s_channel > 5) & hue_mask
moderate_sat_mask = (s_channel < max_saturation * 1.5) & (roi_l > l_threshold + 20) & hue_mask
color_mask = low_sat_mask | moderate_sat_mask

# Fallback to brightness-only if color filter too strict
if combined_strict_count > 100:
    combined = combined_strict
elif combined_relaxed_count > 100:
    combined = combined_relaxed
```
**Rationale**: Different images have different lighting conditions. Brightness is the most reliable indicator, so we use it as primary filter with color as secondary.

### 4. Improved Component Selection
```python
# Filter out components too large to be midrib
max_midrib_area = roi_area * 0.10

# Relaxed minimum requirements
meets_requirements = (aspect_ratio >= 2.5 and  # Was 3.0
                    (width_coverage >= 0.20 or w >= 200) and  # Was 0.3
                    (absolute_height < 150 or (h / roi_height) < 0.3))

# Use absolute height thresholds to handle abnormal ROIs
if absolute_height < 20:  # Very thin
    thinness_score = 1.0
elif absolute_height < 50:  # Moderately thin
    thinness_score = 0.7
# ...
```
**Rationale**:
- Exclude very large components first (>10% of ROI)
- Use absolute pixel thresholds to handle cases where ROI is abnormally large
- Relaxed aspect ratio and width requirements to catch more valid midribs

### 5. Lamina Exclusion
```python
# Adaptive lamina threshold
lamina_threshold = max(50, mean_green_s * 0.7)
lamina_mask = ((s_channel > lamina_threshold) &
               (h_channel >= 35) & (h_channel <= 65))
not_lamina = ~lamina_mask
```
**Rationale**: Explicitly exclude high-saturation green pixels to prevent lamina selection.

## Results

### Test on 10 Images
- **Before**: Algorithm was selecting leaf lamina tissue as midrib
- **After**: 4/10 images now detect midrib successfully

### Successful Detections:
1. **2115**: 106,563 pixels detected
2. **2012**: 2,239 pixels detected
3. **1102**: 182,556 pixels detected
4. **1872**: 2,224 pixels detected

### Remaining Challenges:
1. **Large midrib masks**: Some detected regions (2115, 1102) are very large (>100K pixels), suggesting they may include some lamina tissue. The 10% ROI area threshold may need to be stricter.
2. **Failed detections**: 6/10 images still return 0 pixels due to:
   - Insufficient pixels passing all filters
   - Component shape requirements still too strict
   - Invalid ROI due to overlapping green strip detection
3. **ROI issues**: Some images have overlapping green strips, causing the ROI to expand to the full image, which breaks the spatial constraints.

## Recommendations for Further Improvement

### Short-term:
1. **Tighter size constraints**: Reduce max_midrib_area from 10% to 3-5% of ROI
2. **Post-filtering**: After selecting component, apply additional morphological operations to extract only the central ridge
3. **Skeleton extraction**: Use skeletonization to extract the midrib centerline from the detected region

### Medium-term:
1. **Fix green strip detection**: Improve the upstream green strip detection to avoid overlapping boxes
2. **Machine learning approach**: Consider training a U-Net or similar model for direct midrib segmentation
3. **Multi-scale detection**: Try detecting at multiple scales and combining results

### Long-term:
1. **Active learning**: Allow users to correct detections and retrain
2. **Ensemble methods**: Combine multiple detection strategies (color-based, edge-based, machine learning)

## Code Location
- Main algorithm: `src/segment_green_strips/detect_midrib.py` (detect_midrib_region function, lines 153-381)
- Diagnostic script: `src/segment_green_strips/analyze_midrib_colors.py`
- Test images: Listed in `src/segment_green_strips/test_images_list.txt`
- Results: `midrib_output_final/`

## Key Takeaways
1. **Brightness is the most reliable feature** for midrib detection (2-3x brighter than lamina)
2. **Color is variable** across images due to lighting conditions - use as secondary filter
3. **Shape constraints must be adaptive** to handle varying ROI sizes
4. **Size filtering is critical** to avoid selecting large lamina regions
5. **Diagnostic analysis is essential** - assumptions about color properties (like B channel) can be wrong
