# ColorChecker Color Normalization

This tool normalizes colors across multiple images using a Calibrite ColorChecker Classic Nano card for accurate cross-image comparisons.

## Overview

The script `colorchecker_normalize.py` performs the following steps:
1. **Rotates** the image 90° clockwise
2. **Detects** the ColorChecker card in bottom-right corner of rotated image (using fixed position)
3. **Extracts** color values from all 24 patches
4. **Computes** a color correction transform by comparing measured vs reference values
5. **Applies** the correction to the rotated image
6. **Saves** the rotated, normalized images maintaining the original directory structure

**IMPORTANT**: Output images are rotated 90° clockwise from the originals.

## Requirements

All required packages are already in `requirements.txt`:
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- pillow >= 9.5.0

## Usage

### Basic Usage

Process all images in a directory:

```bash
python colorchecker_normalize.py --input data/ne2025 --output data/ne2025_normalized
```

### Command Line Options

```bash
python colorchecker_normalize.py [OPTIONS]

Required:
  --input, -i PATH     Input directory containing images
  --output, -o PATH    Output directory for normalized images

Optional:
  --debug              Save debug visualizations showing detected card location
  --region REGION      Region to search for card (default: top-right)
                       Choices: top-left, top-right, full
  --method METHOD      Color correction method (default: linear)
                       Choices: linear, polynomial
```

### Examples

**Process specific device folder:**
```bash
python colorchecker_normalize.py -i data/ne2025/device1 -o output/device1_normalized
```

**Process with debug visualizations:**
```bash
python colorchecker_normalize.py -i data/ne2025 -o output/normalized --debug
```

**Use polynomial correction (experimental):**
```bash
python colorchecker_normalize.py -i data/ne2025 -o output/normalized --method polynomial
```

## Output

The script will:
- Create **rotated (90° CW)** and normalized images in the output directory
- Preserve the original directory structure
- Generate a summary report showing:
  - Number of successfully processed images
  - Number of failed images
  - List of failed images (if any)
- If `--debug` is used, save rotated images with green boxes showing detected ColorChecker location

**Note**: All output images are rotated 90° clockwise from the original orientation.

## ColorChecker Configuration

The script is pre-configured for your setup:
- **Card type**: Calibrite ColorChecker Classic Nano
- **Processing**: Images are rotated 90° clockwise before processing
- **Card location**: Bottom-right corner of rotated image
- **Card size**: 768 × 493 pixels (width × height in rotated image)
- **Position**: 516 pixels from right edge, 291 pixels from bottom
- **Output**: Rotated (90° CW), color-normalized images

### Adjusting Card Position

If your ColorChecker card position varies, edit the `ColorCheckerDetector` initialization in the script:

```python
detector = ColorCheckerDetector(
    use_fixed_position=True,
    card_width=768,           # Width after rotation (adjust if needed)
    card_height=493,          # Height after rotation (adjust if needed)
    offset_from_right=516,    # Distance from right edge
    offset_from_top=291       # Distance from bottom edge
)
```

Or set `use_fixed_position=False` to use automatic detection (less reliable).

## Color Correction Methods

### Linear (Recommended)
- **Pros**: Stable, fast, works well for varying lighting
- **Cons**: Limited to linear color transformations
- **Use when**: Default choice for most applications

### Polynomial (Experimental)
- **Pros**: Can model non-linear color relationships
- **Cons**: Can be unstable with extreme pixel values
- **Use when**: Linear correction is insufficient (test carefully)

## Technical Details

### Reference Values
Uses post-2014 ColorChecker Classic reference values (sRGB D65 color space).

### Color Correction Algorithm
1. Extract mean RGB from center 50% of each patch (24 patches)
2. Fit transform: measured RGB → reference RGB
3. Apply transform to all pixels in image
4. Clip values to valid range [0, 255]

**Linear method:**
- Fits a 3×4 affine transform matrix
- Formulation: `RGB_corrected = RGB_measured × M + b`

**Polynomial method:**
- Fits degree-2 polynomial with cross terms
- Features: [1, R, G, B, R², G², B², RG, RB, GB]

### Typical Accuracy
- Average error: ~45-55 RGB units (RMSE)
- Best performance on neutral colors (grays)
- Some variability on saturated colors due to lighting/camera limitations

## Troubleshooting

### "Could not detect ColorChecker"
- Verify card is in the correct location (top-right corner)
- Check that card dimensions match (493×768 pixels)
- Adjust `offset_from_right` and `offset_from_top` parameters
- Use `--debug` to visualize detection

### Colors look wrong after normalization
- Try switching between `--method linear` and `--method polynomial`
- Verify reference values match your ColorChecker version
- Check that images all have the same card position

### Processing is slow
- Processing speed: ~1-3 seconds per 12MP image
- Use parallel processing for large batches (see below)

## Batch Processing Tips

For large datasets, you can process subdirectories in parallel:

```bash
# Process each device folder separately (can run in parallel)
for device in data/ne2025/device*; do
    device_name=$(basename $device)
    python colorchecker_normalize.py \
        -i "$device" \
        -o "output/$device_name" &
done
wait  # Wait for all parallel jobs to complete
```

## Expected Results

After normalization:
- Colors should be consistent across images from the same scene under different lighting
- ColorChecker patches should closely match reference values
- Images can be directly compared for color-based analysis
- Relative color relationships within each image are preserved

## Limitations

1. **Fixed card position**: Card must be in the same location across all images
2. **Lighting assumptions**: Works best with consistent lighting type (same color temperature)
3. **Linear limitations**: Linear method can't correct non-linear camera responses perfectly
4. **Out-of-gamut colors**: Some ColorChecker patches (e.g., cyan) may be outside sRGB gamut

## Citation

If you use this tool in published research, please cite:
- ColorChecker reference: X-Rite/Calibrite ColorChecker Classic specifications
- OpenCV library: Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.
