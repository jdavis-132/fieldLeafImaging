# Comprehensive Model Comparison Framework

## Overview

This framework provides a comprehensive comparison of three trained PyTorch UNet autoencoder models for sorghum leaf disease (anthracnose) analysis:

1. **disease_autoencoder**: Uses weighted MSE loss (LAB-based weighting) with SAM2.1 leaf masks applied during training
2. **disease_autoencoder_cropped**: Uses weighted MSE loss (LAB-based weighting) on pre-cropped images
3. **autoencoder_no_weighting**: Uses standard MSE loss on pre-cropped images

## Features

- **Multi-dimensional Evaluation**: MSE, weighted MSE, masked MSE, SSIM, PSNR
- **Latent Space Analysis**: t-SNE and UMAP dimensionality reduction
- **Comprehensive Visualizations**:
  - Training/validation loss curves
  - Reconstruction quality grids with error maps
  - Latent space scatter plots
  - Error distribution histograms
- **Detailed Reports**: CSV summaries and JSON detailed metrics
- **Modular Design**: Easy to extend for additional models or metrics

## Installation

### Required Dependencies

```bash
pip install torch torchvision
pip install numpy opencv-python
pip install matplotlib seaborn
pip install scikit-learn scikit-image
pip install pandas tqdm
pip install umap-learn  # Optional, for UMAP analysis
```

## Usage

### Quick Start

Run the comparison on a sample of validation data:

```bash
cd /home/schnable/Documents/fieldLeafImaging/src
python comprehensive_model_comparison.py --num-samples 50
```

### Full Evaluation

Run on the complete validation set (814 images):

```bash
python comprehensive_model_comparison.py
```

### Command Line Options

```
--output-dir      Output directory for results (default: comparison_results)
--num-samples     Number of validation samples to evaluate (default: all)
--skip-latent     Skip latent space analysis (faster)
```

### Examples

```bash
# Run on 100 samples with custom output directory
python comprehensive_model_comparison.py --num-samples 100 --output-dir my_comparison

# Skip latent space analysis for faster execution
python comprehensive_model_comparison.py --skip-latent
```

## Output Structure

After running, the following directory structure is created:

```
comparison_results/
├── summary_table.csv              # Quick summary of key metrics
├── detailed_metrics.json          # Complete metrics and configurations
├── visualizations/
│   ├── loss_curves.png           # Training/validation loss over epochs
│   ├── reconstruction_grid.png   # Sample reconstructions and error maps
│   ├── latent_space_comparison_t-sne.png  # t-SNE latent space plots
│   ├── latent_space_comparison_umap.png   # UMAP latent space plots
│   ├── metrics_comparison.png    # Bar charts of key metrics
│   └── error_distributions.png   # Histograms of reconstruction errors
└── model_specific/
    ├── disease_autoencoder/
    ├── disease_autoencoder_cropped/
    └── autoencoder_no_weighting/
```

## Metrics Explained

### Reconstruction Quality Metrics

- **MSE (Mean Squared Error)**: Average squared difference between prediction and target
  - Computed on all pixels (unmasked)
  - Lower is better

- **Masked MSE**: MSE computed only on leaf pixels (masked region)
  - Fair comparison accounting for background
  - Lower is better

- **Weighted MSE**: MSE with disease-based pixel weighting
  - Weights: 1.0 (healthy) to 2.0 (diseased)
  - Based on LAB color deviation from healthy tissue
  - Only applicable for weighted models

- **SSIM (Structural Similarity Index)**: Perceptual quality metric (0-1)
  - Accounts for luminance, contrast, and structure
  - Higher is better (1.0 = perfect)

- **PSNR (Peak Signal-to-Noise Ratio)**: Quality metric in dB
  - Logarithmic scale, sensitive to large errors
  - Higher is better (typically 20-40 dB)

### Performance Metrics

- **Inference Time**: Average time per image (seconds)
- **Model Size**: Total model parameters and memory footprint (MB)
- **Latent Dimension**: Size of learned embedding space (256 for all models)

## Understanding the Results

### Training Loss Curves

- **Convergence**: Check if validation loss plateaus
- **Overfitting**: Watch for train/val divergence
- **Comparison**: Lower weighted models may have higher raw loss values (expected)

### Reconstruction Quality

- **Visual Inspection**: Look at reconstruction grid
- **Error Maps**: Hotter colors indicate larger errors
- **Model Differences**:
  - `disease_autoencoder`: May preserve disease regions better
  - `disease_autoencoder_cropped`: Balance of disease and context
  - `autoencoder_no_weighting`: Most balanced overall reconstruction

### Latent Space Analysis

- **Clustering**: Good separation suggests meaningful features
- **Genotype Grouping**: Related genotypes should cluster
- **Comparison**: Disease-weighted models may separate disease states better

### Which Model to Use?

**Use `disease_autoencoder` when:**
- Disease detection is the primary goal
- Background pixels should be ignored
- Maximum focus on diseased regions needed

**Use `disease_autoencoder_cropped` when:**
- Disease emphasis with some context
- Background reconstruction matters
- More stable training desired

**Use `autoencoder_no_weighting` when:**
- Balanced representation needed
- Healthy tissue variation is important
- Unbiased embeddings for clustering
- Comparing healthy vs diseased equally

## Technical Details

### LAB Color Space

All models use LAB color space:
- **L***: Lightness (0-100, stored as 0-255 in OpenCV format)
- **a***: Green (-) to Red (+) axis
- **b***: Blue (-) to Yellow (+) axis

### Disease Weighting

Weighted models compute pixel weights based on deviation from healthy tissue:
- Healthy tissue: a* ≈ -30 (green), b* ≈ 30 (yellow-green)
- Diseased tissue: higher a* (reddish), varied b*
- Weight range: 1.0 (healthy) to 2.0 (diseased)

### Data Preprocessing Differences

| Stage | disease_autoencoder | disease_autoencoder_cropped | autoencoder_no_weighting |
|-------|---------------------|------------------------------|--------------------------|
| Crop | Bounding box + padding | Bounding box + padding | Bounding box + padding |
| Resize | 224x224 | 224x224 | 224x224 |
| LAB conversion | Yes | Yes | Yes |
| Normalization | Masked pixels only | All pixels | All pixels |
| **Background pixels** | **Set to 0** | **Preserved** | **Preserved** |

## Troubleshooting

### Memory Issues

If you encounter out-of-memory errors:

```bash
# Reduce batch size (edit comprehensive_model_comparison.py)
# Line ~70: batch_size: int = 8  # Default is 16

# Or evaluate on fewer samples
python comprehensive_model_comparison.py --num-samples 100
```

### Missing Checkpoints

Error: `Checkpoint not found`

**Solution**: Ensure all three models have been trained and have `checkpoint_best.pth` files:
```bash
ls src/disease_autoencoder/models/checkpoint_best.pth
ls src/disease_autoencoder_cropped/models/checkpoint_best.pth
ls src/autoencoder_no_weighting/models/checkpoint_best.pth
```

### UMAP Import Error

Error: `No module named 'umap'`

**Solution**: Install umap-learn:
```bash
pip install umap-learn
```

Or skip UMAP analysis:
```bash
python comprehensive_model_comparison.py --skip-latent
```

### Image Loading Errors

If you see multiple "Error loading image" messages:

**Check**:
1. Image paths in `logs/image_splits.json` are correct
2. Mask files exist at specified paths
3. File permissions allow reading

## Extending the Framework

### Adding a New Model

1. Create a new `ModelConfig` in `run_comprehensive_comparison()`:

```python
ModelConfig(
    name="my_new_model",
    directory=base_dir / "my_new_model",
    description="Description of the new model",
    uses_masks=False,  # Does it mask background?
    uses_weighting=False  # Does it use disease weighting?
)
```

2. Ensure the model has:
   - `config.py` with configuration class
   - `model.py` with `create_model()` function
   - `models/checkpoint_best.pth` checkpoint file
   - `logs/lab_statistics.json` normalization stats
   - `logs/image_splits.json` data splits

### Adding New Metrics

Extend the `LossCalculator` class in `comprehensive_model_comparison.py`:

```python
@staticmethod
def compute_custom_metric(pred: np.ndarray, target: np.ndarray) -> float:
    """Your custom metric implementation."""
    # Compute metric
    return metric_value
```

Then add to the evaluation loop in `_evaluate_single_model()`.

## Citation

If you use this comparison framework in your research, please cite:

```
Schnable Lab - Sorghum Leaf Disease Analysis
University of Nebraska-Lincoln
2025
```

## Contact

For questions or issues:
- Check existing model training logs
- Review error messages carefully
- Ensure all dependencies are installed

## License

Research use only.
