# Disease Autoencoder - Testing and Visualization Guide

This guide covers the comprehensive testing and visualization tools for the disease autoencoder model.

## Overview

Three main scripts are provided for testing and visualizing the model:

1. **`test_model.py`** - Comprehensive unit testing of model components
2. **`visualize_model.py`** - Advanced visualizations of model behavior
3. **`demo.py`** - Interactive demo and report generation

---

## 1. Comprehensive Testing (`test_model.py`)

Tests all aspects of the model to ensure it works correctly.

### What It Tests

- ✅ Model architecture and parameter counts
- ✅ Forward pass (reconstruction and encoding)
- ✅ Backward pass and gradient flow
- ✅ Loss function (disease-weighted and simple MSE)
- ✅ Dataset loading and preprocessing
- ✅ Memory usage
- ✅ Output quality metrics (MSE, PSNR, correlation)
- ✅ Model robustness to edge cases
- ✅ Save and load checkpoint functionality

### Usage

```bash
# Run all tests
python -m disease_autoencoder.test_model
```

### Example Output

```
==============================================================
Disease Autoencoder - Comprehensive Testing
==============================================================

==============================================================
Test 1: Model Architecture
==============================================================

✓ PASS Model creation
     2,847,235 total parameters
✓ PASS Trainable parameters
     2,847,235 trainable
✓ PASS Has encoder
✓ PASS Has decoder
✓ PASS Has attention
✓ PASS Has embedding extractor
✓ PASS Model to device
     Device: mps

...

==============================================================
Test Summary
==============================================================

✓ Architecture
✓ Forward Pass
✓ Backward Pass
✓ Loss Function
✓ Dataset Loading
✓ Memory Usage
✓ Output Metrics
✓ Model Robustness
✓ Save/Load Model

Results: 9/9 tests passed

🎉 All tests passed!

The model is ready for training and evaluation.
```

### When to Use

- **Before training** - Verify setup is correct
- **After changes** - Ensure modifications didn't break anything
- **Debugging** - Identify issues with model components
- **CI/CD** - Automated testing in deployment pipelines

---

## 2. Advanced Visualization (`visualize_model.py`)

Creates detailed visualizations of model internals and behavior.

### What It Visualizes

1. **Attention Maps** - Where the model focuses in the bottleneck
2. **Disease Weight Heatmaps** - Which regions the model considers diseased
3. **LAB Channel Reconstruction** - Per-channel analysis (L*, a*, b*)
4. **Error Heatmaps** - Spatial distribution of reconstruction errors
5. **Latent Space Analysis** - PCA and t-SNE of embeddings
6. **Feature Maps** - Intermediate encoder activations

### Usage

```bash
# Generate all visualizations
python -m disease_autoencoder.visualize_model
```

### Requirements

- Trained model checkpoint at `disease_autoencoder/models/checkpoint_best.pth`
- Trained data splits at `disease_autoencoder/logs/image_splits.json`
- LAB statistics at `disease_autoencoder/logs/lab_statistics.json`

### Output

All visualizations are saved to `disease_autoencoder/visualizations/advanced/`:

- `attention_maps.png` - Attention mechanism visualization
- `disease_weights.png` - Disease weight heatmaps
- `lab_channels.png` - Per-channel LAB reconstruction
- `error_heatmaps.png` - Reconstruction error analysis
- `latent_space_pca_tsne.png` - Dimensionality reduction plots
- `feature_maps.png` - Encoder feature activations

### Example Visualizations

#### Attention Maps
Shows where the model focuses attention in the bottleneck. Useful for understanding what the model considers important.

#### Disease Weights
Highlights regions the model identifies as diseased based on color deviation from healthy tissue in LAB space. Red = more diseased, green = healthy.

#### LAB Channel Reconstruction
Compares original and reconstructed images in each LAB channel:
- **L* (lightness)** - Brightness information
- **a* (green-red)** - Color shift from green to red
- **b* (blue-yellow)** - Color shift from blue to yellow

#### Error Heatmaps
Shows per-channel reconstruction errors. Hot colors = higher error. Useful for identifying where the model struggles.

### When to Use

- **After training** - Understand what the model learned
- **Model analysis** - Debug reconstruction quality issues
- **Paper/presentations** - Generate figures
- **Feature engineering** - Identify important spatial regions

---

## 3. Interactive Demo (`demo.py`)

Test the model on samples and generate comprehensive reports.

### Modes

#### Random Samples Mode (Default)

Test on random samples from the test set.

```bash
# Test on 5 random samples (default)
python -m disease_autoencoder.demo

# Test on 10 random samples
python -m disease_autoencoder.demo --num_samples 10

# Use different random seed
python -m disease_autoencoder.demo --num_samples 5 --seed 123
```

**Output:**
- Individual visualizations for each sample
- Reconstruction metrics (MSE, MAE, PSNR)
- Per-channel MAE
- Summary statistics

**Saved to:** `disease_autoencoder/visualizations/demo_samples/`

#### Comprehensive Report Mode

Generate detailed statistics and plots for all test samples.

```bash
python -m disease_autoencoder.demo --report
```

**Output:**
- Overall metrics (mean, std)
- Per-genotype statistics (top 10)
- Distribution plots (MSE, PSNR)
- Per-channel error boxplots
- Per-device comparison
- JSON report with detailed data

**Saved to:** `disease_autoencoder/visualizations/report/`

### Understanding the Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **MSE** | Mean Squared Error (lower is better) | < 0.01 |
| **MAE** | Mean Absolute Error (lower is better) | < 0.05 |
| **PSNR** | Peak Signal-to-Noise Ratio (higher is better) | > 25 dB |
| **L* MAE** | Lightness reconstruction error | < 0.05 |
| **a* MAE** | Green-red reconstruction error | < 0.05 |
| **b* MAE** | Blue-yellow reconstruction error | < 0.05 |

### Example Output

```
==============================================================
Testing on 5 Random Samples
==============================================================

Sample 1/5 (Index: 42)
----------------------------------------
Genotype: PI123456_SubLine_A
Device: 1, Strip: 5
Metrics: MSE=0.003421, PSNR=24.66 dB
✓ Saved visualization to .../sample_42_reconstruction.png

...

==============================================================
Summary Statistics
==============================================================

Average MSE:  0.003856
Average MAE:  0.043212
Average PSNR: 24.14 dB

Per-channel MAE:
  L*: 0.0421
  a*: 0.0389
  b*: 0.0476

✓ Visualizations saved to: disease_autoencoder/visualizations/demo_samples
```

### When to Use

- **Quick testing** - Fast check of model performance
- **Presentations** - Generate example reconstructions
- **Debugging** - Analyze specific failure cases
- **Benchmarking** - Compare models with comprehensive reports

---

## Complete Workflow

### 1. Initial Setup and Testing

```bash
# Verify everything is installed and working
python -m disease_autoencoder.test_setup

# Run comprehensive tests
python -m disease_autoencoder.test_model
```

### 2. Training

```bash
# Train the model
python -m disease_autoencoder.train
```

### 3. Evaluation

```bash
# Extract embeddings and create basic visualizations
python -m disease_autoencoder.evaluate
```

### 4. Advanced Analysis

```bash
# Generate advanced visualizations
python -m disease_autoencoder.visualize_model

# Test on random samples
python -m disease_autoencoder.demo --num_samples 10

# Generate comprehensive report
python -m disease_autoencoder.demo --report
```

### 5. Clustering

```bash
# Perform clustering analysis
python -m disease_autoencoder.cluster_analysis --method kmeans --n_clusters 5
```

---

## Troubleshooting

### Issue: "No trained model found"

**Solution:** Train the model first:
```bash
python -m disease_autoencoder.train
```

### Issue: "Embeddings not found" (in visualize_model.py)

**Solution:** Run evaluation first:
```bash
python -m disease_autoencoder.evaluate
```

### Issue: Out of Memory

**Solution:** Reduce batch size in `config.py`:
```python
self.batch_size = 8  # or smaller
```

### Issue: Tests failing

**Solution:**
1. Check if data and masks are available
2. Verify CSV file exists
3. Run `test_setup.py` for detailed diagnostics
4. Check error messages for specific issues

---

## File Outputs Summary

| Script | Output Directory | Files Generated |
|--------|------------------|-----------------|
| `test_model.py` | Terminal only | None (runs tests) |
| `visualize_model.py` | `visualizations/advanced/` | 6 PNG files |
| `demo.py` (samples) | `visualizations/demo_samples/` | N PNG files (N = num_samples) |
| `demo.py` (report) | `visualizations/report/` | 2 files (PNG + JSON) |
| `evaluate.py` | `visualizations/` | 2 PNG files |
| `evaluate.py` | `embeddings/` | 4 files (NPY + JSON) |

---

## Tips and Best Practices

### For Testing
- Run `test_model.py` before committing code changes
- Use it to benchmark different model configurations
- Monitor memory usage to optimize batch sizes

### For Visualization
- Generate visualizations after each training run
- Compare attention maps across epochs to see learning progression
- Use disease weight heatmaps to validate the weighting strategy

### For Demo
- Use `--report` mode to create benchmark datasets
- Test on multiple random seeds to ensure consistency
- Save demo outputs for presentations and papers

---

## Advanced Usage

### Custom Visualization Scripts

You can import functions from the visualization scripts:

```python
from disease_autoencoder.visualize_model import visualize_attention_maps
from disease_autoencoder.config import DiseaseConfig

config = DiseaseConfig()
# ... load model and dataset ...
visualize_attention_maps(model, dataset, config, save_dir='custom_output/')
```

### Programmatic Testing

```python
from disease_autoencoder.test_model import test_model_architecture
from disease_autoencoder.config import DiseaseConfig

config = DiseaseConfig()
passed, model = test_model_architecture(config)

if passed:
    print("Model is ready!")
```

### Batch Processing

```python
# Test multiple checkpoints
checkpoints = ['checkpoint_epoch_10.pth', 'checkpoint_epoch_20.pth', 'checkpoint_best.pth']

for ckpt in checkpoints:
    model = load_model(config, ckpt)
    # Run analysis...
```

---

## Further Reading

- **Main documentation**: `disease_autoencoder/README.md`
- **Quick start**: `disease_autoencoder/QUICKSTART.md`
- **Project summary**: `DISEASE_AUTOENCODER_SUMMARY.md`
- **Configuration**: `disease_autoencoder/config.py`

---

## Questions?

If you encounter issues or have questions:

1. Check this documentation
2. Run `test_setup.py` for diagnostics
3. Review error messages carefully
4. Check that all required files exist (model, splits, stats)

Happy testing and visualizing! 🎉
