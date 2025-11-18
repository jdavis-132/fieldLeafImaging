# Disease Autoencoder - Quick Reference Card

## 🚀 Quick Start Commands

```bash
# 1. Test setup
python -m disease_autoencoder.test_setup

# 2. Run comprehensive tests
python -m disease_autoencoder.test_model

# 3. Train model
python -m disease_autoencoder.train

# 4. Basic evaluation
python -m disease_autoencoder.evaluate

# 5. Advanced visualizations
python -m disease_autoencoder.visualize_model

# 6. Interactive demo
python -m disease_autoencoder.demo --num_samples 10

# 7. Generate report
python -m disease_autoencoder.demo --report

# 8. Cluster analysis
python -m disease_autoencoder.cluster_analysis --method kmeans --n_clusters 5
```

---

## 📊 Available Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `test_setup.py` | Verify installation | Before everything |
| `test_model.py` | Comprehensive testing | Before training, after changes |
| `train.py` | Train the model | After setup |
| `evaluate.py` | Extract embeddings | After training |
| `visualize_model.py` | Advanced visualizations | After training |
| `demo.py` | Interactive testing | Anytime after training |
| `cluster_analysis.py` | Clustering | After evaluation |

---

## 🎨 Visualization Outputs

### `evaluate.py` generates:
- `visualizations/umap_latent_space.png`
- `visualizations/reconstructions.png`

### `visualize_model.py` generates:
- `visualizations/advanced/attention_maps.png`
- `visualizations/advanced/disease_weights.png`
- `visualizations/advanced/lab_channels.png`
- `visualizations/advanced/error_heatmaps.png`
- `visualizations/advanced/latent_space_pca_tsne.png`
- `visualizations/advanced/feature_maps.png`

### `demo.py` generates:
- `visualizations/demo_samples/sample_*_reconstruction.png` (multiple)
- `visualizations/report/metrics_report.png`
- `visualizations/report/metrics_report.json`

---

## 📈 Understanding Metrics

| Metric | Good Value | Interpretation |
|--------|------------|----------------|
| MSE | < 0.01 | Lower is better, squared error |
| MAE | < 0.05 | Lower is better, absolute error |
| PSNR | > 25 dB | Higher is better, signal quality |
| L* MAE | < 0.05 | Lightness reconstruction |
| a* MAE | < 0.05 | Green-red color accuracy |
| b* MAE | < 0.05 | Blue-yellow color accuracy |

---

## 🔧 Common Issues

### "No trained model found"
```bash
python -m disease_autoencoder.train
```

### "Embeddings not found"
```bash
python -m disease_autoencoder.evaluate
```

### Out of memory
Edit `config.py`:
```python
self.batch_size = 8  # reduce from 16
self.image_size = 192  # reduce from 224
```

### Tests failing
```bash
python -m disease_autoencoder.test_setup  # diagnose issues
```

---

## 📁 Key Files

```
disease_autoencoder/
├── config.py              # ⚙️  Edit this for settings
├── train.py               # 🎓 Training script
├── evaluate.py            # 📊 Basic evaluation
├── test_model.py          # ✅ NEW: Comprehensive tests
├── visualize_model.py     # 🎨 NEW: Advanced visualizations
├── demo.py                # 🚀 NEW: Interactive demo
├── models/
│   └── checkpoint_best.pth       # 💾 Trained model
├── embeddings/
│   ├── latent_embeddings.npy     # 🧬 Embeddings (N x 256)
│   └── embeddings_metadata.json  # 📋 Image info
├── visualizations/
│   ├── advanced/                 # 🎨 Advanced viz
│   ├── demo_samples/             # 🖼️  Demo outputs
│   └── report/                   # 📊 Report outputs
└── logs/
    ├── image_splits.json         # 📂 Train/val/test splits
    └── lab_statistics.json       # 📐 LAB normalization
```

---

## 🎯 Typical Workflow

```
1. test_setup.py        ✅ Verify installation
         ↓
2. test_model.py        ✅ Run tests (optional but recommended)
         ↓
3. train.py             🎓 Train model (2-4 hours on GPU)
         ↓
4. evaluate.py          📊 Extract embeddings + basic viz
         ↓
5. visualize_model.py   🎨 Advanced visualizations
         ↓
6. demo.py              🚀 Test samples + reports
         ↓
7. cluster_analysis.py  🔬 Cluster embeddings
```

---

## 💡 Pro Tips

- **Before training**: Run `test_model.py` to verify setup
- **After training**: Run all visualization scripts
- **For presentations**: Use `demo.py` to generate examples
- **For papers**: Use `--report` mode for statistics
- **Multiple runs**: Change `random_seed` in `config.py`

---

## 🎓 Command Examples

### Basic Testing
```bash
# Quick test (no output files)
python -m disease_autoencoder.test_model
```

### Generate All Visualizations
```bash
# After training, run these in order:
python -m disease_autoencoder.evaluate
python -m disease_autoencoder.visualize_model
python -m disease_autoencoder.demo --num_samples 10
python -m disease_autoencoder.demo --report
```

### Clustering
```bash
# K-means
python -m disease_autoencoder.cluster_analysis --method kmeans --n_clusters 5

# DBSCAN
python -m disease_autoencoder.cluster_analysis --method dbscan --eps 0.5

# Hierarchical
python -m disease_autoencoder.cluster_analysis --method hierarchical --n_clusters 5

# Gaussian Mixture
python -m disease_autoencoder.cluster_analysis --method gmm --n_clusters 5
```

---

## 📚 Documentation

- **Full guide**: `TESTING_AND_VISUALIZATION.md`
- **Main README**: `README.md`
- **Quick start**: `QUICKSTART.md`
- **Project summary**: `../DISEASE_AUTOENCODER_SUMMARY.md`

---

## 🆘 Getting Help

1. Check error messages
2. Run `test_setup.py` for diagnostics
3. Review documentation
4. Check that required files exist

---

**Created**: 2025-10-29
**Version**: 1.0
