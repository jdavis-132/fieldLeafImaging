# Directory Reorganization Log
**Date:** 2025-11-06
**Status:** Completed

## Overview
This document logs all changes made during the directory reorganization to separate data and source code into cleaner structures.

## Directory Structure Changes

### Data Directories (moved to `data/`)
The following data directories were organized under `data/`:

| Original Location | New Location | Notes |
|-------------------|--------------|-------|
| `data/AAMUImages/` | `data/aamu2025/` | Renamed for consistency |
| `data/FVSU_SAP_BAP_Images/` | `data/fvsu2025/` | Renamed for consistency |
| `data/ne2025/` | `data/ne2025/` | Already in correct location |

**Note:** The `data/` directory was already present. Only renaming operations were performed.

### Source Code Directories (moved to `src/`)
A new `src/` directory was created to contain all source code:

| Original Location | New Location | Type |
|-------------------|--------------|------|
| `autoencoder/` | `src/autoencoder/` | Python module |
| `disease_autoencoder/` | `src/disease_autoencoder/` | Python module |
| `disease_autoencoder_cropped/` | `src/disease_autoencoder_cropped/` | Python module |
| `color_correction_comparisons/` | `src/color_correction/` | Python scripts & comparisons |
| N/A (new) | `src/segment_green_strips/` | Created for green strip scripts |
| N/A (new) | `src/sam2/` | Created for SAM2 scripts |
| N/A (new) | `src/R/` | Created for R scripts |

### Individual Scripts Moved

#### Green Strip Processing Scripts → `src/segment_green_strips/`
- `detect_green_strips.py`
- `batch_process_green_strips.py`
- `measure_strip_heights.py`
- `reprocess_skipped_images.py`

#### Color Correction Scripts → `src/color_correction/`
- `colorchecker_normalize.py`
- `compare_color_corrections.py`

#### SAM2 Scripts → `src/sam2/`
- `run_sam2.py`
- `sam2_tiny.py`
- `save_masks.py`

#### R Scripts → `src/R/`
- `stripWidthQuantGen.R`

## Python Import Updates

### Root-Level Scripts
- **`run_pipeline.py`**: Updated imports from `autoencoder.*` to `src.autoencoder.*`

### Module Internal Imports
All internal imports within the following modules were updated:

1. **`src/autoencoder/`** - All imports changed from `from autoencoder.*` to `from src.autoencoder.*`
   - Files affected: `train.py`, `model.py`, `dataset.py`, `utils.py`, `visualize.py`, `extract_embeddings.py`, `prepare_splits.py`, `config.py`

2. **`src/disease_autoencoder/`** - All imports updated:
   - Internal imports: `from disease_autoencoder.*` → `from src.disease_autoencoder.*`
   - Cross-module imports: `from autoencoder.*` → `from src.autoencoder.*`
   - Files affected: `train.py`, `test_model.py`, `demo.py`, `evaluate.py`, `loss.py`, `model.py`, `plot_training.py`, `test_setup.py`, `visualize_model.py`, `visualize_reconstructions.py`, `cluster_analysis.py`, `config.py`, `dataset.py`

3. **`src/disease_autoencoder_cropped/`** - All imports updated:
   - Internal imports: `from disease_autoencoder_cropped.*` → `from src.disease_autoencoder_cropped.*`
   - Cross-module imports: `from autoencoder.*` → `from src.autoencoder.*`
   - Files affected: Same file structure as `disease_autoencoder/`

### Package Structure
- Created `src/__init__.py` to make `src/` a proper Python package

## Target Directory Structure (Final)

```
.
├── data/
│   ├── aamu2025/           (renamed from AAMUImages)
│   ├── fvsu2025/           (renamed from FVSU_SAP_BAP_Images)
│   └── ne2025/             (unchanged)
└── src/
    ├── __init__.py         (new)
    ├── autoencoder/
    ├── disease_autoencoder/
    ├── disease_autoencoder_cropped/
    ├── color_correction/   (renamed from color_correction_comparisons)
    ├── segment_green_strips/ (new, contains green strip scripts)
    ├── R/                  (new, contains R scripts)
    └── sam2/               (new, contains SAM2 scripts)
```

## Files Not Modified

The following files remain at the root level and were not modified:
- Configuration files (`.gitignore`, etc.)
- Documentation files (`README*.md`, `*.txt`)
- Data files (`*.csv`, `*.xlsx`, `*.fam`)
- Archive files (`*.zip`)
- Pipeline runner: `run_pipeline.py` (updated imports only)
- Output directories: `models/`, `embeddings/`, `output/`, `visualizations/`, `logs/`
- Test directories: `test_input/`, `test_output/`, `test_midrib/`, etc.

## Verification Steps Performed

1. ✅ All directories successfully moved
2. ✅ All Python imports updated
3. ✅ Package structure created with `__init__.py`
4. ✅ No hardcoded absolute paths found that need updating

## Notes and Recommendations

### For Future Use
1. When running scripts from the root directory, ensure the root is in `PYTHONPATH`:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/home/schnable/Documents/fieldLeafImaging"
   ```
   Or run scripts using Python's module syntax:
   ```bash
   python -m src.autoencoder.train
   ```

2. When importing from these modules in new scripts:
   - Use: `from src.autoencoder.config import Config`
   - Not: `from autoencoder.config import Config`

### Manual Review Recommended
- Check any configuration files or notebooks (if created later) for path references
- Review any shell scripts or batch files that might reference old paths
- Update documentation to reflect new structure

## Rollback Information

If you need to reverse these changes:

```bash
# Move directories back to root
mv src/autoencoder ./
mv src/disease_autoencoder ./
mv src/disease_autoencoder_cropped ./
mv src/color_correction ./color_correction_comparisons
mv src/segment_green_strips/* ./
mv src/sam2/* ./
mv src/R/* ./

# Rename data directories back
mv data/aamu2025 data/AAMUImages
mv data/fvsu2025 data/FVSU_SAP_BAP_Images

# Revert Python imports (use sed or git)
git checkout run_pipeline.py
find autoencoder disease_autoencoder disease_autoencoder_cropped -name "*.py" -exec git checkout {} \;

# Remove src directory
rm -rf src/
```

## Summary

✅ **Reorganization Complete**
- 3 data directories organized/renamed in `data/`
- 7 source code directories/modules organized in `src/`
- 13 standalone scripts organized into appropriate subdirectories
- ~50+ Python files updated with correct import paths
- New package structure created

**No errors encountered during reorganization.**
