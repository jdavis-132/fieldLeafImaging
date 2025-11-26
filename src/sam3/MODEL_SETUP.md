# SAM3 Model Setup

This directory contains the configuration files for the SAM3 (Segment Anything Model 3) model used for leaf segmentation.

## Large Model Files

The actual model weights (`model.safetensors` and `sam3.pt`) are **not** tracked in git due to their large size (~6.4GB total). They are stored in `.gitignore`.

## Setting Up on a New Machine

When cloning this repository to a new machine, you need to set up the model files. Follow these steps:

### 1. Install Required Dependencies

Make sure you have the correct version of transformers installed:

```bash
pip install git+https://github.com/huggingface/transformers.git
```

### 2. Download the SAM3 Model

Download the SAM3 model from HuggingFace (this will cache it in `~/.cache/huggingface/`):

```bash
python -c "from transformers import Sam3Model, Sam3Processor; Sam3Model.from_pretrained('facebook/sam3'); Sam3Processor.from_pretrained('facebook/sam3')"
```

### 3. Run the Setup Script

Copy the model files from the cache to this directory:

```bash
cd src/sam3
./setup_model.sh
```

This will copy:
- `model.safetensors` (3.2GB)
- `sam3.pt` (3.2GB)

### 4. Verify Setup

Check that the model files exist:

```bash
ls -lh model.safetensors sam3.pt
```

You should see both files with sizes around 3.2GB each.

## Files in This Directory

- **Tracked in Git:**
  - `config.json` - Model configuration
  - `processor_config.json` - Processor configuration
  - `tokenizer*.json`, `vocab.json`, `merges.txt` - Tokenizer files
  - `LICENSE`, `README.md` - Documentation
  - `setup_model.sh` - Setup script
  - `*.py` - Python code for using the model

- **Not Tracked (in .gitignore):**
  - `model.safetensors` - Main model weights
  - `sam3.pt` - Alternative model format
  - `__pycache__/` - Python cache files

## Troubleshooting

If you get errors about missing model files:
1. Make sure you ran `./setup_model.sh`
2. Check that the HuggingFace cache exists at `~/.cache/huggingface/hub/models--facebook--sam3/`
3. Re-download the model using the command in step 2 above
