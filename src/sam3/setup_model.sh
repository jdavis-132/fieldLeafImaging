#!/bin/bash
# Setup script to copy SAM3 model files from HuggingFace cache
# Run this script once on each new machine to set up the model files

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${HOME}/.cache/huggingface/hub/models--facebook--sam3/blobs"

echo "Setting up SAM3 model files..."

# Check if cache directory exists
if [ ! -d "$CACHE_DIR" ]; then
    echo "Error: HuggingFace cache directory not found at $CACHE_DIR"
    echo "Please download the SAM3 model first using:"
    echo "  python -c \"from transformers import Sam3Model; Sam3Model.from_pretrained('facebook/sam3')\""
    exit 1
fi

# Copy model.safetensors
if [ ! -f "$SCRIPT_DIR/model.safetensors" ]; then
    echo "Copying model.safetensors..."
    cp "$CACHE_DIR/6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a" "$SCRIPT_DIR/model.safetensors"
    echo "  ✓ model.safetensors copied"
else
    echo "  ✓ model.safetensors already exists"
fi

# Copy sam3.pt
if [ ! -f "$SCRIPT_DIR/sam3.pt" ]; then
    echo "Copying sam3.pt..."
    cp "$CACHE_DIR/9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e" "$SCRIPT_DIR/sam3.pt"
    echo "  ✓ sam3.pt copied"
else
    echo "  ✓ sam3.pt already exists"
fi

echo ""
echo "SAM3 model setup complete!"
echo "Model files are ready to use."
