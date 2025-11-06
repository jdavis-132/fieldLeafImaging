# GPU Setup Instructions for SAM 2.1

## Current Status
- ✅ NVIDIA GeForce RTX 3060 GPU detected
- ✅ NVIDIA drivers (version 470) installed
- ❌ Driver version 470 (CUDA 11.4) too old for PyTorch 2.8 (requires CUDA 12.8)
- ❌ CUDA not available to PyTorch

## Root Cause
Your PyTorch was installed with CUDA 12.8 support, but your NVIDIA driver (470) only supports CUDA 11.4. There are two solutions:

### Solution A: Update NVIDIA Drivers (Recommended)
### Solution B: Downgrade PyTorch to Match Your Drivers

## To Enable GPU Support

### Solution A: Update NVIDIA Drivers (Recommended)

#### Step 1: Update to newer NVIDIA drivers
```bash
# Remove old drivers
sudo apt purge nvidia-driver-470

# Add NVIDIA driver repository
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install newer drivers (525+ supports CUDA 12)
sudo apt install nvidia-driver-525
sudo reboot
```

#### Step 2: Verify after reboot
```bash
nvidia-smi  # Should show driver version 525+
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Solution B: Downgrade PyTorch (Easier, but less optimal)

If you prefer not to update drivers, downgrade PyTorch to match CUDA 11.4:

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision

# Install PyTorch with CUDA 11.8 (compatible with driver 470)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Load NVIDIA kernel modules
sudo modprobe nvidia

# Test
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Quick Test (Choose one solution above first)
```bash
# Load kernel modules if needed
sudo modprobe nvidia

# Test SAM 2.1 with GPU
python3 -c "from sam2_tiny import SAM2TinyPredictor; predictor = SAM2TinyPredictor()"
```

### Step 4: Test GPU Support
After loading the NVIDIA modules, test with:

```bash
# Test basic NVIDIA functionality
nvidia-smi

# Test PyTorch CUDA support
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Test SAM 2.1 with GPU
python3 -c "from sam2_tiny import SAM2TinyPredictor; predictor = SAM2TinyPredictor()"
```

### Step 5: Run SAM 2.1 with GPU
Once GPU is working, you can force GPU usage:

```bash
# Force GPU usage
python3 run_sam2.py -i "data/ne2025/device1/1201_LeafPhotoA_2025-09-08 10_44_12.793-05_00.jpg" -m auto --device cuda

# Or let it auto-detect (default)
python3 run_sam2.py -i "data/ne2025/device1/1201_LeafPhotoA_2025-09-08 10_44_12.793-05_00.jpg" -m auto
```

## Expected Performance Improvements

### RTX 3060 Specifications:
- CUDA Cores: 3584
- Memory: 12GB GDDR6
- Memory Bandwidth: 360 GB/s

### Performance Comparison (estimated):
- **CPU (current)**: ~30-60 seconds per image
- **RTX 3060**: ~2-5 seconds per image
- **Speed improvement**: ~10-20x faster

## Troubleshooting

### Issue: "nvidia-smi" shows no output
- Run: `sudo modprobe nvidia`
- Check: `lsmod | grep nvidia`

### Issue: CUDA out of memory
- Reduce batch size or image resolution
- Use `torch.cuda.empty_cache()` between runs

### Issue: Driver version mismatch
- Update drivers: `sudo apt update && sudo apt upgrade`
- Reboot system

## Alternative: Use CPU with Optimizations
If GPU setup doesn't work, you can still optimize CPU performance:

```bash
# Set CPU thread count
export OMP_NUM_THREADS=$(nproc)

# Use optimized BLAS libraries
pip install intel-pytorch-extension
```

## Verify Installation
Run this test script to verify everything is working:

```python
python3 -c "
from sam2_tiny import SAM2TinyPredictor
import time

# Test with auto device selection
start = time.time()
predictor = SAM2TinyPredictor()
load_time = time.time() - start

print(f'Model load time: {load_time:.2f}s')
print(f'Using device: {predictor.device}')

# Test prediction speed
sample_image = 'data/ne2025/device1/1201_LeafPhotoA_2025-09-08 10_44_12.793-05_00.jpg'
start = time.time()
result = predictor.predict_everything(sample_image)
pred_time = time.time() - start

print(f'Prediction time: {pred_time:.2f}s')
print(f'Segments found: {len(result[\"masks\"])}')
"
```