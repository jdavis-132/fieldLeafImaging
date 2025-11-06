# Complete CUDA Setup Guide for SAM 2.1

## Current Status ✅ ❌
- ✅ NVIDIA GeForce RTX 3060 detected
- ✅ NVIDIA drivers version 470 installed
- ❌ NVIDIA kernel modules not loaded
- ❌ SAM 2.1 requires PyTorch >= 2.5.1 but driver 470 only supports CUDA 11.4
- ❌ PyTorch 2.1.0+cu118 installed (compromised version)

## The Real Solution (Recommended)

### Option 1: Update NVIDIA Drivers to Support Recent CUDA

```bash
# 1. Load current NVIDIA modules first
sudo modprobe nvidia
nvidia-smi  # This should work with your current driver 470

# 2. Update to newer NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-525  # Or newer
sudo reboot

# 3. After reboot, reinstall proper PyTorch
pip uninstall torch torchvision sam2 -y
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install sam2

# 4. Test
nvidia-smi
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
python3 -c "from sam2_tiny import SAM2TinyPredictor; p = SAM2TinyPredictor()"
```

### Option 2: Make Current Setup Work (Driver 470 + CUDA 11.4)

```bash
# 1. Load NVIDIA kernel modules
sudo modprobe nvidia
nvidia-smi  # Should work

# 2. Install compatible versions
pip uninstall torch torchvision sam2 -y

# Install SAM 2.0 (older, compatible version)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/facebookresearch/segment-anything-2.git@v1.0

# 3. Test
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Quick Fix to Test Current Hardware

### Step 1: Load NVIDIA Modules
```bash
# This is the key step - load the kernel modules
sudo modprobe nvidia
sudo modprobe nvidia_uvm

# Verify
nvidia-smi
lsmod | grep nvidia
```

### Step 2: Test PyTorch CUDA
```bash
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('CUDA capability:', torch.cuda.get_device_capability(0))
    # Simple tensor test
    x = torch.randn(3, 3).cuda()
    print('GPU tensor test passed!')
"
```

### Step 3: Test SAM 2.1 (may have compatibility issues)
```bash
python3 -c "
try:
    from sam2_tiny import SAM2TinyPredictor
    predictor = SAM2TinyPredictor(device='cuda')
    print('✅ SAM 2.1 GPU test successful!')
except Exception as e:
    print('❌ SAM 2.1 GPU test failed:', e)
    print('🔄 Trying CPU fallback...')
    predictor = SAM2TinyPredictor(device='cpu')
    print('✅ SAM 2.1 CPU fallback successful!')
"
```

## Expected Performance Gains

Once GPU is working:

| Mode | Device | Time/Image | Memory Usage |
|------|--------|------------|--------------|
| Current | CPU | ~30-60s | ~2GB RAM |
| With GPU | RTX 3060 | ~2-5s | ~4GB VRAM |
| **Speedup** | **GPU** | **10-20x faster** | **Dedicated VRAM** |

## Troubleshooting Commands

### Check GPU Hardware
```bash
lspci | grep -i nvidia
lshw -C display  # Detailed GPU info
```

### Check Driver Status
```bash
nvidia-smi                    # Driver version and GPU status
nvidia-settings              # GUI driver settings
cat /proc/driver/nvidia/version
```

### Check CUDA
```bash
nvcc --version               # CUDA compiler
nvidia-ml-py --version      # Python NVIDIA ML library
```

### Check PyTorch
```bash
python3 -c "
import torch
print('Version:', torch.__version__)
print('CUDA compiled:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
"
```

## Currently Working Setup (CPU)

Your implementation will work on CPU with these optimizations:

```bash
# Set optimal CPU threads
export OMP_NUM_THREADS=$(nproc)

# Run SAM 2.1
python3 run_sam2.py -i "data/ne2025/device1/1201_LeafPhotoA_2025-09-08 10_44_12.793-05_00.jpg" -m auto
```

## Next Steps

1. **Immediate**: Run `sudo modprobe nvidia` and test
2. **Short-term**: Update NVIDIA drivers to 525+
3. **Long-term**: Consider upgrading to RTX 4060/4070 for even better performance

The SAM 2.1 implementation is ready and will automatically use GPU once you enable it!