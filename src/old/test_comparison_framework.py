"""
Quick test script to validate the comparison framework installation and setup.

This script performs basic checks without running the full comparison.
"""

import sys
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_error(msg):
    print(f"{Colors.RED}✗{Colors.END} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n" + "="*60)
    print("Checking Dependencies")
    print("="*60)

    dependencies = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('sklearn', 'Scikit-learn'),
        ('skimage', 'Scikit-image'),
        ('pandas', 'Pandas'),
        ('tqdm', 'tqdm'),
    ]

    optional = [
        ('umap', 'UMAP-learn'),
    ]

    all_ok = True

    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print_success(f"{display_name} installed")
        except ImportError:
            print_error(f"{display_name} NOT installed")
            print(f"         Install with: pip install {module_name}")
            all_ok = False

    print("\nOptional dependencies:")
    for module_name, display_name in optional:
        try:
            __import__(module_name)
            print_success(f"{display_name} installed")
        except ImportError:
            print_warning(f"{display_name} NOT installed (optional)")
            print(f"         Install with: pip install {module_name}")

    return all_ok


def check_model_directories():
    """Check if model directories exist with required files."""
    print("\n" + "="*60)
    print("Checking Model Directories")
    print("="*60)

    base_dir = Path(__file__).parent

    models = [
        'disease_autoencoder',
        'disease_autoencoder_cropped',
        'autoencoder_no_weighting'
    ]

    required_files = [
        'config.py',
        'model.py',
        'models/checkpoint_best.pth',
        'logs/lab_statistics.json',
        'logs/image_splits.json',
        'logs/training_history.json'
    ]

    all_ok = True

    for model_name in models:
        print(f"\n{model_name}:")
        model_dir = base_dir / model_name

        if not model_dir.exists():
            print_error(f"Directory not found: {model_dir}")
            all_ok = False
            continue

        model_ok = True
        for req_file in required_files:
            file_path = model_dir / req_file
            if file_path.exists():
                print_success(f"  {req_file}")
            else:
                print_error(f"  {req_file} NOT FOUND")
                model_ok = False
                all_ok = False

        if model_ok:
            print_success(f"All required files present")

    return all_ok


def check_cuda():
    """Check CUDA availability."""
    print("\n" + "="*60)
    print("Checking CUDA/GPU")
    print("="*60)

    try:
        import torch
        if torch.cuda.is_available():
            print_success(f"CUDA available")
            print_info(f"  Device: {torch.cuda.get_device_name(0)}")
            print_info(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print_warning("CUDA not available - will use CPU (slower)")
            return False
    except Exception as e:
        print_error(f"Error checking CUDA: {e}")
        return False


def test_load_model():
    """Test loading one model."""
    print("\n" + "="*60)
    print("Testing Model Loading")
    print("="*60)

    try:
        from comprehensive_model_comparison import ModelLoader
        from pathlib import Path

        base_dir = Path(__file__).parent
        model_dir = base_dir / "disease_autoencoder_cropped"

        print_info("Loading disease_autoencoder_cropped...")

        model, checkpoint, config_dict = ModelLoader.load_model(
            model_dir,
            "checkpoint_best.pth",
            "cpu"  # Use CPU for testing
        )

        print_success("Model loaded successfully")
        print_info(f"  Epochs trained: {checkpoint.get('epoch', 'N/A')}")
        print_info(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.6f}")
        print_info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return True

    except Exception as e:
        print_error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_access():
    """Check if validation data is accessible."""
    print("\n" + "="*60)
    print("Checking Data Access")
    print("="*60)

    try:
        import json
        base_dir = Path(__file__).parent
        splits_path = base_dir / "disease_autoencoder_cropped" / "logs" / "image_splits.json"

        with open(splits_path, 'r') as f:
            splits = json.load(f)

        val_images = [img for img in splits['images'] if img['split'] == 'val']
        print_success(f"Found {len(val_images)} validation images")

        # Check if a few sample images exist
        sample_count = 0
        for img in val_images[:5]:
            import os
            if os.path.exists(img['image_path']):
                sample_count += 1

        if sample_count > 0:
            print_success(f"Verified {sample_count}/5 sample images accessible")
            return True
        else:
            print_warning("Could not access sample images - check paths")
            return False

    except Exception as e:
        print_error(f"Failed to check data access: {e}")
        return False


def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("Model Comparison Framework - Installation Test")
    print("="*60)

    results = {
        'Dependencies': check_dependencies(),
        'Model Directories': check_model_directories(),
        'CUDA/GPU': check_cuda(),
        'Model Loading': test_load_model(),
        'Data Access': check_data_access()
    }

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    all_passed = True
    for check_name, passed in results.items():
        if passed:
            print_success(f"{check_name}: PASSED")
        else:
            print_error(f"{check_name}: FAILED")
            all_passed = False

    print("\n" + "="*60)

    if all_passed:
        print_success("All checks passed! Ready to run comparison.")
        print("\nRun comparison with:")
        print("  ./run_model_comparison.sh 10    # Test with 10 samples")
        print("  ./run_model_comparison.sh       # Full comparison")
        print()
        return 0
    else:
        print_error("Some checks failed. Please fix issues above.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
