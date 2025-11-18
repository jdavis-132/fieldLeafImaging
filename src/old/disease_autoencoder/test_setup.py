"""
Test script to verify the disease autoencoder setup.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.disease_autoencoder.config import DiseaseConfig
        print("✓ Config imported successfully")

        from src.disease_autoencoder.model import DiseaseAutoencoder, create_model
        print("✓ Model imported successfully")

        from src.disease_autoencoder.loss import create_loss_function
        print("✓ Loss function imported successfully")

        from src.disease_autoencoder.dataset import DiseaseLeafDataset
        print("✓ Dataset imported successfully")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_creation():
    """Test that model can be created."""
    print("\nTesting model creation...")

    try:
        from src.disease_autoencoder.config import DiseaseConfig
        from src.disease_autoencoder.model import test_model_shapes

        config = DiseaseConfig()
        model = test_model_shapes(config)

        print("✓ Model created and tested successfully")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_function():
    """Test that loss function works."""
    print("\nTesting loss function...")

    try:
        import torch
        from src.disease_autoencoder.config import DiseaseConfig
        from src.disease_autoencoder.loss import create_loss_function

        config = DiseaseConfig()
        loss_fn = create_loss_function(config, use_disease_weighting=True)

        # Create dummy data
        predictions = torch.randn(4, 3, 224, 224)
        targets = torch.randn(4, 3, 224, 224)
        masks = torch.ones(4, 1, 224, 224)
        embeddings = torch.randn(4, 256)

        lab_stats = {
            'lab_mean': [50.0, 0.0, 20.0],
            'lab_std': [20.0, 30.0, 30.0]
        }

        loss, loss_dict = loss_fn(predictions, targets, masks, embeddings, lab_stats)

        print(f"  Loss value: {loss.item():.4f}")
        print("✓ Loss function works correctly")
        return True
    except Exception as e:
        print(f"✗ Loss function failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_availability():
    """Test that data files exist."""
    print("\nTesting data availability...")

    from src.disease_autoencoder.config import DiseaseConfig

    config = DiseaseConfig()

    # Check directories
    if not config.data_dir.exists():
        print(f"✗ Data directory not found: {config.data_dir}")
        return False
    print(f"✓ Data directory exists: {config.data_dir}")

    if not config.output_dir.exists():
        print(f"✗ Output directory not found: {config.output_dir}")
        return False
    print(f"✓ Output directory exists: {config.output_dir}")

    if not config.csv_path.exists():
        print(f"✗ CSV file not found: {config.csv_path}")
        return False
    print(f"✓ CSV file exists: {config.csv_path}")

    # Check for some images
    device_dirs = list(config.data_dir.glob('device*'))
    if len(device_dirs) == 0:
        print(f"✗ No device directories found in {config.data_dir}")
        return False
    print(f"✓ Found {len(device_dirs)} device directories")

    # Check for some masks
    mask_dirs = list((config.output_dir / 'stripSegmentation' / 'ne2025').glob('device*'))
    if len(mask_dirs) == 0:
        print(f"✗ No mask directories found")
        return False
    print(f"✓ Found {len(mask_dirs)} mask directories")

    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Disease Autoencoder Setup Test")
    print("="*60)

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test data
    results.append(("Data Availability", test_data_availability()))

    # Test model
    results.append(("Model Creation", test_model_creation()))

    # Test loss
    results.append(("Loss Function", test_loss_function()))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n🎉 All tests passed! You're ready to train the model.")
        print("\nNext steps:")
        print("  1. Train: python -m disease_autoencoder.train")
        print("  2. Evaluate: python -m disease_autoencoder.evaluate")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")

    return all_passed


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
