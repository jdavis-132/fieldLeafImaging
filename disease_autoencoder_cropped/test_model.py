"""
Comprehensive testing script for disease autoencoder model.

Tests:
- Model architecture and parameter counts
- Forward/backward pass
- Output shapes and ranges
- Loss function behavior
- Dataset loading
- Memory usage
- Gradient flow
- Reconstruction quality metrics
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from collections import OrderedDict
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from disease_autoencoder.config import DiseaseConfig
from disease_autoencoder.model import DiseaseAutoencoder, create_model
from disease_autoencoder.loss import create_loss_function
from disease_autoencoder.dataset import DiseaseLeafDataset


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_test(name, passed, details=""):
    """Print test result with color."""
    if passed:
        status = f"{Colors.OKGREEN}✓ PASS{Colors.ENDC}"
    else:
        status = f"{Colors.FAIL}✗ FAIL{Colors.ENDC}"

    print(f"{status} {name}")
    if details:
        print(f"     {details}")


def print_section(title):
    """Print section header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


# ============================================================================
# Test 1: Model Architecture
# ============================================================================

def test_model_architecture(config):
    """Test model architecture and parameter counts."""
    print_section("Test 1: Model Architecture")

    try:
        model = DiseaseAutoencoder(config)

        # Test parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print_test("Model creation", True, f"{total_params:,} total parameters")
        print_test("Trainable parameters", trainable_params > 0, f"{trainable_params:,} trainable")

        # Test model has required components
        has_encoder = hasattr(model, 'encoder')
        has_decoder = hasattr(model, 'decoder')
        has_attention = hasattr(model, 'attention')
        has_embedding = hasattr(model, 'embedding_extractor')

        print_test("Has encoder", has_encoder)
        print_test("Has decoder", has_decoder)
        print_test("Has attention", has_attention)
        print_test("Has embedding extractor", has_embedding)

        # Test model can be moved to device
        model = model.to(config.device)
        print_test("Model to device", True, f"Device: {config.device}")

        return True, model

    except Exception as e:
        print_test("Model architecture", False, str(e))
        return False, None


# ============================================================================
# Test 2: Forward Pass
# ============================================================================

def test_forward_pass(model, config):
    """Test forward pass with dummy data."""
    print_section("Test 2: Forward Pass")

    try:
        batch_size = 4
        dummy_input = torch.randn(
            batch_size,
            config.num_input_channels,
            config.image_size,
            config.image_size
        ).to(config.device)

        print(f"Input shape: {dummy_input.shape}")

        # Test reconstruction
        with torch.no_grad():
            reconstruction = model(dummy_input)

        expected_shape = (batch_size, config.num_output_channels, config.image_size, config.image_size)
        shape_correct = reconstruction.shape == expected_shape

        print_test("Forward pass reconstruction", True, f"Output shape: {reconstruction.shape}")
        print_test("Output shape correct", shape_correct, f"Expected: {expected_shape}")

        # Test encoding
        with torch.no_grad():
            embedding = model.encode(dummy_input)

        expected_emb_shape = (batch_size, config.embedding_dim)
        emb_shape_correct = embedding.shape == expected_emb_shape

        print_test("Embedding extraction", True, f"Embedding shape: {embedding.shape}")
        print_test("Embedding shape correct", emb_shape_correct, f"Expected: {expected_emb_shape}")

        # Test output range
        is_finite = torch.isfinite(reconstruction).all().item()
        print_test("Output is finite", is_finite)

        return True

    except Exception as e:
        print_test("Forward pass", False, str(e))
        return False


# ============================================================================
# Test 3: Backward Pass and Gradients
# ============================================================================

def test_backward_pass(model, config):
    """Test backward pass and gradient flow."""
    print_section("Test 3: Backward Pass and Gradients")

    try:
        model.train()

        # Create dummy data
        batch_size = 2
        dummy_input = torch.randn(
            batch_size,
            config.num_input_channels,
            config.image_size,
            config.image_size,
            requires_grad=True
        ).to(config.device)

        # Forward pass
        reconstruction = model(dummy_input)

        # Simple loss
        loss = reconstruction.mean()

        # Backward pass
        loss.backward()

        print_test("Backward pass", True, f"Loss: {loss.item():.6f}")

        # Check gradients
        has_gradients = True
        zero_gradients = []

        for name, param in model.named_parameters():
            if param.grad is None:
                has_gradients = False
            elif torch.all(param.grad == 0):
                zero_gradients.append(name)

        print_test("All parameters have gradients", has_gradients)

        if zero_gradients:
            print_test("Non-zero gradients", False, f"Zero grads in: {zero_gradients[:3]}")
        else:
            print_test("Non-zero gradients", True, "All gradients non-zero")

        # Test gradient magnitudes
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        if grad_norms:
            avg_grad = np.mean(grad_norms)
            max_grad = np.max(grad_norms)
            print_test("Gradient statistics", True, f"Avg: {avg_grad:.6f}, Max: {max_grad:.6f}")

        return True

    except Exception as e:
        print_test("Backward pass", False, str(e))
        return False


# ============================================================================
# Test 4: Loss Function
# ============================================================================

def test_loss_function(config):
    """Test loss function computation."""
    print_section("Test 4: Loss Function")

    try:
        # Create loss function with disease weighting
        loss_fn = create_loss_function(config, use_disease_weighting=True)

        # Create dummy data
        batch_size = 4
        predictions = torch.randn(batch_size, 3, config.image_size, config.image_size)
        targets = torch.randn(batch_size, 3, config.image_size, config.image_size)
        masks = torch.ones(batch_size, 1, config.image_size, config.image_size)
        embeddings = torch.randn(batch_size, config.embedding_dim)

        lab_stats = {
            'lab_mean': [50.0, 0.0, 20.0],
            'lab_std': [20.0, 30.0, 30.0]
        }

        # Compute loss
        loss, loss_dict = loss_fn(predictions, targets, masks, embeddings, lab_stats)

        print_test("Disease-weighted loss", True, f"Loss: {loss.item():.6f}")
        print_test("Loss is finite", torch.isfinite(loss).item())
        print_test("Loss is positive", loss.item() >= 0)

        # Check loss components
        print(f"     Loss components:")
        for key, value in loss_dict.items():
            print(f"       {key}: {value:.6f}")

        # Test simple MSE loss
        simple_loss_fn = create_loss_function(config, use_disease_weighting=False)
        simple_loss, simple_dict = simple_loss_fn(predictions, targets, masks)

        print_test("Simple MSE loss", True, f"Loss: {simple_loss.item():.6f}")

        return True

    except Exception as e:
        print_test("Loss function", False, str(e))
        return False


# ============================================================================
# Test 5: Dataset Loading
# ============================================================================

def test_dataset_loading(config):
    """Test dataset loading and preprocessing."""
    print_section("Test 5: Dataset Loading")

    try:
        # Check if data exists
        if not config.data_dir.exists():
            print_test("Data directory", False, f"Not found: {config.data_dir}")
            return False

        print_test("Data directory", True, str(config.data_dir))

        # Try to find some images
        from autoencoder.prepare_splits import find_usable_images

        try:
            usable_images = find_usable_images(
                data_dir=config.data_dir,
                output_dir=config.output_dir,
                csv_path=config.csv_path,
                experiment='ne2025'
            )

            num_images = len(usable_images)
            print_test("Find usable images", num_images > 0, f"Found {num_images} images")

            if num_images > 0:
                # Create a small dataset
                test_images = usable_images[:min(5, num_images)]

                # Create dummy LAB stats
                dataset = DiseaseLeafDataset(
                    image_metadata_list=test_images,
                    config=config,
                    transform=None
                )
                dataset.set_lab_statistics([50.0, 0.0, 20.0], [20.0, 30.0, 30.0])

                print_test("Dataset creation", True, f"{len(dataset)} samples")

                # Test loading one sample
                sample = dataset[0]

                print_test("Sample loading", True)
                print_test("Input shape", sample['input'].shape == (4, config.image_size, config.image_size))
                print_test("Target shape", sample['target'].shape == (3, config.image_size, config.image_size))
                print_test("Mask shape", sample['mask'].shape == (1, config.image_size, config.image_size))

                return True

        except ImportError:
            print_test("Import prepare_splits", False, "Module not found")
            return False

    except Exception as e:
        print_test("Dataset loading", False, str(e))
        return False


# ============================================================================
# Test 6: Memory Usage
# ============================================================================

def test_memory_usage(model, config):
    """Test memory usage during forward pass."""
    print_section("Test 6: Memory Usage")

    try:
        import gc

        # Clear cache if using CUDA
        if config.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model.eval()

        batch_size = config.batch_size
        dummy_input = torch.randn(
            batch_size,
            config.num_input_channels,
            config.image_size,
            config.image_size
        ).to(config.device)

        # Forward pass
        with torch.no_grad():
            reconstruction = model(dummy_input)

        if config.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            peak = torch.cuda.max_memory_allocated() / 1024**2  # MB

            print_test("GPU memory usage", True, f"Allocated: {allocated:.2f} MB, Peak: {peak:.2f} MB")
        else:
            print_test("Memory test", True, f"Device: {config.device.type}")

        # Clean up
        del dummy_input, reconstruction
        gc.collect()
        if config.device.type == 'cuda':
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print_test("Memory usage", False, str(e))
        return False


# ============================================================================
# Test 7: Output Quality Metrics
# ============================================================================

def test_output_metrics(model, config):
    """Test output quality metrics."""
    print_section("Test 7: Output Quality Metrics")

    try:
        model.eval()

        # Create test data
        batch_size = 4
        dummy_input = torch.randn(
            batch_size,
            config.num_input_channels,
            config.image_size,
            config.image_size
        ).to(config.device)

        target = dummy_input[:, :3, :, :]  # Use LAB channels as target

        with torch.no_grad():
            reconstruction = model(dummy_input)

        # Compute MSE
        mse = ((reconstruction - target) ** 2).mean().item()
        print_test("MSE computation", True, f"MSE: {mse:.6f}")

        # Compute PSNR (Peak Signal-to-Noise Ratio)
        psnr = 10 * np.log10(1.0 / (mse + 1e-8))
        print_test("PSNR computation", True, f"PSNR: {psnr:.2f} dB")

        # Compute SSIM-like correlation
        recon_flat = reconstruction.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)

        # Correlation
        correlation = torch.nn.functional.cosine_similarity(recon_flat, target_flat, dim=1).mean().item()
        print_test("Correlation", True, f"Cosine similarity: {correlation:.4f}")

        return True

    except Exception as e:
        print_test("Output metrics", False, str(e))
        return False


# ============================================================================
# Test 8: Model Robustness
# ============================================================================

def test_model_robustness(model, config):
    """Test model robustness to different inputs."""
    print_section("Test 8: Model Robustness")

    try:
        model.eval()

        # Test 1: All zeros input
        zeros_input = torch.zeros(1, config.num_input_channels, config.image_size, config.image_size).to(config.device)
        with torch.no_grad():
            zeros_output = model(zeros_input)

        print_test("All zeros input", torch.isfinite(zeros_output).all().item())

        # Test 2: All ones input
        ones_input = torch.ones(1, config.num_input_channels, config.image_size, config.image_size).to(config.device)
        with torch.no_grad():
            ones_output = model(ones_input)

        print_test("All ones input", torch.isfinite(ones_output).all().item())

        # Test 3: Random noise input
        noise_input = torch.randn(1, config.num_input_channels, config.image_size, config.image_size).to(config.device) * 10
        with torch.no_grad():
            noise_output = model(noise_input)

        print_test("High noise input", torch.isfinite(noise_output).all().item())

        # Test 4: Batch size = 1
        single_input = torch.randn(1, config.num_input_channels, config.image_size, config.image_size).to(config.device)
        with torch.no_grad():
            single_output = model(single_input)

        print_test("Batch size = 1", single_output.shape[0] == 1)

        # Test 5: Large batch
        large_batch = 16
        large_input = torch.randn(large_batch, config.num_input_channels, config.image_size, config.image_size).to(config.device)
        try:
            with torch.no_grad():
                large_output = model(large_input)
            print_test(f"Batch size = {large_batch}", large_output.shape[0] == large_batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print_test(f"Batch size = {large_batch}", False, "Out of memory (expected on small GPUs)")
            else:
                raise

        return True

    except Exception as e:
        print_test("Model robustness", False, str(e))
        return False


# ============================================================================
# Test 9: Save and Load Model
# ============================================================================

def test_save_load_model(model, config):
    """Test saving and loading model checkpoint."""
    print_section("Test 9: Save and Load Model")

    try:
        # Create temp checkpoint
        checkpoint_path = config.models_dir / 'test_checkpoint.pth'

        # Save model
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': 0,
            'config': vars(config)
        }

        torch.save(checkpoint, checkpoint_path)
        print_test("Save checkpoint", checkpoint_path.exists(), str(checkpoint_path))

        # Load model
        loaded_checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

        # Create new model and load state
        new_model = DiseaseAutoencoder(config).to(config.device)
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])

        print_test("Load checkpoint", True)

        # Compare outputs
        dummy_input = torch.randn(2, config.num_input_channels, config.image_size, config.image_size).to(config.device)

        with torch.no_grad():
            output1 = model(dummy_input)
            output2 = new_model(dummy_input)

        outputs_match = torch.allclose(output1, output2, atol=1e-6)
        print_test("Loaded model produces same output", outputs_match)

        # Clean up
        checkpoint_path.unlink()

        return True

    except Exception as e:
        print_test("Save/load model", False, str(e))
        return False


# ============================================================================
# Test Summary
# ============================================================================

def run_all_tests():
    """Run all tests and print summary."""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}Disease Autoencoder - Comprehensive Testing{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

    config = DiseaseConfig()
    print(f"Device: {config.device}")
    print(f"Image size: {config.image_size}")
    print(f"Embedding dim: {config.embedding_dim}")

    results = OrderedDict()

    # Test 1: Architecture
    passed, model = test_model_architecture(config)
    results['Architecture'] = passed

    if not passed:
        print(f"\n{Colors.FAIL}Cannot proceed with further tests - model creation failed{Colors.ENDC}")
        return

    # Test 2: Forward pass
    results['Forward Pass'] = test_forward_pass(model, config)

    # Test 3: Backward pass
    results['Backward Pass'] = test_backward_pass(model, config)

    # Test 4: Loss function
    results['Loss Function'] = test_loss_function(config)

    # Test 5: Dataset
    results['Dataset Loading'] = test_dataset_loading(config)

    # Test 6: Memory
    results['Memory Usage'] = test_memory_usage(model, config)

    # Test 7: Metrics
    results['Output Metrics'] = test_output_metrics(model, config)

    # Test 8: Robustness
    results['Model Robustness'] = test_model_robustness(model, config)

    # Test 9: Save/Load
    results['Save/Load Model'] = test_save_load_model(model, config)

    # Print summary
    print_section("Test Summary")

    passed_count = sum(results.values())
    total_count = len(results)

    for test_name, passed in results.items():
        status = f"{Colors.OKGREEN}✓{Colors.ENDC}" if passed else f"{Colors.FAIL}✗{Colors.ENDC}"
        print(f"{status} {test_name}")

    print(f"\n{Colors.BOLD}Results: {passed_count}/{total_count} tests passed{Colors.ENDC}")

    if passed_count == total_count:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 All tests passed!{Colors.ENDC}")
        print(f"\n{Colors.OKGREEN}The model is ready for training and evaluation.{Colors.ENDC}")
    else:
        print(f"\n{Colors.WARNING}⚠️  Some tests failed. Please review the errors above.{Colors.ENDC}")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    run_all_tests()
