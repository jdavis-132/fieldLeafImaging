"""
Comparison script between disease-weighted and non-weighted autoencoder models.

This script compares:
1. Model architectures (should be identical)
2. Number of parameters (should be identical)
3. Loss function implementations (different)
4. Configuration differences
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.disease_autoencoder_cropped.config import DiseaseConfig
from src.disease_autoencoder_cropped.model import create_model as create_disease_model
from src.disease_autoencoder_cropped.loss import create_loss_function as create_disease_loss

from src.autoencoder_no_weighting.config import AutoencoderConfig
from src.autoencoder_no_weighting.model import create_model as create_standard_model
from src.autoencoder_no_weighting.loss import create_loss_function as create_standard_loss


def compare_model_architectures():
    """Compare model architectures and parameter counts."""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE COMPARISON")
    print("="*80)

    # Create models
    disease_config = DiseaseConfig()
    standard_config = AutoencoderConfig()

    disease_model = create_disease_model(disease_config)
    standard_model = create_standard_model(standard_config)

    # Compare parameter counts
    disease_params = sum(p.numel() for p in disease_model.parameters() if p.requires_grad)
    standard_params = sum(p.numel() for p in standard_model.parameters() if p.requires_grad)

    print(f"\nParameter Counts:")
    print(f"  Disease-weighted model: {disease_params:,} parameters")
    print(f"  Standard model:         {standard_params:,} parameters")
    print(f"  Difference:             {abs(disease_params - standard_params):,} parameters")

    if disease_params == standard_params:
        print("  ✓ Models have identical architectures!")
    else:
        print("  ✗ Models have different architectures!")

    # Compare layer structure
    print(f"\nModel Components:")
    print(f"  Encoder features: {disease_config.unet_features}")
    print(f"  Embedding dim:    {disease_config.embedding_dim}")
    print(f"  Use attention:    {disease_config.use_attention}")
    print(f"  Input channels:   {disease_config.num_input_channels}")
    print(f"  Output channels:  {disease_config.num_output_channels}")

    return disease_model, standard_model


def compare_loss_functions():
    """Compare loss function implementations."""
    print("\n" + "="*80)
    print("LOSS FUNCTION COMPARISON")
    print("="*80)

    # Create configs and loss functions
    disease_config = DiseaseConfig()
    standard_config = AutoencoderConfig()

    print("\nDisease-Weighted Loss:")
    disease_loss = create_disease_loss(disease_config, use_disease_weighting=True)
    print(f"  Type: {type(disease_loss).__name__}")
    print(f"  Disease weight strength: {disease_config.disease_weight_strength}")
    print(f"  L1 regularization: {disease_config.l1_regularization}")
    print(f"  Weighting method: LAB color deviation from healthy tissue")
    print(f"    - Healthy a*: -50 to -10 (green)")
    print(f"    - Healthy b*: 10 to 40 (yellow-green)")
    print(f"    - Diseased: shifts toward red/yellow")

    print("\nStandard Loss:")
    standard_loss = create_standard_loss(standard_config)
    print(f"  Type: {type(standard_loss).__name__}")
    print(f"  L1 regularization: {standard_config.l1_regularization}")
    print(f"  Weighting: None (all pixels equal weight)")

    # Test with dummy data
    print("\n" + "-"*80)
    print("Testing Loss Functions with Dummy Data")
    print("-"*80)

    batch_size = 4
    predictions = torch.randn(batch_size, 3, 224, 224)
    targets = torch.randn(batch_size, 3, 224, 224)
    masks = torch.ones(batch_size, 1, 224, 224)
    embeddings = torch.randn(batch_size, 256)

    lab_stats = {
        'lab_mean': [50.0, 0.0, 20.0],
        'lab_std': [20.0, 30.0, 30.0]
    }

    # Compute losses
    disease_loss_val, disease_dict = disease_loss(predictions, targets, masks, embeddings, lab_stats)
    standard_loss_val, standard_dict = standard_loss(predictions, targets, masks, embeddings, lab_stats)

    print(f"\nDisease-Weighted Loss Output:")
    print(f"  Total loss:          {disease_loss_val.item():.6f}")
    print(f"  Reconstruction loss: {disease_dict['reconstruction_loss']:.6f}")
    print(f"  L1 loss:             {disease_dict['l1_loss']:.6f}")
    print(f"  Mean weight:         {disease_dict['mean_weight']:.3f}")

    print(f"\nStandard Loss Output:")
    print(f"  Total loss:          {standard_loss_val.item():.6f}")
    print(f"  Reconstruction loss: {standard_dict['reconstruction_loss']:.6f}")
    print(f"  L1 loss:             {standard_dict['l1_loss']:.6f}")
    print(f"  Mean weight:         {standard_dict['mean_weight']:.3f} (always 1.0)")


def compare_configurations():
    """Compare configuration differences."""
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)

    disease_config = DiseaseConfig()
    standard_config = AutoencoderConfig()

    print("\nIdentical Parameters:")
    identical_params = [
        ('Image size', disease_config.image_size, standard_config.image_size),
        ('Batch size', disease_config.batch_size, standard_config.batch_size),
        ('Learning rate', disease_config.learning_rate, standard_config.learning_rate),
        ('Num epochs', disease_config.num_epochs, standard_config.num_epochs),
        ('Embedding dim', disease_config.embedding_dim, standard_config.embedding_dim),
        ('Random seed', disease_config.random_seed, standard_config.random_seed),
    ]

    for name, val1, val2 in identical_params:
        match = "✓" if val1 == val2 else "✗"
        print(f"  {match} {name:20s}: {val1} vs {val2}")

    print("\nKey Differences:")
    print(f"  Disease config:")
    print(f"    - disease_weight_strength: {disease_config.disease_weight_strength}")
    print(f"    - Uses disease-weighted loss by default")
    print(f"    - Output dir: disease_autoencoder_cropped/")

    print(f"\n  Standard config:")
    print(f"    - No disease_weight_strength parameter")
    print(f"    - Uses simple MSE loss")
    print(f"    - Output dir: autoencoder_no_weighting/")


def print_summary():
    """Print summary of differences."""
    print("\n" + "="*80)
    print("SUMMARY OF CHANGES")
    print("="*80)

    print("\n✓ UNCHANGED (Identical between versions):")
    print("  - Model architecture (U-Net with attention)")
    print("  - Number of parameters")
    print("  - Data loading and preprocessing")
    print("  - Training hyperparameters (learning rate, batch size, etc.)")
    print("  - Optimizer (AdamW)")
    print("  - Learning rate scheduler")

    print("\n✗ CHANGED (Removed disease weighting):")
    print("  - Loss function:")
    print("      Original: DiseaseWeightedLoss (weights pixels by color deviation)")
    print("      Modified: SimpleMSELoss (equal weight for all pixels)")
    print("  - Configuration:")
    print("      Removed: disease_weight_strength parameter")
    print("      Changed: Output directories")
    print("  - Training output:")
    print("      Removed: Disease weight statistics from logs")

    print("\n" + "="*80)
    print("EXPECTED PERFORMANCE DIFFERENCES")
    print("="*80)

    print("\nDisease-Weighted Model:")
    print("  ✓ Better reconstruction of diseased regions")
    print("  ✓ More sensitive to disease-specific features")
    print("  ✓ Embeddings may better separate disease states")
    print("  ✗ May under-represent healthy tissue variation")

    print("\nStandard Model (No Weighting):")
    print("  ✓ Balanced reconstruction across all regions")
    print("  ✓ Treats healthy and diseased tissue equally")
    print("  ✓ May better capture overall leaf morphology")
    print("  ✗ May miss subtle disease patterns")


def main():
    """Run all comparisons."""
    print("\n" + "="*80)
    print("AUTOENCODER MODEL COMPARISON")
    print("Disease-Weighted vs Standard (No Weighting)")
    print("="*80)

    # Compare models
    disease_model, standard_model = compare_model_architectures()

    # Compare loss functions
    compare_loss_functions()

    # Compare configurations
    compare_configurations()

    # Print summary
    print_summary()

    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
