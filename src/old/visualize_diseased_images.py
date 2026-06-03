"""
Visualize autoencoder reconstructions for specific diseased images.

This script generates visualizations (original, reconstruction, error map) for images
identified in lnk_35pctdiseased.txt using multiple trained autoencoder models.
"""

import os
import sys
from pathlib import Path
import glob
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'autoencoder'))

# Import from eval.py
from src.autoencoder.eval import (
    load_model_info,
    compute_image_metrics
)
from src.autoencoder.model import ConvolutionalAutoencoder


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for model input.

    Args:
        image_path: Path to image file
        target_size: Target size (height, width)

    Returns:
        Tuple of (preprocessed_tensor, original_numpy)
    """
    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize if needed
    if img.shape[:2] != target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]),
                        interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1]
    img_normalized = img.astype(np.float32) / 255.0

    # Convert to tensor (C, H, W)
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)

    return img_tensor, img_normalized


def find_matching_images(image_identifier, base_data_dir):
    """
    Find all image files matching the identifier.

    Args:
        image_identifier: String in format "device*/filename"
        base_data_dir: Base directory for data

    Returns:
        List of Path objects for matching images
    """
    # Parse identifier
    parts = image_identifier.strip().split('/')
    if len(parts) != 2:
        return []

    device, filename = parts

    # Construct search pattern
    search_pattern = f"{base_data_dir}/ne2025/{device}/cropped/{filename}_*.png"

    # Find matching files
    matching_files = sorted(glob.glob(search_pattern))

    return [Path(f) for f in matching_files]


def visualize_reconstruction(model_info, image_path, device, output_path, image_id):
    """
    Create visualization for a single image reconstruction.

    Args:
        model_info: Dictionary with model information from load_model_info
        image_path: Path to image
        device: Device for computation
        output_path: Path to save visualization
        image_id: Image identifier for title
    """
    model = model_info['model']
    is_vae = model_info['config']['model'].get('is_vae', False)
    model_name = model_info['name']

    # Get target size from model config
    target_size = tuple(model_info['config']['image']['target_size'])

    # Load and preprocess image
    img_tensor, img_np = load_and_preprocess_image(image_path, target_size=target_size)

    # Add batch dimension and move to device
    img_batch = img_tensor.unsqueeze(0).to(device)

    # Generate reconstruction
    with torch.no_grad():
        if is_vae:
            reconstruction, _, _ = model(img_batch)
        else:
            reconstruction = model(img_batch)

    # Convert to numpy
    recon_np = reconstruction.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # Compute metrics using function from eval.py
    metrics = compute_image_metrics(img_np, recon_np, mask=None, data_range=1.0)

    # Create visualization
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, hspace=0.05, wspace=0.15)

    # Original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.clip(img_np, 0, 1))
    ax.axis('off')
    ax.set_title('Original', fontsize=12, fontweight='bold')

    # Reconstruction
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(np.clip(recon_np, 0, 1))
    ax.axis('off')
    ax.set_title('Reconstruction', fontsize=12, fontweight='bold')

    # Error map
    ax = fig.add_subplot(gs[0, 2])
    error = np.abs(img_np - recon_np).mean(axis=2)
    im = ax.imshow(error, cmap='hot', vmin=0, vmax=0.3)
    ax.axis('off')
    ax.set_title('Absolute Error', fontsize=12, fontweight='bold')

    # Add colorbar for error map
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Error', rotation=270, labelpad=15)

    # Add metrics as text
    metrics_text = (
        f"MSE: {metrics['mse']:.6f}\n"
        f"MAE: {metrics['mae']:.6f}\n"
        f"SSIM: {metrics['ssim']:.4f}\n"
        f"PSNR: {metrics['psnr']:.2f} dB"
    )

    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Title
    plt.suptitle(f"{model_name}\n{image_id} - {Path(image_path).name}",
                fontsize=14, fontweight='bold', y=0.98)

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_multi_model_comparison(models_info_list, image_path, image_id, output_path, device):
    """
    Create a comparison visualization across all models for a single image.

    Args:
        models_info_list: List of model_info dictionaries from load_model_info
        image_path: Path to image
        image_id: Image identifier
        output_path: Path to save visualization
        device: Device for computation
    """
    n_models = len(models_info_list)

    # Get target size from first model's config (assuming all models use same size)
    target_size = tuple(models_info_list[0]['config']['image']['target_size'])

    # Load and preprocess image once
    img_tensor, img_np = load_and_preprocess_image(image_path, target_size=target_size)
    img_batch = img_tensor.unsqueeze(0).to(device)

    # Create figure with 5 columns
    fig = plt.figure(figsize=(30, 3.5 * n_models))
    gs = GridSpec(n_models, 5, figure=fig, hspace=0.3, wspace=0.15)

    for row, model_info in enumerate(models_info_list):
        model = model_info['model']
        is_vae = model_info['config']['model'].get('is_vae', False)
        model_name = model_info['name']

        # Generate reconstruction
        with torch.no_grad():
            if is_vae:
                reconstruction, _, _ = model(img_batch)
            else:
                reconstruction = model(img_batch)

        # Convert to numpy
        recon_np = reconstruction.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        # Compute metrics using function from eval.py
        metrics = compute_image_metrics(img_np, recon_np, mask=None, data_range=1.0)

        # Compute error map
        error = np.abs(img_np - recon_np).mean(axis=2)

        # Original
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(np.clip(img_np, 0, 1))
        ax.axis('off')
        if row == 0:
            ax.set_title('Original', fontsize=12, fontweight='bold')
        ax.text(-0.1, 0.5, model_name, rotation=90, va='center', ha='right',
               transform=ax.transAxes, fontsize=10, fontweight='bold')

        # Reconstruction
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(np.clip(recon_np, 0, 1))
        ax.axis('off')
        if row == 0:
            ax.set_title('Reconstruction', fontsize=12, fontweight='bold')

        # Error map
        ax = fig.add_subplot(gs[row, 2])
        im = ax.imshow(error, cmap='hot', vmin=0, vmax=0.3)
        ax.axis('off')
        if row == 0:
            ax.set_title('Absolute Error', fontsize=12, fontweight='bold')

        # Error overlay on reconstruction
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(np.clip(recon_np, 0, 1))
        ax.imshow(error, cmap='hot', alpha=0.5, vmin=0, vmax=0.3)
        ax.axis('off')
        if row == 0:
            ax.set_title('Error on Reconstruction', fontsize=12, fontweight='bold')

        # Error overlay on original
        ax = fig.add_subplot(gs[row, 4])
        ax.imshow(np.clip(img_np, 0, 1))
        ax.imshow(error, cmap='hot', alpha=0.5, vmin=0, vmax=0.3)
        ax.axis('off')
        if row == 0:
            ax.set_title('Error on Original', fontsize=12, fontweight='bold')

        # Add metrics text
        metrics_text = (
            f"SSIM: {metrics['ssim']:.4f} | "
            f"PSNR: {metrics['psnr']:.1f} dB | "
            f"MSE: {metrics['mse']:.6f}"
        )
        ax.text(0.5, -0.15, metrics_text, transform=ax.transAxes,
               ha='center', fontsize=9)

        # Add colorbar to rightmost column
        if row == n_models - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Error', rotation=270, labelpad=15)

    # Title
    plt.suptitle(f"Model Comparison: {image_id} - {Path(image_path).name}",
                fontsize=14, fontweight='bold', y=0.995)

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main execution."""
    # Configuration
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data' / 'processed'
    lnk_file = base_dir / 'output' / 'lnk_35pctdiseased.txt'

    model_dirs = [
        'models/autoencoder_20260107_143328_standard_lr0.001_bs32_l1',
        'models/autoencoder_20260107_214228_standard_lr0.001_bs32_l1',
	'models/autoencoder_20260108_183222_standard_lr0.001_bs32_l1_attention',
	'models/autoencoder_20260109_205057_standard_lr0.001_bs32_l1',
	'models/autoencoder_20260110_045517_standard_lr0.001_bs32_disease_weighted_l1'
    ]

    output_dir = base_dir / 'output' / 'diseased_reconstructions'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each model
    for model_dir in model_dirs:
        model_name = Path(model_dir).name
        (output_dir / model_name).mkdir(parents=True, exist_ok=True)

    # Create comparison directory
    (output_dir / 'comparisons').mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load all models using load_model_info from eval.py
    print("\nLoading models...")
    models_info_list = []
    for model_dir in model_dirs:
        model_path = base_dir / model_dir
        if not model_path.exists():
            print(f"Warning: Model directory not found: {model_path}")
            continue

        try:
            model_info = load_model_info(model_path, device)
            if model_info is not None:
                models_info_list.append(model_info)
                print(f"  ✓ Loaded {model_info['name']} as '{model_info['short_name']}'")
            else:
                print(f"  ✗ Failed to load {model_dir}")
        except Exception as e:
            print(f"  ✗ Failed to load {model_dir}: {e}")

    if not models_info_list:
        print("Error: No models could be loaded!")
        return

    print(f"\nSuccessfully loaded {len(models_info_list)} models")

    # Read image identifiers
    print(f"\nReading image identifiers from: {lnk_file}")
    with open(lnk_file, 'r') as f:
        image_identifiers = [line.strip() for line in f if line.strip()]

    print(f"Found {len(image_identifiers)} image identifiers")

    # Process each image
    print("\nGenerating visualizations...")
    total_images_processed = 0

    for image_id in tqdm(image_identifiers, desc="Processing images"):
        # Find matching image files
        matching_images = find_matching_images(image_id, data_dir)

        if not matching_images:
            print(f"Warning: No images found for {image_id}")
            continue

        # Process each matching image file
        for image_path in matching_images:
            image_basename = image_path.stem  # Filename without extension

            # Generate individual model visualizations
            for model_info in models_info_list:
                model_name = model_info['name']
                output_path = output_dir / model_name / f"{image_basename}.png"

                try:
                    visualize_reconstruction(
                        model_info, image_path, device,
                        output_path, image_id
                    )
                except Exception as e:
                    print(f"Error processing {image_path} with {model_name}: {e}")

            # Generate comparison visualization across all models
            comparison_output = output_dir / 'comparisons' / f"{image_basename}_comparison.png"
            try:
                create_multi_model_comparison(
                    models_info_list, image_path, image_id,
                    comparison_output, device
                )
            except Exception as e:
                print(f"Error creating comparison for {image_path}: {e}")

            total_images_processed += 1

    print(f"\n{'='*80}")
    print(f"COMPLETE")
    print(f"{'='*80}")
    print(f"Processed {total_images_processed} images")
    print(f"Output directory: {output_dir}")
    print(f"  - Individual model reconstructions: {output_dir}/<model_name>/")
    print(f"  - Multi-model comparisons: {output_dir}/comparisons/")


if __name__ == "__main__":
    main()
