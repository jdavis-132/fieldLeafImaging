#!/usr/bin/env python3
"""
Complete pipeline runner for autoencoder training.

Runs all steps in sequence:
1. Prepare splits (genotype-based)
2. Visualize dataset
3. Train autoencoder
4. Extract embeddings
5. Generate final visualizations

Usage:
    python run_pipeline.py               # Run full pipeline
    python run_pipeline.py --skip-train  # Skip training (use existing model)
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from autoencoder.config import Config
from autoencoder import prepare_splits, train, extract_embeddings, visualize


def main():
    parser = argparse.ArgumentParser(
        description='Run complete autoencoder pipeline'
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip training and use existing model'
    )
    parser.add_argument(
        '--skip-splits',
        action='store_true',
        help='Skip split preparation and use existing splits'
    )
    args = parser.parse_args()

    config = Config()

    print("=" * 80)
    print("AUTOENCODER PIPELINE")
    print("=" * 80)
    print(f"Device: {config.device}")
    print(f"Image size: {config.image_size}")
    print(f"Embedding dim: {config.embedding_dim}")
    print(f"Batch size: {config.batch_size}")
    print("=" * 80)

    # Step 1: Prepare splits
    if not args.skip_splits:
        print("\n" + "=" * 80)
        print("STEP 1: Preparing Train/Val/Test Splits")
        print("=" * 80)
        prepare_splits.main()
    else:
        print("\n✓ Skipping split preparation (using existing splits)")

    # Step 2: Visualize dataset
    print("\n" + "=" * 80)
    print("STEP 2: Visualizing Dataset")
    print("=" * 80)
    visualize.main()

    # Step 3: Train model
    if not args.skip_train:
        print("\n" + "=" * 80)
        print("STEP 3: Training Autoencoder")
        print("=" * 80)
        train.main()
    else:
        print("\n✓ Skipping training (using existing model)")

        # Check if model exists
        model_path = config.models_dir / 'best_model.pth'
        if not model_path.exists():
            print(f"ERROR: Model not found at {model_path}")
            print("Please train the model first or run without --skip-train")
            sys.exit(1)

    # Step 4: Extract embeddings
    print("\n" + "=" * 80)
    print("STEP 4: Extracting Embeddings")
    print("=" * 80)
    extract_embeddings.main()

    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nGenerated outputs:")
    print(f"  Models:         {config.models_dir}/")
    print(f"  Embeddings:     {config.embeddings_dir}/")
    print(f"  Visualizations: {config.visualizations_dir}/")
    print(f"  Logs:           {config.logs_dir}/")
    print("\nKey files:")
    print(f"  - {config.models_dir / 'best_model.pth'}")
    print(f"  - {config.embeddings_dir / 'embeddings.csv'}")
    print(f"  - {config.visualizations_dir / 'SUMMARY_REPORT.txt'}")
    print(f"  - {config.visualizations_dir / 'training_curves.png'}")
    print(f"  - {config.visualizations_dir / 'embeddings_umap.png'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
