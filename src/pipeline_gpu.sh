#!/bin/bash
set -e  # Exit on error
set -x  # Print commands (for debugging)

# preprocess the data
#./src/autoencoder/preprocess_data.sh

# get DINOv2 embeddings
#python src/dinov2/extract_dinov2_features.py

# get SAM3 embeddings 
#python src/sam3/extract_embeddings.py "data/processed/ne2025/device*/cropped/*.png" -o output/sam3_embeddings.csv
python src/sam3/extract_embeddings.py "data/processed/aamu2025/block*/cropped/*.png" -o output/sam3_embeddings_aamu.csv
python src/sam3/extract_embeddings.py "data/processed/fvsu2025/*ap*/cropped/*.png" -o output/sam3_embeddings_fvsu.csv
# train models

# Loop through each config file in src/autoencoder/configs/
#for config_file in src/autoencoder/configs/config_*.yaml; do
    # Check if any config files exist
 #   if [ ! -f "$config_file" ]; then
  #      echo "No config files found in src/autoencoder/configs/"
   #     break
    #fi

    #echo "=================================================="
    #echo "Training model with config: $config_file"
    #echo "=================================================="

    # Get list of existing model directories before training
    #before_models=$(ls -dt models/autoencoder_* 2>/dev/null | head -1)

    # Train the model with the current config file
    #python src/autoencoder/train.py --config "$config_file"

    # Check if training was successful
    #if [ $? -ne 0 ]; then
     #   echo "ERROR: Training failed for $config_file"
      #  continue
    #fi

    # Find the newly created model directory (most recent)
    #after_models=$(ls -dt models/autoencoder_* 2>/dev/null | head -1)

    #if [ "$after_models" = "$before_models" ] || [ -z "$after_models" ]; then
     #   echo "ERROR: Could not find newly created model directory"
      #  continue
    #fi

    #model_dir="$after_models"
    #echo ""
    #echo "Model trained successfully: $model_dir"
    #echo ""

    # Extract embeddings for all splits and save to model directory
    #echo "Extracting embeddings for all splits..."
    #python src/autoencoder/extract_embeddings.py \
     #   --model_dir "$model_dir" \
      #  --output_dir "$model_dir" \
       # --splits train val test

    # Check if embedding extraction was successful
    #if [ $? -ne 0 ]; then
     #   echo "ERROR: Embedding extraction failed for $model_dir"
      #  continue
    #fi

    #echo ""
    #echo "Running evaluation pipeline on validation split..."

    # Run full evaluation on validation split
    #python src/autoencoder/eval.py \
     #   --model_paths "$model_dir" \
      #  --splits val \
       # --comparison_types reconstruction latent training \
        #--output_dir "${model_dir}/evaluation"

    # Check if evaluation was successful
    #if [ $? -ne 0 ]; then
     #   echo "ERROR: Evaluation failed for $model_dir"
      #  continue
    #fi

#    echo ""
 #   echo "=================================================="
  #  echo "Completed pipeline for: $config_file"
   # echo "Model directory: $model_dir"
    #echo "Embeddings saved to: $model_dir/embeddings_*.csv"
    #echo "Evaluation results: ${model_dir}/evaluation/"
    #echo "=================================================="
    #echo ""

#done

#echo "All models trained and evaluated!"

#python src/autoencoder/train.py --config models/autoencoder_20260108_012555_standard_lr0.001_bs32_l1_attention/config.yaml --resume models/autoencoder_20260108_012555_standard_lr0.001_bs32_l1_attention/checkpoints/checkpoint_epoch_500.pt
#python src/autoencoder/train.py --config models/autoencoder_20260109_034107_standard_lr0.001_bs32_l1_attention/config.yaml --resume models/autoencoder_20260109_034107_standard_lr0.001_bs32_l1_attention/checkpoints/checkpoint_epoch_500.pt
#python src/autoencoder/extract_embeddings.py --model_dir models/autoencoder_20260108_012555_standard_lr0.001_bs32_l1_attention --output_dir models/autoencoder_20260108_012555_standard_lr0.001_bs32_l1_attention --splits train val test
#python src/autoencoder/extract_embeddings.py --model_dir models/autoencoder_20260109_034107_standard_lr0.001_bs32_l1_attention --output_dir models/autoencoder_20260109_034107_standard_lr0.001_bs32_l1_attention --splits train val test
