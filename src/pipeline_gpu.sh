# preprocess the data
./src/autoencoder/preprocess_data.sh

# get DINOv2 embeddings
python src/dinov2/extract_dinov2_features.py

# get SAM3 embeddings 
python src/sam3/extract_embeddings.py
