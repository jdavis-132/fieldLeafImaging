import json
import numpy as np
import pandas as pd

# Load embeddings array
print("Loading embeddings array...")
embeddings = np.load('/home/schnable/Documents/fieldLeafImaging/src/autoencoder_no_weighting/embeddings/latent_embeddings.npy')
print(f"Embeddings shape: {embeddings.shape}")

# Load metadata
print("Loading metadata...")
with open('/home/schnable/Documents/fieldLeafImaging/src/autoencoder_no_weighting/embeddings/embeddings_metadata.json', 'r') as f:
    metadata = json.load(f)

# Load splits to create image_path -> split mapping
print("Loading splits...")
with open('/home/schnable/Documents/fieldLeafImaging/src/autoencoder_no_weighting/logs/image_splits.json', 'r') as f:
    splits = json.load(f)

# Create mapping of image_path to split
image_to_split = {}
for split_name in ['train', 'test', 'val']:
    for item in splits[split_name]:
        image_to_split[item['image_path']] = split_name

# Create DataFrame
print("Creating DataFrame...")
data = {
    'image_path': [item['image_path'] for item in metadata],
    'split': [image_to_split[item['image_path']] for item in metadata]
}

# Add latent variables as columns
for i in range(embeddings.shape[1]):
    data[f'latent_{i:03d}'] = embeddings[:, i]

df = pd.DataFrame(data)

# Save to CSV
output_path = '/home/schnable/Documents/fieldLeafImaging/src/autoencoder_no_weighting/embeddings/embeddings.csv'
print(f"Saving to {output_path}...")
df.to_csv(output_path, index=False)

print(f"\n✓ Successfully saved embeddings to CSV!")
print(f"  Rows: {len(df)}")
print(f"  Columns: {len(df.columns)}")
print(f"  File: {output_path}")
