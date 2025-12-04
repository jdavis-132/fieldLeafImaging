conda activate jupyterlab-debugger

python src/autoencoder/preprocess_data.py -i data/ne2025/device1 -o data/processed/ne2025/device1 --start_step normalize
python src/autoencoder/preprocess_data.py -i data/ne2025/device2 -o data/processed/ne2025/device2 --start_step normalize
python src/autoencoder/preprocess_data.py -i data/ne2025/device3 -o data/processed/ne2025/device3 --start_step normalize
python src/autoencoder/preprocess_data.py -i data/ne2025/device4 -o data/processed/ne2025/device4 --start_step normalize
python src/autoencoder/preprocess_data.py -i data/ne2025/device5 -o data/processed/ne2025/device5 --start_step normalize
python src/autoencoder/preprocess_data.py -i data/ne2025/device6 -o data/processed/ne2025/device6 --start_step normalize
python src/autoencoder/preprocess_data.py -i data/ne2025/device7 -o data/processed/ne2025/device7 --start_step normalize
python src/autoencoder/preprocess_data.py -i data/ne2025/device8 -o data/processed/ne2025/device8

python src/autoencoder/preprocess_data.py -i data/aamu2025/block1 -o data/processed/aamu2025/block1
python src/autoencoder/preprocess_data.py -i data/aamu2025/block2 -o data/processed/aamu2025/block2

python src/autoencoder/preprocess_data.py -i data/fvsu2025/sap1 -o data/processed/fvsu2025/sap1
python src/autoencoder/preprocess_data.py -i data/fvsu2025/sap2 -o data/processed/fvsu2025/sap2
python src/autoencoder/preprocess_data.py -i data/fvsu2025/sap3 -o data/processed/fvsu2025/sap3
python src/autoencoder/preprocess_data.py -i data/fvsu2025/bap1 -o data/processed/fvsu2025/bap1
python src/autoencoder/preprocess_data.py -i data/fvsu2025/bap2 -o data/processed/fvsu2025/bap2
python src/autoencoder/preprocess_data.py -i data/fvsu2025/bap3 -o data/processed/fvsu2025/bap3
