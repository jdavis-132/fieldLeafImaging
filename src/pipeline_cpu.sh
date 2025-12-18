#!/bin/bash
set -e  # Exit on error
set -x  # Print commands (for debugging)

# manual phenos
#Rscript src/R/LV_quantGen.R -manual -data/manual/scores_813.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -score_average -scores_813 -0.01 data/SbDiv_Mangal2025_genotypes.txt > logs/scores_813_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -manual -data/manual/scores_828.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -score_average -scores_828 -0.01 data/SbDiv_Mangal2025_genotypes.txt > logs/scores_828_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -manual -data/manual/SbDiv_ne2025_FT_clean.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -days_to_flower -ft -0.01 data/SbDiv_Mangal2025_genotypes.txt > logs/ft_quantgen.Rout 2>&1

# run quantgen
#Rscript src/R/LV_quantGen.R -image -output/dinov2_features.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -feature -dinov2 -0.01 data/SbDiv_Mangal2025_genotypes.txt > logs/dinov2_quantgen.Rout 2>&1
# train and cross val RFs
python src/random_forest/main5CV.py data/manual/scores_813.csv output/dinov2_rf_predictors.csv --join_column plotNumber --predictor_prefix feature --output_prefix output/dinov2_features_813_
python src/random_forest/main5CV.py data/manual/scores_813.csv output/dinov2_rf_predictors.csv --join_column plotNumber --predictor_prefix PC --output_prefix output/dinov2_PC_813_
python src/random_forest/main5CV.py data/manual/scores_828.csv output/dinov2_rf_predictors.csv --join_column plotNumber --predictor_prefix feature --output_prefix output/dinov2_features_828_
python src/random_forest/main5CV.py data/manual/scores_828.csv output/dinov2_rf_predictors.csv --join_column plotNumber --predictor_prefix PC --output_prefix output/dinov2_PC_828_

Rscript src/R/LV_quantGen.R -image -output/sam3_embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -embedding -sam3 -0.01 data/SbDiv_Mangal2025_genotypes.txt > logs/sam3_quantgen.Rout 2>&1
python src/random_forest/main5CV.py data/manual/scores_813.csv output/sam3_rf_predictors.csv --join_column plotNumber --predictor_prefix embedding --output_prefix output/sam3_features_813_
python src/random_forest/main5CV.py data/manual/scores_813.csv output/sam3_rf_predictors.csv --join_column plotNumber --predictor_prefix PC --output_prefix output/sam3_PC_813_
python src/random_forest/main5CV.py data/manual/scores_828.csv output/sam3_rf_predictors.csv --join_column plotNumber --predictor_prefix embedding --output_prefix output/sam3_features_828_
python src/random_forest/main5CV.py data/manual/scores_828.csv output/sam3_rf_predictors.csv --join_column plotNumber --predictor_prefix PC --output_prefix output/sam3_PC_828_
