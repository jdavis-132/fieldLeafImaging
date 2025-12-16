#!/bin/bash
set -e  # Exit on error
set -x  # Print commands (for debugging)

Rscript src/R/LV_quantGen.R -image -output/dinov2_features.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -feature -dinov2 -0.01 data/SbDiv_Mangal2025_genotypes.txt > logs/dinov2_quantgen.Rout 2>&1

#Rscript src/R/LV_quantGen.R -image -output/sam3_embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -embedding -sam3 -0.01 data/SbDiv_Mangal2025_genotypes.txt > logs/sam3_quantgen.Rout 2>&1

Rscript src/R/LV_quantGen.R -manual -data/manual/scores_813.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -score_average -scores_813 -0.01 data/SbDiv_Mangal2025_genotypes.txt > logs/scores_813_quantgen.Rout 2>&1

Rscript src/R/LV_quantGen.R -manual -data/manual/scores_828.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -score_average -scores_828 -0.01 data/SbDiv_Mangal2025_genotypes.txt > logs/scores_828_quantgen.Rout 2>&1
