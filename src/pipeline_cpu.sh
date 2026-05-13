#!/bin/bash
set -e  # Exit on error
set -x  # Print commands (for debugging)

# manual phenos
#Rscript src/R/LV_quantGen.R -manual -data/manual/scores_813.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -score_average -scores_813 -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_all.csv > logs/scores_813_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -manual -data/manual/scores_828.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -score_average -scores_828 -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_all.csv > logs/scores_828_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -manual -data/manual/SbDiv_ne2025_FT_clean.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -days_to_flower -ft -0.01 -data/ne2025/genotype_alignment_reseq.csv > logs/ft_quantgen.Rout 2>&1

# run quantgen
#Rscript src/R/LV_quantGen.R -image -output/dinov2_features.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -feature -dinov2 -0.01 data/SbDiv_Mangal2025_genotypes.txt > logs/dinov2_quantgen.Rout 2>&1
# train and cross val RFs
#python src/random_forest/main5CV.py data/manual/scores_813.csv output/dinov2_rf_predictors.csv --join_column plotNumber --predictor_prefix feature --output_prefix output/dinov2_features_813_
#python src/random_forest/main5CV.py data/manual/scores_813.csv output/dinov2_rf_predictors.csv --join_column plotNumber --predictor_prefix PC --output_prefix output/dinov2_PC_813_
#python src/random_forest/main5CV.py data/manual/scores_828.csv output/dinov2_rf_predictors.csv --join_column plotNumber --predictor_prefix feature --output_prefix output/dinov2_features_828_
#python src/random_forest/main5CV.py data/manual/scores_828.csv output/dinov2_rf_predictors.csv --join_column plotNumber --predictor_prefix PC --output_prefix output/dinov2_PC_828_

# Rscript src/R/LV_quantGen.R -image -output/sam3_embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -embedding -sam3 -0.01 -data/ne2025/genotype_alignment_reseq.csv > logs/sam3_quantgen.Rout 2>&1
#python src/random_forest/main5CV.py data/manual/scores_813.csv output/sam3_rf_predictors.csv --join_column plotNumber --predictor_prefix embedding --output_prefix output/sam3_features_813_
#python src/random_forest/main5CV.py data/manual/scores_813.csv output/sam3_rf_predictors.csv --join_column plotNumber --predictor_prefix PC --output_prefix output/sam3_PC_813_
#python src/random_forest/main5CV.py data/manual/scores_828.csv output/sam3_rf_predictors.csv --join_column plotNumber --predictor_prefix embedding --output_prefix output/sam3_features_828_
#python src/random_forest/main5CV.py data/manual/scores_828.csv output/sam3_rf_predictors.csv --join_column plotNumber --predictor_prefix PC --output_prefix output/sam3_PC_828_

#python src/random_forest/main5CV.py output/ne2025_exg_percent_disease_area.csv output/dinov2_rf_predictors.csv --join_column image_name --predictor_prefix feature --output_prefix output/dinov2_features_pctd_
#python src/random_forest/main5CV.py output/ne2025_exg_percent_disease_area.csv output/dinov2_rf_predictors.csv --join_column image_name --predictor_prefix PC --output_prefix output/dinov2_PC_pctd_
#python src/random_forest/main5CV.py output/ne2025_exg_percent_disease_area.csv output/sam3_rf_predictors.csv --join_column image_name --predictor_prefix feature --output_prefix output/sam3_features_pctd_
#python src/random_forest/main5CV.py output/ne2025_exg_percent_disease_area.csv output/sam3_rf_predictors.csv --join_column image_name --predictor_prefix PC --output_prefix output/sam3_PC_pctd_

#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260107_143328_standard_lr0.001_bs32_l1/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae1 -0.01 -data/SbDiv_Mangal2025_genotypes.txt > logs/ae1_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260107_205445_vae_lr0.001_bs32_vae/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae2 -0.01 -data/SbDiv_Mangal2025_genotypes.txt > logs/ae2_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260107_214228_standard_lr0.001_bs32_l1/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae3 -0.01 -data/SbDiv_Mangal2025_genotypes.txt > logs/ae3_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20251217_155126_standard_lr0.001_bs32_l1_attention/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae4 -0.01 -data/SbDiv_Mangal2025_genotypes.txt > logs/ae4_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260108_183222_standard_lr0.001_bs32_l1_attention/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae5 -0.01 -data/SbDiv_Mangal2025_genotypes.txt > logs/ae5_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260109_205057_standard_lr0.001_bs32_l1/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae7 -0.01 -data/SbDiv_Mangal2025_genotypes.txt > logs/ae7_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260110_045517_standard_lr0.001_bs32_disease_weighted_l1/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae8 -0.01 -data/SbDiv_Mangal2025_genotypes.txt > logs/ae8_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260108_012555_standard_lr0.001_bs32_l1_attention/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -laten#t_dim -ae4 -0.01 -data/SbDiv_Mangal2025_genotypes.txt > logs/ae4_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260109_034107_standard_lr0.001_bs32_l1_attention/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae6 -0.01 -data/SbDiv_Mangal2025_genotypes.txt > logs/ae6_quantgen.Rout 2>&1

# with all kept images
#Rscript src/R/LV_quantGen.R -image -output/sam3_embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -embedding_mean -sam3_mean -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_all.csv > logs/sam3_mean_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -output/sam3_embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -embedding_std -sam3_std -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_all.csv > logs/sam3_std_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -output/sam3_embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -embedding -sam3 -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_all.csv > logs/sam3_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -manual -data/ne2025/pctd_all.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -pctd -pctd -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_all.csv > logs/pctd_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -output/dinov2_features.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -feature -dinov2 -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_all.csv > logs/dinov2_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260107_143328_standard_lr0.001_bs32_l1/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae1 -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_all.csv > logs/ae1_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260107_214228_standard_lr0.001_bs32_l1/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae3 -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_all.csv > logs/ae3_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260108_012555_standard_lr0.001_bs32_l1_attention/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae4 -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_all.csv > logs/ae4_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260108_183222_standard_lr0.001_bs32_l1_attention/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae5 -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_all.csv > logs/ae5_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260109_034107_standard_lr0.001_bs32_l1_attention/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae6 -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_all.csv > logs/ae6_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260109_205057_standard_lr0.001_bs32_l1/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae7 -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_all.csv > logs/ae7_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260110_045517_standard_lr0.001_bs32_disease_weighted_l1/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae8 -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_all.csv > logs/ae8_quantgen.Rout 2>&1

# remove senesced leaves
#Rscript src/R/LV_quantGen.R -image -output/sam3_embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -embedding_mean -sam3_mean_rs -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_senesced_removed.csv > logs/sam3_mean_rs_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -output/sam3_embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -embedding_std -sam3_std_rs -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_senesced_removed.csv > logs/sam3_std_rs_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -output/sam3_embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -embedding -sam3_rs -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_senesced_removed.csv > logs/sam3_rs_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -manual -data/ne2025/pctd_senesced_removed.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -pctd -pctd_rs -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_senesced_removed.csv > logs/pctd_rs_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -output/dinov2_features.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -feature -dinov2_rs -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_senesced_removed.csv > logs/dinov2_rs_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260107_143328_standard_lr0.001_bs32_l1/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae1_rs -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_senesced_removed.csv > logs/ae1_rs_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260107_214228_standard_lr0.001_bs32_l1/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae3_rs -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_senesced_removed.csv > logs/ae3_rs_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260108_012555_standard_lr0.001_bs32_l1_attention/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae4_rs -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_senesced_removed.csv > logs/ae4_rs_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260108_183222_standard_lr0.001_bs32_l1_attention/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae5_rs -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_senesced_removed.csv > logs/ae5_rs_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260109_034107_standard_lr0.001_bs32_l1_attention/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae6_rs -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_senesced_removed.csv > logs/ae6_rs_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260109_205057_standard_lr0.001_bs32_l1/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae7_rs -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_senesced_removed.csv > logs/ae7_rs_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -models/autoencoder_20260110_045517_standard_lr0.001_bs32_disease_weighted_l1/embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -latent_dim -ae8_rs -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_senesced_removed.csv > logs/ae8_rs_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -output/rf/sam3_rs_embedding_pctd_senesced_removed_predictions_rf.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -predicted -pctd_rs_pred -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_senesced_removed.csv > logs/pctd_rs_pred_quantgen.Rout 2>&1

#Rscript src/R/LV_quantGen.R -image -data/ne2025/phenotype_data_NEBRASKA_FINAL_GWAS.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -leaf_width_corrected_px -leaf_width -0.01 -data/ne2025/genotype_alignment_reseq.csv -data/ne2025/images_keep_senesced_removed.csv >logs/leaf_width_quantgen.Rout 2>&1

#FVSU + AAMU embeddings with intercept added back
Rscript src/R/LV_quantGen.R -image -output/sam3_embeddings_fvsu.csv -data/fvsu2025/fvsu_field_index.csv -embedding -sam3_fvsu -0.01 -data/genotype_conversion_table.tsv -data/fvsu2025/image_ids_keep.txt >logs/fvsu_quantgen.Rout 2>&1
Rscript src/R/LV_quantGen.R -image -output/sam3_embeddings_aamu.csv -data/aamu2025/aamu_field_index.csv -embedding -sam3_aamu -0.01 -data/genotype_conversion_table.tsv -data/aamu2025/image_ids_keep.txt >logs/aamu_quantgen.Rout 2>&1
#Rscript src/R/LV_quantGen.R -image -output/sam3_embeddings.csv -data/ne2025/SbDiv_ne2025_fieldindex.csv -embedding -sam3  -0.01 -data/genotype_conversion_table.tsv -data/ne2025/images_keep_all.csv > logs/sam3_intercept_quantgen.Rout 2>&1
