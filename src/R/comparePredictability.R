library(tidyverse)
source('src/R/Functions.R')

model_list <- c('dinov2', 'dinov2', 'sam3', 'sam3', 'ae1', 'ae5', 'ae5', 'ae7')
predictor_list <- c('features', 'PC', 'embedding', 'PC', 'latent_dim', 'latent_dim', 'PC', 'PC')
model_descriptors <- c('DINOv2\nFeatures', 'DINOv2\nFeature PCs', 'SAM3\nEmbeddings', 'SAM3\nEmbedding PCs', 'AE1\nLVs', 'AE5\nLVs', 
                       'AE5\nLV PCs', 'AE7\nLV PCs')

for(i in 1:length(model_list))
{
  df <- read_csv(str_c('output/', model_list[i], '_', predictor_list[i], '_pctd_predictions_rf.csv'))
  getRFPredictability(df, model_descriptor = model_descriptors[i])
}
