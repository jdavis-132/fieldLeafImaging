library(tidyverse)
library(paletteer)
source('src/R/Functions.R')

model_list <- c('dinov2', 'dinov2', 'sam3', 'sam3', 'ae1', 'ae1', 'ae5', 'ae5', 'ae7', 'ae7', 'ae8', 'ae8', 'ae3', 'ae3', 'ae4', 'ae4', 
                'ae6', 'ae6')
predictor_list <- c('feature', 'PC', 'embedding', 'PC', 'latent_dim', 'PC', 'latent_dim', 'PC', 'latent_dim', 'PC', 'latent_dim', 'PC',
                    'latent_dim', 'PC', 'latent_dim', 'PC', 'latent_dim', 'PC')
model_descriptors <- c('DINOv2\nFeatures', 'DINOv2\nFeature PCs', 'SAM3\nEmbeddings', 'SAM3\nEmbedding PCs', 'AE1\nLVs', 
                       'AE1\n LV PCs','AE5\nLVs', 'AE5\nLV PCs', 'AE7\nLVs','AE7\nLV PCs', 'AE8\nLVs', 'AE8\nLV PCs', 'AE3\nLVs', 'AE3\nLV PCs', 
                       'AE4\nLVs', 'AE4\nLV PCs', 'AE6\nLVs', 'AE6\nLV PCs')

for(i in c(1:18))
{
  df <- read_csv(str_c('output/', model_list[i], '_', predictor_list[i], '_pctd_predictions_rf.csv'))
  getRFPredictability(df, model_descriptor = model_descriptors[i])
}


