library(tidyverse)
library(paletteer)
source('src/R/Functions.R')

model_list <- c('sam3_mean', 'sam3_std', 'sam3', 'dinov2', 'ae1', 'ae3', 'ae4', 'ae5', 'ae6', 'ae7', 'ae8', 'sam3_mean',
                'sam3_std', 'sam3', 'dinov2', 'ae1', 'ae3', 'ae4', 'ae5', 'ae6', 'ae7', 'ae8', 'sam3_mean_rs',
                'sam3_std_rs', 'sam3_rs', 'dinov2_rs', 'ae1_rs', 'ae3_rs', 'ae4_rs', 'ae5_rs', 'ae6_rs', 'ae7_rs',
                'ae8_rs', 'sam3_mean_rs', 'sam3_std_rs', 'sam3_rs', 'dinov2_rs', 'ae1_rs', 'ae3_rs', 'ae4_rs', 
                'ae5_rs', 'ae6_rs', 'ae7_rs', 'ae8_rs')
predictor_list <- c('embedding_mean', 'embedding_std', 'embedding', 'feature', rep('latent_dim', 7), rep('PC', 11), 'embedding_mean', 
                    'embedding_std', 'embedding', 'feature', rep('latent_dim', 7), rep('PC', 11))
label_list <- rep(c('all', 'senesced_removed'), each = 22)
  
  
  
  
for(i in c(1:44))
{
  df <- read_csv(str_c('output/rf/', model_list[i], '_', predictor_list[i], '_pctd_', label_list[i], '_predictions_rf.csv'))
  getRFPredictability(df, model_descriptor = str_c(model_list[i], predictor_list[i], sep = ':'))
}




