library(tidyverse)
library(paletteer)
source('src/R/Functions.R')

# model_list <- c('sam3_mean', 'sam3_std', 'sam3', 'dinov2', 'ae1', 'ae3', 'ae4', 'ae5', 'ae6', 'ae7', 'ae8', 'sam3_mean',
#                 'sam3_std', 'sam3', 'dinov2', 'ae1', 'ae3', 'ae4', 'ae5', 'ae6', 'ae7', 'ae8', 'sam3_mean_rs',
#                 'sam3_std_rs', 'sam3_rs', 'dinov2_rs', 'ae1_rs', 'ae3_rs', 'ae4_rs', 'ae5_rs', 'ae6_rs', 'ae7_rs',
#                 'ae8_rs', 'sam3_mean_rs', 'sam3_std_rs', 'sam3_rs', 'dinov2_rs', 'ae1_rs', 'ae3_rs', 'ae4_rs', 
#                 'ae5_rs', 'ae6_rs', 'ae7_rs', 'ae8_rs')
# predictor_list <- c('embedding_mean', 'embedding_std', 'embedding', 'feature', rep('latent_dim', 7), rep('PC', 11), 'embedding_mean', 
#                     'embedding_std', 'embedding', 'feature', rep('latent_dim', 7), rep('PC', 11))
# label_list <- rep(c('all', 'senesced_removed'), each = 22)

model_specs <- read_csv('data/rf_model_specs_20260507.csv')
model_specs <- mutate(model_specs, predictiveAbility = -1)
for(i in 1:nrow(model_specs))
{
  df <- read_csv(str_c('output/rf_20260507/', model_specs$model[i], '_', model_specs$label[i], '_', model_specs$predictor_prefix[i], '_predictions_rf.csv'))
  model_specs$predictiveAbility[i] = getRFPredictability(df, 
                      model_descriptor = str_c(model_specs$model[i], model_specs$label[i], model_specs$predictor_prefix[i], sep = ':'))
  
}




