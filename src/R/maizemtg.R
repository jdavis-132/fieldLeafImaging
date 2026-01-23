library(tidyverse)
source('src/R/Functions.R')
source('../parallelgwas/manhattanPlot.R')
preds_pctd_features <- read_csv('output/dinov2_features_pctd_predictions_rf.csv')
fi_pctd_features <- read_csv('output/dinov2_features_pctd_feature_importances_rf.csv')[, 2:385] %>% 
  pivot_longer(cols = everything(), names_to = 'feature', values_to = 'fi') %>% 
  group_by(feature) %>%
  summarise(avg_fi = mean(fi, na.rm = TRUE)) %>% 
  arrange(desc(avg_fi)) %>% 
  mutate(feature = as.numeric(feature))


pred_obs <- ggplot(preds_pctd_features, aes(label, predicted)) + 
  geom_point() +
  geom_smooth(method = 'lm')
pred_obs

preds_pctd_pcs <- read_csv('output/dinov2_PC_pctd_predictions_rf.csv')



rf_preds <- rf_preds %>% 
  dplyr::mutate(image_name = str_split_i(image_path, fixed('/'), 6) %>% 
           str_split_remove_i('_', 7) %>% 
           str_c('_leaf.png'), 
         device = stringr::str_split_i(image_path, fixed('/'), 4)) %>% 
  mutate(image_name = str_c(device, '_', image_name))

pct_disease <- read_csv('output/ne2025_segmented2_exg_p20_results.csv') %>% 
  mutate(image_name = str_c(device, '_', image_name)) %>% 
  rename(score_average = ExG_P20_disease_pct) %>% 
  mutate(plotNumber = str_split_i(image_name, '_', 2) %>% 
           as.numeric())
eff_markers_dinov2 <- 49957
dinov2_thresh <- 0.05/eff_markers_dinov2
signals_dinov2PCs <- summariseSignals('output/gwas/dinov2_PC*')
signals_dinov2PCs <- signals_dinov2PCs[signals_dinov2PCs$pval < dinov2_thresh, ]

pmap_dinov2features <- summariseSignals('output/gwas/feature*')
pmap_dinov2features <- pmap_dinov2features %>% 
  mutate(feature = str_split_i(filename, '_', 2) %>%
           str_remove('.MLM.csv') %>% 
           as.numeric())
signals_dinov2features <- pmap_dinov2features[pmap_dinov2features$pval < dinov2_thresh, ]
top38_pmap <- filter(pmap_dinov2features, feature %in% c(51, 70, 94, 198, 284, 321))
top38_pmap <- mutate(top38_pmap, CHROM = as.numeric(CHROM))
top38_manhattan <- plotManhattan(top38_pmap, pval, multitrait = TRUE, trait = feature, resampling = FALSE, threshold = -log10(dinov2_thresh), main = 'Top 10% of DINOv2 features', species = 'sorghum')
signals_top38 <- filter(signals_dinov2features, feature %in% c(51, 70, 94, 198, 284, 321))

leaf_width <- read_csv('../../Downloads/phenotype_data_COMPLETE_v2_9_FINAL.csv') %>% 
  mutate(plotNumber = str_split_i(image_name, '_', 1) %>% 
           as.numeric()) %>% 
  select(image_name, plotNumber, leaf_width_corrected_px)
fvsu_field_idx <- read_csv('../FVSU_SAP_BAP.csv') %>% 
  rename(plotNumber = `Plot No`, 
         genotype = Accession) %>% 
  mutate(block = str_c(Population, Replication, sep = ' '))
aamu_field_idx <- read_csv('../../Desktop/2025 Sorghum_Update_073125-XK_Index.csv') %>% 
  rename(plotNumber = `Plot Number`,
         genotype = `Variety Name`,
         block = Block) %>% 
  mutate(block = as.character(block))

fvsu_width <- left_join(fvsu_field_idx, leaf_width, join_by(plotNumber)) %>% 
  mutate(environment = 'fvsu')

aamu_width <- left_join(aamu_field_idx, leaf_width, join_by(plotNumber)) %>% 
  mutate(environment = 'aamu')

width <- bind_rows(fvsu_width, aamu_width)

vp <- partitionVariance3(width, 'leaf_width_corrected_px', 'leaf width', '~ (1|environment) + (1|block) + (1|genotype)')





