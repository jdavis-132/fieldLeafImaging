library(tidyverse)
source('src/R/Functions.R')
library(paletteer)
library(ggcorrplot)
source('../parallelgwas/manhattanPlot.R')
preds_pctd_features <- read_csv('output/sam3_embedding_pctd_predictions_rf.csv')
spearman_r2 <- cor(preds_pctd_features[['label']], preds_pctd_features[['predicted']], use = 'complete.obs', method = 'spearman')^2
sam3_preds_plot <- ggplot(preds_pctd_features, aes(label, predicted)) + 
  geom_abline(color = paletteer_d('fishualize::Chlorurus_microrhinos', 2)[2], size = 1) +
  # geom_bin2d() + 
  geom_point(color = paletteer_d('fishualize::Chlorurus_microrhinos', 1)) +
  annotate(geom = 'text', x = 10, y=35, label = str_c('R^2==', spearman_r2), parse = TRUE) +
  # scale_fill_viridis(direction = -1) +
  # guides(fill = guide_colorbar(barwidth = 10)) +
  labs(x = 'Observed Percent Diseased Area\n(ExG Threshold)', 
       y = 'Predicted Percent Disease Area\n(RF)', 
       title = 'SAM3\nEmbeddings') + 
  theme_use 
sam3_preds_plot
ggsave('output/sam3_pred_accuracy.png', dpi = 2000, plot = sam3_preds_plot)
pctd <- read_csv('output/ne2025_exg_percent_disease_area.csv')
high_err_preds <- preds_pctd_features %>% 
  filter(label > 75) %>% 
  filter(predicted < 10) 

fi_pctd_features <- read_csv('output/sam3_embedding_pctd_feature_importances_rf.csv')[, 2:1025] %>% 
  pivot_longer(cols = everything(), names_to = 'feature', values_to = 'fi') %>% 
  group_by(feature) %>%
  summarise(avg_fi = mean(fi, na.rm = TRUE)) %>% 
  arrange(desc(avg_fi)) %>% 
  mutate(feature = as.numeric(feature))

fi_plot <- ggplot(fi_pctd_features, aes(avg_fi)) + 
  geom_histogram(fill = paletteer_d('fishualize::Chlorurus_microrhinos', 4)[4]) +
  labs(x = 'Average Feature Importance (RF 5-fold CV)', 
       y = 'Number of SAM3 Embededings') +
  scale_x_continuous(expand = c(0, 0)) + 
  scale_y_continuous(expand = c(0, 0)) +
  theme_use
fi_plot
ggsave('output/FI_distribution.png', dpi = 2000, plot = fi_plot)


high_fi <- fi_pctd_features %>% 
  arrange(desc(avg_fi))
high_fi <- high_fi[1:30, ]

high_fi_features <- str_c('embedding_', high_fi$feature)

embeddings <- read_csv('output/sam3_rf_predictors.csv')
high_fi_embeddings <- embeddings %>% 
  select(any_of(high_fi_features)) %>% 
  as.matrix()

cor_mat <- cor(high_fi_embeddings)
p <- ggcorrplot(cor_mat, 
           type = 'upper', 
           ggtheme = theme_use, 
           title = 'SAM3 Top 30 Embeddings',
           legend.title = 'Pearson Correlation Coefficient', 
           outline.color = 'transparent', 
           hc.order = TRUE) + 
  theme(axis.text.x = element_text(angle = 90))
print(p)

feature_blues <- read_csv('output/sam3_blues.csv')

for(feature in high_fi_features)
{
  printHistogram(feature_blues, feature, title = feature)
}


blues_winsor <- feature_blues %>% 
  select(c(genotype, high_fi_features))
winsor_strength <- 0.02
for(feature in high_fi_features)
{
  blues_winsor <- winsorize(blues_winsor, feature, winsor_strength, 1 - winsor_strength)
  printHistogram(blues_winsor, feature, title = feature)
}

write_csv(blues_winsor, 'output/sam3_high_fi_blues_winsorized.csv')

for(feature in high_fi_features)
{
  samplePhenotypesForResampling('output/sam3_high_fi_blues_winsorized.csv', 'genotype', feature)
}

# all_farmcpu_hits <- summariseSignals_PANICLE('output/gwas/sam3/farmcpu/GWAS_embedding*') %>%
#   mutate(feature = str_split_i(filename, '_', 3) %>% 
#            as.numeric(), 
#          iter = str_split_i(filename, '_', 4) %>% 
#            as.numeric())
# write_csv(all_farmcpu_hits, 'output/sam3_high_fi_allfarmcpuhits.csv')

all_farmcpu_hits <- read_csv('output/sam3_high_fi_allfarmcpuhits.csv')

rmip <- all_farmcpu_hits %>% 
  group_by(SNP, CHROM, POS, feature) %>% 
  summarise(RMIP = n()/100, 
            min_p = min(pval, na.rm = TRUE), 
            mean_effect = mean(effect, na.rm = TRUE))

rmip_0.1features <- rmip %>% 
  ungroup() %>% 
  filter(RMIP >= 0.10) %>% 
  distinct(feature) %>% 
  pull(feature)
n_features <- length(rmip_0.1features)

rmip_0.1features <- rmip %>% 
  filter(feature %in% rmip_0.1features) %>% 
  ungroup() %>% 
  mutate(CHROM = str_remove(CHROM, 'Chr') %>% 
           as.numeric())

plotManhattan(rmip_0.1features, RMIP, multitrait = TRUE, trait = feature, threshold = 0.2, 
              main = 'SAM3 Top 30 Embeddings\nFarmCPU Resampling (PANICLE)', 
              species = 'sorghum')


panicle_mlm_rnaseq <- summariseSignals_PANICLE('output/gwas/sam3/mlm_rnaseq/GWAS_*_all_results.csv')

# fi_pctd_PC <- read_csv('output/sam3_PC_pctd_feature_importances_rf.csv')[, 2:101] %>% 
#   pivot_longer(cols = everything(), names_to = 'feature', values_to = 'fi') %>% 
#   group_by(feature) %>%
#   summarise(avg_fi = mean(fi, na.rm = TRUE)) %>% 
#   arrange(desc(avg_fi)) %>% 
#   mutate(feature = as.numeric(feature))
# 
# eff_markers <- 49957
# thresh <- 0.05/eff_markers
# thresh_log10 <- -1*log10(thresh)
# 
# sam3_pmap_features <- summariseSignals('output/gwas/sam3/embedding*')
# write_csv(sam3_pmap_features, 'output/sam3_features_pmap.csv')
# sam3_pmap_features <- read_csv('output/sam3_features_pmap.csv')
# sam3_signals_features <- sam3_pmap_features[sam3_pmap_features$`p-val` < thresh, ]
# write_csv(sam3_signals_features, 'output/sam3_features_signals.csv')
# sam3_signals_features <- sam3_signals_features %>% 
#   group_by(filename) %>% 
#   mutate(feature = str_remove(filename, 'output/gwas/sam3/embedding_') %>% 
#            str_remove('.MLM.csv') %>% 
#            as.numeric(), 
#          CHROM = as.numeric(CHROM)) %>% 
#   rename(pval = `p-val`)
# 
# high_fi_signals <- filter(sam3_signals_features, feature %in% high_fi$feature)
# high_fi_pmap <- sam3_pmap_features %>% 
#   rename(pval = `p-val`) %>% 
#   group_by(filename) %>% 
#   mutate(feature = str_remove(filename, 'output/gwas/sam3/embedding_') %>% 
#            str_remove('.MLM.csv') %>% 
#            as.numeric(), 
#          CHROM = as.numeric(CHROM)) %>% 
#   filter(feature %in% high_fi_signals$feature)
# manhattan <- plotManhattan(high_fi_pmap, pval, multitrait = TRUE, trait = feature, resampling = FALSE, main = 'Significant GWAS Signals\nTop 30 SAM3 Features by FI', species = 'sorghum', threshold = thresh_log10)
# 
# sam3_pmap_PC <- summariseSignals('output/gwas/sam3/sam3_PC*')
# sam3_pmap_PC <- sam3_pmap_PC %>% 
#   group_by(filename) %>% 
#   mutate(feature = str_remove(filename, 'output/gwas/sam3/sam3_PC') %>% 
#            str_remove('.MLM.csv') %>% 
#            as.numeric(), 
#          CHROM = as.numeric(CHROM)) %>% 
#   rename(pval = `p-val`)
# write_csv(sam3_pmap_PC, 'output/sam3_PCs_pmap.csv')
# sam3_signals_PC <- sam3_pmap_PC[sam3_pmap_PC$`p-val` < thresh, ]
# write_csv(sam3_signals_PC, 'output/sam3_PCs_signals.csv')
