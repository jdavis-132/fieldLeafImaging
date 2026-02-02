library(tidyverse)
source('src/R/Functions.R')
library(paletteer)
library(ggcorrplot)
source('../parallelgwas/manhattanPlot.R')

theme_use <- theme_minimal() +
  theme(axis.text.x = element_text(size = 9, color = 'black', margin = margin(0, 0, 0, 0), 
                                   vjust = 0.5, hjust = 0.5),
        axis.text.y = element_text(size = 9, color = 'black', vjust = 0, hjust = 0.5),
        legend.text = element_text(size = 9, color = 'black', vjust = 0.5, hjust = 0.5),
        plot.title = element_text(size = 9, color = 'black', vjust = 0, hjust = 0.5),
        plot.subtitle = element_text(size = 9, color = 'black', vjust = 0, hjust = 0.5),
        text = element_text(size = 9, color = 'black'),
        legend.position = 'top',
        line = element_line(color = 'black', linewidth = 1),
        axis.line.x.bottom = element_line(color = 'black', linewidth = 0.5),
        axis.line.y.left = element_line(color = 'black', linewidth = 0.5),
        panel.grid = element_blank())

preds_pctd_features <- read_csv('output/rf/predictions/sam3_embedding_pctd_all_predictions_rf.csv')
spearman_r2 <- cor(preds_pctd_features[['label']], preds_pctd_features[['predicted']], use = 'complete.obs', method = 'spearman')^2
sam3_preds_plot <- ggplot(preds_pctd_features, aes(label, predicted)) + 
  geom_point(color = paletteer_d('MetBrewer::Archambault', 1)) +
  annotate(geom = 'text', x = 10, y=35, label = str_c('R^2==', spearman_r2), parse = TRUE) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 90)) + 
  scale_y_continuous(expand = c(0, 0), limits = c(0, 40)) +
  labs(x = 'Percent Diseased Area\n(ExG Threshold)', 
       y = 'Predicted Percent Disease Area\n(RF)', 
       title = 'SAM3\nEmbedding Means & Standard Deviations') + 
  theme_use 
sam3_preds_plot
ggsave('output/sam3_pred_accuracy.png', dpi = 2000, plot = sam3_preds_plot)
pctd <- read_csv('output/ne2025_exg_percent_disease_area.csv')
high_err_preds <- preds_pctd_features %>% 
  filter(label > 75) %>% 
  filter(predicted < 10) 

fi_pctd_features <- read_csv('output/rf/sam3_rs_embedding_pctd_senesced_removed_feature_importances_rf.csv')%>% 
  pivot_longer(cols = everything(), names_to = 'feature', values_to = 'fi') %>% 
  group_by(feature) %>%
  summarise(avg_fi = mean(fi, na.rm = TRUE)) %>% 
  arrange(desc(avg_fi)) %>% 
  mutate(feature = as.numeric(feature)) %>% 
  mutate(stat = case_when(feature < 1024 ~ 'mean', .default = 'std'), 
         embedding_num = case_when(feature > 1023 ~ feature - 1024, .default = feature))

fi_plot <- ggplot(fi_pctd_features, aes(avg_fi)) + 
  geom_histogram(fill = paletteer_d('MetBrewer::Archambault', 6)[3]) +
  labs(x = 'Average Feature Importance (RF 5-fold CV)', 
       y = 'Number of SAM3 Embededings') +
  scale_x_continuous(expand = c(0, 0)) + 
  scale_y_continuous(expand = c(0, 0)) +
  theme_use
fi_plot
ggsave('output/FI_distribution.png', dpi = 2000, plot = fi_plot, height = 2.75, width = 5)


high_fi <- fi_pctd_features %>% 
  arrange(desc(avg_fi))
high_fi <- high_fi[1:30, ]

high_fi_features <- str_c('embedding', high_fi$stat, high_fi$embedding_num, sep = '_')

embeddings <- read_csv('output/rf/sam3_rs_rf_predictors.csv')
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

feature_blues <- read_csv('output/sam3_rs_blues.csv')

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
  samplePhenotypesForResampling('output/sam3_rs_blues.csv', 'genotype', feature)
}

# all_farmcpu_hits <- summariseSignals_PANICLE('output/gwas/sam3/farmcpu/GWAS_embedding*') %>%
#   mutate(stat = str_split_i(filename, '_', 3),
#          feature = str_split_i(filename, '_', 4) %>%
#            as.numeric(),
#          iter = str_split_i(filename, '_', 5) %>%
#            as.numeric())
# write_csv(all_farmcpu_hits, 'output/sam3_high_fi_allfarmcpuhits.csv')

all_farmcpu_hits <- read_csv('output/sam3_high_fi_allfarmcpuhits.csv')

rmip <- all_farmcpu_hits %>% 
  group_by(SNP, CHROM, POS, feature, stat) %>% 
  summarise(RMIP = n()/100, 
            min_p = min(pval, na.rm = TRUE), 
            mean_effect = mean(effect, na.rm = TRUE)) %>% 
  mutate(embedding = str_c('embedding', stat, feature, sep = '_')) %>% 
  arrange(desc(RMIP))

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

select_features <- intersect(head(rmip$embedding, n=15), head(high_fi_features, n=15))
n_features <- length(select_features)
rmip_selected <- rmip %>% 
  filter(embedding %in% select_features) %>% 
  ungroup() %>% 
  mutate(CHROM = str_remove(CHROM, 'Chr') %>% 
           as.numeric(), 
         label = str_c(feature, '\n(', str_to_title(stat), ')'))

plotManhattan(rmip_selected, RMIP, multitrait = TRUE, trait = label, threshold = 0.2, 
              colors = paletteer_d('MetBrewer::Archambault', n_features),
              species = 'sorghum', theme = theme_use)
ggsave('output/selected_embeddings_farmcpu.png', width = 5, height = 2.5, dpi = 1000, 
       bg = 'transparent')
