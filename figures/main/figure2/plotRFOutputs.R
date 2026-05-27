library(tidyverse)
library(cowplot)
library(paletteer)
source('src/R/Functions.R')

theme_use <- theme_minimal() +
  theme(axis.text.x = element_text(size = 9, color = 'black', margin = margin(0, 0, 0, 0), 
                                   vjust = 0.5, hjust = 0.5),
        axis.text.y = element_text(size = 9, color = 'black', vjust = 0.5, hjust = 0.5),
        legend.text = element_text(size = 9, color = 'black', vjust = 0.5, hjust = 0.5),
        plot.title = element_text(size = 9, color = 'black', vjust = 0, hjust = 0.5),
        plot.subtitle = element_text(size = 9, color = 'black', vjust = 0, hjust = 0.5),
        text = element_text(size = 9, color = 'black'),
        legend.position = 'top',
        line = element_line(color = 'black', linewidth = 1),
        axis.ticks = element_line(color = 'black', linewidth = 0.5),
        axis.line.x.bottom = element_line(color = 'black', linewidth = 0.5),
        axis.line.y.left = element_line(color = 'black', linewidth = 0.5),
        panel.grid = element_blank(), 
        panel.background = element_blank())

model_specs <- read_csv('data/rf_model_specs_20260524.csv')
model_specs <- mutate(model_specs, predictiveAbility = -1)
model_specs <- model_specs[1:13, ]
for(i in 1:nrow(model_specs))
{
  df <- read_csv(str_c('output/rf_20260524/', model_specs$model[i], '_', model_specs$label[i], '_', model_specs$predictor_prefix[i], '_predictions_rf.csv'))
  model_specs$predictiveAbility[i] = getRFPredictability(df, 
                                                         model_descriptor = str_c(model_specs$model[i], model_specs$label[i], model_specs$predictor_prefix[i], sep = ':'))
  
}

model_order <- c('ae1', 'ae2', 'ae3', 'ae4', 'ae5', 'ae6', 'ae7', 
                 'dinov2_mean', 'dinov2_std', 'dinov2', 'sam3_mean', 'sam3_std', 'sam3')
model_specs <- mutate(model_specs, 
                         model = factor(model, 
                                        levels = model_order, 
                                        labels = c('AE1', 'AE2', 'AE3', 'AE4', 'AE5', 'AE6', 'AE7', 
                                                   'DINOv2 Mean', 'DINOv2 SD', 'DINOv2 All','SAM3 Mean', 'SAM3 SD', 'SAM3 All')))

predictability_bars <- ggplot(model_specs, aes(model, predictiveAbility, fill = model)) + 
  geom_col() + 
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0), name = expression("Spearman "~ R^2)) +
  scale_fill_manual(values = c(paletteer_d("ggsci::default_gsea", 12)[7:12], 
                               paletteer_d("dichromat::DarkRedtoBlue_12", 12)[c(12, 4, 3, 1)], 
                               paletteer_d("ggsci::default_gsea", 12)[c(5, 3, 1)])) +
  labs(x = NULL) +
  theme_use + 
  theme(axis.text.x = element_text(angle = 90), 
        legend.position = 'none')
predictability_bars
ggsave(filename = 'figures/main/figure2/predictability_bars.svg', plot = predictability_bars, width = 3.3, height = 1.85, units = 'in', dpi = 1e3, bg = 'transparent')

sam3_rf_predictions_rs <- read_csv('output/rf_20260524/sam3_human_scores_embedding_predictions_rf.csv')

rf_predictability <- ggplot(sam3_rf_predictions_rs, aes(label, predicted)) + 
  geom_point(color = paletteer_d("ggsci::default_gsea", 1)[1], alpha = 0.25) + 
  geom_abline(intercept = 0, slope = 1, color = 'black') + 
  geom_smooth(method = 'lm', linetype = 'dashed', se = FALSE, color = 'black') + 
  scale_x_continuous(expand = c(0, 0), limits = c(0.75, 5)) + 
  scale_y_continuous(expand = c(0, 0), limits = c(0.75, 5)) +
  labs(x = 'Human Score (Mean)', 
       y = 'Predicted Human Score') + 
  theme_use
rf_predictability

fi_pctd_features <- read_csv('output/rf_20260524/sam3_human_scores_embedding_feature_importances_rf.csv') %>% 
  pivot_longer(cols = everything(), names_to = 'feature', values_to = 'fi') %>% 
  group_by(feature) %>%
  summarise(avg_fi = mean(fi, na.rm = TRUE)) %>% 
  arrange(desc(avg_fi)) %>% 
  mutate(feature = as.numeric(feature)) %>% 
  mutate(stat = case_when(feature < 1024 ~ 'mean', .default = 'std'), 
         embedding_num = case_when(feature > 1023 ~ feature - 1024, .default = feature), 
         embedding = str_c('embedding_', stat, '_', embedding_num))

fi_plot <- ggplot(fi_pctd_features, aes(avg_fi)) + 
  geom_histogram(fill = paletteer_d("ggsci::default_gsea", 1)[1]) +
  geom_segment(x = 0.003, xend=0.003, y=0, yend=200, color = 'black') +
  labs(x = 'Mean Feature Importance \n(RF 5-fold CV)', 
       y = 'SAM3 Embededings') +
  scale_x_continuous(expand = c(0, 0)) + 
  scale_y_continuous(expand = c(0, 0)) +
  theme_use
fi_plot

sam3_rf <- plot_grid(rf_predictability, fi_plot, ncol = 2, labels = c('c', 'd'))
ggsave(filename = 'figures/main/figure2/sam3_rf.svg', plot = sam3_rf, width = 6.5, height = 3.3, units = 'in', dpi = 1e3, bg = 'transparent')
