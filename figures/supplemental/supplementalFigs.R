library(tidyverse)
library(paletteer)
library(cowplot)
library(ggcorrplot)
library(readxl)
source('src/R/Functions.R')

theme_use <- theme_minimal() +
  theme(axis.text.x = element_text(size = 9, color = 'black', margin = margin(0, 0, 0, 0), 
                                   vjust = 0.5, hjust = 0.5),
        axis.text.y = element_text(size = 9, color = 'black', vjust = 0.5, hjust = 0.5),
        legend.text = element_text(size = 9, color = 'black', vjust = 0.5, hjust = 0.5),
        plot.title = element_text(size = 9, color = 'black', vjust = 0, hjust = 0.5),
        plot.subtitle = element_text(size = 9, color = 'black', vjust = 0, hjust = 0.5),
        text = element_text(size = 9, color = 'black'),
        legend.position = 'bottom',
        line = element_line(color = 'black', linewidth = 1),
        axis.ticks = element_line(color = 'black', linewidth = 0.5),
        axis.line.x.bottom = element_line(color = 'black', linewidth = 0.5),
        axis.line.y.left = element_line(color = 'black', linewidth = 0.5),
        panel.grid = element_blank(), 
        panel.background = element_blank())

images_keep_unl <- read_csv('data/ne2025/images_keep_all.csv', col_names = c('image_id'))
pctd_ne <- read_csv('data/ne2025/pctd_all.csv') %>%
  mutate(image_id = str_c(str_split_remove_i(image_path, fixed('-'), 4))) %>% 
  filter(image_id %in% images_keep_unl$image_id) %>% 
  mutate(location = 'UNL', 
         plotNumber = str_split_i(image_id, '_', 1) %>% 
           as.numeric())
ne_field_idx <- read_csv('data/ne2025/SbDiv_ne2025_fieldindex.csv') %>% 
  mutate(block = as.numeric(rep)) %>% 
  select(!rep)
pctd_ne <- left_join(pctd_ne, ne_field_idx, join_by(plotNumber))

pctd_fvsu <- read_csv('data/fvsu2025/pctd_all.csv') %>%
  mutate(location = 'FVSU',
         plotNumber = str_split_i(image_path, '_', 1) %>%
           as.numeric(),
         image_id = str_split_remove_i(image_path, fixed('-'), 4)) %>%
  rename(pctd = ExG_P20_disease_pct)
idx_fvsu <- read_csv('data/fvsu2025/fvsu_field_index.csv') %>%
  mutate(block = str_split_i(rep, fixed(' '), 2) %>%
           as.numeric()) %>%
  select(!c(S.No, rep))

pctd_fvsu <- left_join(pctd_fvsu, idx_fvsu, join_by(plotNumber))

pctd_aamu <- read_csv('data/aamu2025/pctd_all.csv') %>%
  mutate(location = 'AAMU',
         plotNumber = str_split_i(image_path, '_', 1) %>%
           as.numeric(),
         image_id = str_split_remove_i(image_path, fixed('-'), 4)) %>%
  rename(pctd = ExG_P20_disease_pct)
idx_aamu <- read_csv('data/aamu2025/aamu_field_index.csv')
pctd_aamu <- left_join(pctd_aamu, idx_aamu, join_by(plotNumber))

genotype_alignment <- read_tsv('data/genotype_conversion_table.tsv', col_names = c('genotype_idx', 'genotype_markers')) %>% 
  mutate(genotype_idx = str_remove_all(genotype_idx, ' '))

pctd_ne_summary <- pctd_ne %>% 
  mutate(genotype = str_remove_all(genotype, ' ')) %>%
  left_join(genotype_alignment, join_by(genotype==genotype_idx)) %>%
  mutate(genotype_markers = case_when(is.na(genotype_markers) ~ genotype,
                                      .default = genotype_markers)) %>% 
  filter(!is.na(genotype_markers)) %>%
  select(!c(genotype)) %>% 
  rename(genotype = genotype_markers) %>% 
  group_by(location, range, row, block, genotype, plotNumber) %>% 
  summarise(pctd = mean(pctd, na.rm = TRUE))

# pctd_ne <- filter(pctd_all, location=='UNL')

pctd_vp <- partitionVariance3(pctd_ne_summary, response = 'pctd', label = 'pctd', modelStatement = '~ (1|range) + (1|row) + (1|genotype)')

v_p <- sum(pctd_vp$vcov) - pctd_vp$vcov[which(pctd_vp$grp=='Residual')]/2
v_r <- pctd_vp$vcov[which(pctd_vp$grp=='Residual')]/2
v_g <- pctd_vp$vcov[which(pctd_vp$grp=='genotype')]
r_n <- 1 - (v_r/v_p)
h_pctd <- v_g/v_p

vp.plot <- ggplot(pctd_vp, aes(label, pctVar, fill = grp)) + 
  geom_col() + 
  labs(title = 'ExG')
vp.plot

# image_scores_lg200 <- read_csv('data/manual/20260318_ScoresSorghum_LFGT.csv') %>%
#   filter(project=='UNL')
# image_scores_lg800 <- read_csv('data/manual/scores800_LFGT.csv')
# image_scores_lg <- bind_rows(image_scores_lg200, image_scores_lg800)
# 
# image_scores_rr200 <- read_csv('data/manual/20260319_ScoresSorghum_RRR.csv') %>%
#   filter(project=='UNL')
# image_scores_rr800 <- read_csv('data/manual/scores800_RRR.csv')
# image_scores_rr <- bind_rows(image_scores_rr200, image_scores_rr800)
# 
# image_scores <- bind_rows(image_scores_lg, image_scores_rr) %>%
#   mutate(plotNumber = str_split_i(image, '_', 1) %>%
#            as.numeric(),
#          image_id = str_remove(image, '-05_00.jpg')) %>%
#   left_join(ne_field_idx, join_by(plotNumber))
# # write_csv(image_scores, 'data/manual/all_image_scores.csv')
# 
# # image_scores <- read_csv('data/manual/all_image_scores.csv')
# sap135_scores_rr <- read_csv('data/manual/20260520_RRR_1.csv') %>% 
#   filter(project=='sap135images')
# sap135_scores_lg <- read_excel('data/manual/20260518_SAP135_scores.xlsx') %>% 
#   mutate(timestamp = as_datetime(timestamp))
# sap135_scores <- bind_rows(sap135_scores_lg, sap135_scores_rr) %>% 
#   mutate(genotype = 'PI576385', 
#          image_id = basename(image) %>% 
#          str_split_remove_i(fixed('-'), 4))
# image_scores <- bind_rows(image_scores, sap135_scores)
# write_csv(image_scores, 'data/manual/all_image_scores.csv')

image_scores <- read_csv('data/manual/all_image_scores.csv')

image_scores <- image_scores %>%
  mutate(genotype = str_remove_all(genotype, ' ')) %>%
  left_join(genotype_alignment, join_by(genotype==genotype_idx)) %>%
  mutate(genotype_markers = case_when(is.na(genotype_markers) ~ genotype,
                                      .default = genotype_markers)) %>% 
  filter(!is.na(genotype_markers)) %>%
  select(!c(genotype)) %>% 
  rename(genotype = genotype_markers)



interrater_image_scores <- image_scores %>%
  pivot_wider(id_cols = c(project, image, plotNumber, image_id, range, row, experiment, block, genotype),
              names_from = username, 
              values_from = score)

cor(interrater_image_scores$Libia, interrater_image_scores$Ruben, method = 'spearman', use = 'complete.obs')^2

interrater_corplot <- ggplot(interrater_image_scores, aes(Libia, Ruben)) + 
  geom_point(color = paletteer_d("ggsci::default_gsea", 4)[4], alpha = 0.25) +
  geom_abline(intercept = 0, slope = 1, color = 'black') + 
  geom_smooth(method = 'lm', linetype = 'dashed', se = FALSE, color = 'black') + 
  annotate('text', x = 2, y = 5, label = "'Spearman '~rho^2==0.46", parse = TRUE, 
           size = 9, size.unit = 'pt') +
  scale_x_continuous(name = 'Human Score A', limits = c(1, 5)) + 
  scale_y_continuous(name = 'Human Score B', limits = c(1, 5)) +
  theme_use
interrater_corplot


human_vi_scores <- pctd_ne %>% 
  right_join(interrater_image_scores, join_by(image_id, plotNumber)) %>%
  mutate(mean_score = (Libia + Ruben)/2)

cor(human_vi_scores$mean_score, human_vi_scores$pctd, method = 'spearman', use = 'complete.obs')^2

human_vi_corplot <- ggplot(human_vi_scores, aes(mean_score, pctd)) + 
  geom_point(color = paletteer_d("ggsci::default_gsea", 4)[4], alpha = 0.25) +
  geom_smooth(method = 'lm', linetype = 'dashed', se = FALSE, color = 'black') + 
  annotate('text', x = 2, y = 70, label = "'Spearman '~rho^2==0.58", parse = TRUE, 
           size = 9, size.unit = 'pt') +
  scale_x_continuous(name = 'Human Score (Mean)', limits = c(0.75, 5)) + 
  scale_y_continuous(name = 'Area Under ExG Threshold',
                     labels = ~str_c(.x, '%')) +
  theme_use
human_vi_corplot

score_vi_cor <- plot_grid(interrater_corplot, human_vi_corplot, nrow = 1, labels = 'auto')
ggsave('figures/supplemental/human_vi_correlation.png', plot = score_vi_cor, width = 6.16, height = 2.89, dpi = 1e3, bg = NULL)

gte25 <- filter(human_vi_scores, pctd >= 25)

score_plot_avg <- image_scores %>% 
  mutate(score = score*100) %>% 
  group_by(plotNumber, row, range, genotype, username) %>% 
  summarise(score = mean(score, na.rm = TRUE))
vp_scores <- partitionVariance3(score_plot_avg, response = 'score', label = 'score', modelStatement = '~ (1|username) + (1|row) + (1|range) + (1|genotype)')
v_p <- sum(vp_scores$vcov) - vp_scores$vcov[which(vp_scores$grp=='Residual')]/2
v_s <- sum(vp_scores$vcov[which(vp_scores$grp %in% c('Residual', 'username'))])
v_g <- vp_scores$vcov[which(vp_scores$grp=='genotype')]
r_scores <- 1 - (v_s/v_p)
h_scores <- v_g/v_p

vp.plot <- ggplot(vp_scores, aes(label, pctVar, fill = grp)) + 
  geom_col() + 
  labs(title = 'human scores')
vp.plot

score_plot_summary <- human_vi_scores %>% 
  add_count(plotNumber) %>% 
  filter(n > 1) %>% 
  group_by(plotNumber) %>% 
  summarise(mean_score_range = range(mean_score, na.rm = TRUE), 
            pctd_range = range(pctd, na.rm = TRUE))

model_specs07 <- read_csv('data/rf_model_specs_20260507.csv')
model_specs07 <- mutate(model_specs07, predictiveAbility = -1)

pctd_model_df <- read_csv(str_c('output/rf_20260507/', model_specs07$model[22], '_', model_specs07$label[22], '_', model_specs07$predictor_prefix[22], '_predictions_rf.csv'))
spearman_r2 <- cor(pctd_model_df[['label']], pctd_model_df[['predicted']], use = 'complete.obs', method = 'spearman')^2
pctd_pred <- ggplot(pctd_model_df, aes(label, predicted)) + 
  geom_point(color = paletteer_d("ggsci::default_gsea", 4)[4], alpha = 0.5) + 
  geom_abline(intercept = 0, slope = 1, color = 'black') + 
  geom_smooth(method = 'lm', linetype = 'dashed', se = FALSE, color = 'black') + 
  annotate('text', 25, 100, label = "'Spearman '~rho^2==0.56", parse = TRUE, 
           size = 9, size.unit = 'pt') +
  scale_x_continuous(expand = c(0, 0), 
                     limits = c(0, 105),
                     labels = ~str_c(.x, '%')) + 
  scale_y_continuous(expand = c(0, 0), 
                     limits = c(0, 105),
                     labels = ~str_c(.x, '%')) + 
  labs(x = 'Area Below ExG Threshold', y = 'Predicted Area Below ExG Threshold') + 
  theme_use
pctd_pred
ggsave('figures/supplemental/exg_RF_all.png', plot = pctd_pred, dpi = 1e3, bg = 'transparent', width = 3.72, height = 3.86)

high_error <- filter(pctd_model_df, 
                     label > 70 & predicted < 25)

model_specs <- read_csv('data/rf_model_specs_20260524.csv')
model_specs <- mutate(model_specs, predictiveAbility = -1)
model_specs <- model_specs[40:52, ]
for(i in 1:nrow(model_specs))
{
  df <- read_csv(str_c('output/rf_20260524/', model_specs$model[i], '_', model_specs$label[i], '_', model_specs$predictor_prefix[i], '_predictions_rf.csv'))
  model_specs$predictiveAbility[i] = getRFPredictability(df, 
                                                         model_descriptor = str_c(model_specs$model[i], model_specs$label[i], model_specs$predictor_prefix[i], sep = ':'))
  
}

model_order <- c('ae1', 'ae2', 'ae3', 'ae4', 'ae5', 'ae6', 'ae7', 'dinov2_mean', 'dinov2_std', 'dinov2', 'sam3_mean', 'sam3_std', 'sam3')
model_specs <- mutate(model_specs, 
                      model = factor(model, 
                                     levels = model_order, 
                                     labels = c('AE1', 'AE2', 'AE3', 'AE4', 'AE5', 'AE6', 'AE7', 
                                                'DINOv2 Mean', 'DINOv2 SD', 'DINOv2 All', 'SAM3 Mean', 'SAM3 SD', 'SAM3 All')))

predictability_bars_pc <- ggplot(model_specs, aes(model, predictiveAbility, fill = model)) + 
  geom_col() + 
  scale_y_continuous(expand = c(0, 0), name = expression("Spearman "~ R^2)) +
  scale_fill_manual(values = c(paletteer_d("ggsci::default_gsea", 12)[7:12], 
                               paletteer_d("dichromat::DarkRedtoBlue_12", 12)[c(12, 4, 3, 1)], 
                               paletteer_d("ggsci::default_gsea", 12)[c(5, 3, 1)])) +
  labs(x = NULL) +
  theme_use + 
  theme(axis.text.x = element_text(angle = 90), 
        legend.position = 'none')
predictability_bars_pc
ggsave(filename = 'figures/supplemental/predictability_bars_100PCs.png', plot = predictability_bars_pc, width = 3.3, height = 1.85, units = 'in', dpi = 1e3, bg = 'transparent')

fi_human_features <- read_csv('output/rf_20260524/sam3_human_scores_embedding_feature_importances_rf.csv') %>% 
  pivot_longer(cols = everything(), names_to = 'feature', values_to = 'fi') %>% 
  group_by(feature) %>%
  summarise(avg_fi = mean(fi, na.rm = TRUE)) %>% 
  arrange(desc(avg_fi)) %>% 
  mutate(feature = as.numeric(feature)) %>% 
  mutate(stat = case_when(feature < 1024 ~ 'mean', .default = 'std'), 
         embedding_num = case_when(feature > 1023 ~ feature - 1024, .default = feature))
write_csv(fi_human_features, 'data/rf_feature_importances_sam3_human_scores.csv')

fi_pctd_features <- read_csv('output/sam3_rs_embedding_pctd_senesced_removed_feature_importances_rf.csv') %>% 
  pivot_longer(cols = everything(), names_to = 'feature', values_to = 'fi') %>% 
  group_by(feature) %>%
  summarise(avg_fi = mean(fi, na.rm = TRUE)) %>% 
  arrange(desc(avg_fi)) %>% 
  mutate(feature = as.numeric(feature)) %>% 
  mutate(stat = case_when(feature < 1024 ~ 'mean', .default = 'std'), 
         embedding_num = case_when(feature > 1023 ~ feature - 1024, .default = feature))
write_csv(fi_pctd_features, )

high_fi <- fi_human_features[1:47, ]

high_fi_features <- str_c('embedding', high_fi$stat, high_fi$embedding_num, sep = '_')

low_fi_features <- fi_human_features[2002:2048, ]
low_fi_features <- str_c('embedding', low_fi_features$stat, low_fi_features$embedding_num, sep = '_')

embeddings <- read_csv('output/sam3_pctd_crops_rf.csv') %>% 
  mutate(location = 'UNL')

high_fi_embeddings <- embeddings %>% 
  select(any_of(c('image_name', high_fi_features))) %>% 
  pivot_longer(!image_name, values_to = 'value', names_to = 'embedding') %>% 
  mutate(stat = str_split_i(embedding, '_', 2) %>% 
           str_replace('std', 'sd'), 
         feature = str_split_i(embedding, '_', 3)) %>% 
  mutate(embedding_label = str_c(feature, ' (', str_to_title(stat), ')') %>% 
           str_replace('Sd', 'SD')) %>% 
  select(embedding_label, value, image_name) %>% 
  pivot_wider(id_cols = image_name, names_from = embedding_label, values_from = value) %>% 
  select(!image_name) %>%
  as.matrix()

cor_mat <- cor(high_fi_embeddings)
embedding_corplot <- ggcorrplot(cor_mat, 
                type = 'upper', 
                ggtheme = theme_use, 
                title = 'Top SAM3 Embeddings',
                legend.title = 'Pearson Correlation Coefficient', 
                outline.color = 'transparent', 
                hc.order = TRUE) + 
  labs(x = NULL, y = NULL) +
   theme_use +
  theme(axis.text.x = element_text(angle = 90))
embedding_corplot
ggsave('figures/supplemental/embedding_corrplot.png', plot = embedding_corplot, 
       width = 6.5, height = 7.7,
       dpi = 1e3, bg = NULL)

cor_mat_filt <- replace(cor_mat, cor_mat==1, NA)
max(cor_mat_filt, na.rm = TRUE)
min(cor_mat_filt, na.rm = TRUE)

embeddings_fvsu <- read_csv('output/sam3_embeddings_fvsu.csv') %>%
  mutate(image_id = basename(image_path) %>%
           str_split_remove_i('_', 7) %>%
           str_remove('-05_00'),
         plotNumber = str_split_i(image_id, '_', 1) %>%
           as.numeric(),
         location = 'FVSU') %>%
  filter(image_id %in% pctd_fvsu$image_id) %>%
  left_join(idx_fvsu, join_by(plotNumber), relationship = 'many-to-one')

embeddings_aamu <- read_csv('output/sam3_embeddings_aamu.csv') %>%
  mutate(image_id = basename(image_path) %>%
           str_split_remove_i('_', 7) %>%
           str_remove('-05_00'),
         plotNumber = str_split_i(image_id, '_', 1) %>%
           as.numeric(),
         location = 'AAMU') %>%
  filter(image_id %in% pctd_aamu$image_id) %>%
  left_join(idx_aamu, join_by(plotNumber), relationship = 'many-to-one')

embeddings_all <- bind_rows(embeddings, embeddings_aamu, embeddings_fvsu) %>%
  mutate(genotype = str_remove_all(genotype, ' '), 
         image_name = case_when(is.na(image_name) ~ basename(image_path), .default = image_name)) %>%
  left_join(genotype_alignment, join_by(genotype==genotype_idx)) %>%
  mutate(genotype_markers = case_when(is.na(genotype_markers) ~ genotype,
                                      .default = genotype_markers)) %>%
  filter(!is.na(genotype_markers)) %>%
  select(!c(genotype)) %>%
  rename(genotype = genotype_markers) %>% 
  mutate(across(contains('embedding'), ~.x*1e3))

embeddings_count <- embeddings_all %>% 
  group_by(location, plotNumber) %>% 
  count()
hist <- ggplot(embeddings_count, aes(n)) + 
  geom_histogram()
hist

# embeddings_avg_ne <- embeddings_all %>%
#   group_by(location, plotNumber, range, row, block, genotype) %>%
#   summarise(across(contains('embedding'), ~mean(.x, na.rm = TRUE))) %>%
#   filter(location=='UNL')
# embeddings_vp_ne <- tibble()
# lv_cols <- c("embedding_std_930", "embedding_std_552", "embedding_std_918", "embedding_mean_637", "embedding_std_976", "embedding_mean_886",
#             "embedding_mean_210", "embedding_std_383", "embedding_std_687", "embedding_mean_984", "embedding_mean_68", "embedding_mean_836",
#             "embedding_mean_586", "embedding_std_968", "embedding_mean_546", "embedding_std_821", "embedding_mean_656", "embedding_mean_968",
#             "embedding_mean_698", "embedding_mean_165", "embedding_mean_37", "embedding_mean_582", "embedding_mean_214", "embedding_mean_792",
#             "embedding_std_606", "embedding_mean_930", "embedding_mean_734", "embedding_mean_108", "embedding_mean_560", "embedding_std_166",
#             "embedding_std_983", "embedding_std_981", "embedding_mean_197", "embedding_std_817", "embedding_std_132", "embedding_std_82",
#             "embedding_mean_950", "embedding_mean_119", "embedding_std_128", "embedding_mean_139", "embedding_std_76", "embedding_std_839",
#             "embedding_std_326", "embedding_mean_842", "embedding_std_567", "embedding_mean_237", "embedding_std_793")

# lv_cols <- colnames(embeddings_all)[str_detect(colnames(embeddings_all), 'embedding')]

# for(lv in lv_cols)
# {
#   embeddings_vp_ne <- bind_rows(embeddings_vp_ne,
#                              partitionVariance3(embeddings_avg_ne, lv, lv,
#                                                 '~ (1|range) + (1|row) + (1|genotype)'))
# }
# write_csv(embeddings_vp_ne, 'output/embedding_vp_ne.csv')

embeddings_vp_ne <- read_csv('output/embedding_vp_ne.csv')

vp.plot <- ggplot(embeddings_vp_ne, aes(label, pctVar, fill = grp)) +
  geom_col() +
  labs(title = 'embeddings - all')
vp.plot

embeddings_vp_ne <- embeddings_vp_ne %>% 
  select(responseVar, grp, vcov) %>% 
  pivot_wider(id_cols =  responseVar, 
              names_from = grp, 
              values_from =  vcov) %>% 
  rowwise() %>% 
  mutate(v_r = Residual/2,
         h = genotype / sum(c(genotype, range, row, v_r), na.rm = TRUE)) %>% 
  arrange(desc(h))

high_fi_vp <- filter(embeddings_vp_ne, responseVar %in% high_fi_features)
min(high_fi_vp$h, na.rm = TRUE)
max(high_fi_vp$h, na.rm = TRUE)
low_fi_vp <- filter(embeddings_vp_ne, !(responseVar %in% high_fi_features)) %>% 
  arrange(desc(h))

low_fi_highH <- low_fi_vp$responseVar[1:10]

rmip_pctd <- read_csv('output/farmcpu_20260515/pctd_farmcpu_hits.csv') %>% 
  mutate(embedding = 'pctd') %>% 
  group_by(SNP, CHROM, POS, embedding) %>% 
  summarise(RMIP = n()/100, 
            min_p = min(FarmCPU_P, na.rm = TRUE), 
            mean_effect = mean(FarmCPU_Effect, na.rm = TRUE)) %>% 
  arrange(desc(RMIP))
pctd_manhattan <- plotManhattan(rmip_pctd, RMIP, threshold = 0.2, theme = theme_use, species = 'sorghum', chrGap = 8e6, 
                                colors = paletteer_d("RColorBrewer::Paired", 10), main = 'Percent Unhealthy Tissue')
ggsave('figures/supplemental/pctd_manhattan.png', plot = pctd_manhattan, width = 6.5, height = 3.25, dpi = 1e3, bg = NULL)

rmip_scores <- read_csv('output/farmcpu_20260515/human_scores_farmcpu_hits.csv') %>% 
  mutate(embedding = 'human_score') %>%
  group_by(SNP, CHROM, POS, embedding) %>% 
  summarise(RMIP = n()/100, 
            min_p = min(FarmCPU_P, na.rm = TRUE), 
            mean_effect = mean(FarmCPU_Effect, na.rm = TRUE)) %>% 
  arrange(desc(RMIP))
scores_manhattan <- plotManhattan(rmip_scores, RMIP, threshold = 0.2, theme = theme_use, species = 'sorghum', chrGap = 8e6,
                                  colors = paletteer_d('RColorBrewer::Paired', 10), main = 'Human Disease Severity Scores')
ggsave('figures/supplemental/scores_manhattan.png', plot = scores_manhattan, width = 6.5, height = 3.25, dpi = 1e3, bg = NULL)

all_farmcpu_hits <- read_csv('output/farmcpu_20260515/all_farmcpu_hits.csv') %>% 
  filter(embedding %in% high_fi_features)

rmip <- all_farmcpu_hits %>% 
  group_by(SNP, CHROM, POS, embedding) %>% 
  summarise(RMIP = n()/100, 
            min_p = min(FarmCPU_P, na.rm = TRUE), 
            mean_effect = mean(FarmCPU_Effect, na.rm = TRUE)) %>% 
  arrange(desc(RMIP))

lrr2_chr5loc <- mean(59834505:59836269) + 348316776
cdl1_chr9loc <- mean(60010749:60014092) + 654779914
cs1a_chr9loc <- mean(60240330:60245571) + 654779914
bak1_chr1loc <- mean(65789782:65792511)

embeddings_manhattan <- plotManhattan(rmip, RMIP, threshold = 0.2, theme = theme_use, species = 'sorghum', chrGap = 8e6,
                                      colors = paletteer_d('RColorBrewer::Paired', 10), main = 'All High Feature Importance SAM3 Embeddings')  + 
  annotate('point', x = c(bak1_chr1loc, lrr2_chr5loc, cdl1_chr9loc, cs1a_chr9loc), y = rep(0, 4), size = 4, color = 'blue', shape = 17)
ggsave('figures/supplemental/embeddings_manhattan.png', plot = embeddings_manhattan, width = 6.5, height = 3.25, dpi = 1e3, bg = NULL)

rmip_0.2 <- filter(rmip, RMIP > 0.2) %>% 
  mutate(label = str_c(feature, ' (', str_to_title(stat), ')') %>% 
           str_replace('Std', 'SD')) %>% 
  ungroup() %>% 
  select(SNP, CHROM, POS, RMIP, label)
colnames(rmip_0.2) <- c('Marker', 'Chromosome', 'Position', 'RMIP', 'Embedding')
write_csv(rmip_0.2, 'figures/supplemental/GWAS_significant_associations_all.csv')

rmip_0.2SNPs <- rmip_0.2 %>%
  group_by(Chromosome, Position) %>%
  summarise(n_hits = n()) %>% 
  arrange(Chromosome, Position)

rmip_0.2SNPs %>% 
  select(Chromosome, Position) %>% 
  write_tsv('output/all_sig_snps.txt', col_names = FALSE)

vcf_sig <- read_tsv('output/all_sig_snps.recode.vcf', skip = 98)
colnames(vcf_sig) <- c('CHROM', str_replace(colnames(vcf_sig)[2:length(colnames(vcf_sig))], 'SC', 'SC '))
vcf_sig <- vcf_sig %>%
  select(!c(QUAL, INFO, FORMAT, FILTER)) %>%
  mutate(ID = str_c(CHROM, POS, REF, ALT, sep = ':')) %>%
  pivot_longer(cols = !c(CHROM, POS, ID, REF, ALT), 
               names_to = 'genotype', 
               values_to = 'allele') %>% 
  select(ID, genotype, allele) %>% 
  pivot_wider(id_cols = genotype, 
              names_from = ID, 
              values_from = allele)
markers <- colnames(vcf_sig)[2:ncol(vcf_sig)]

image_scores_markers <- image_scores %>% 
  right_join(vcf_sig, join_by(genotype)) %>% 
  select(genotype, score, all_of(markers)) %>%
  mutate(across(all_of(markers), 
                ~case_when(.x %in% c('0/1', '1/0', './.', '.', '0|1') ~ NA, 
                           .x=='1|1' ~ '1/1', 
                           .x=='0|0' ~ '0/0',
                           .default = .x)))

r_gene_donors <- c('PI576385', 'PI656070', 'PI533869')
r_gene_donor_names <- c('SAP-135', 'SC 748-5', 'SC 283')
r_gene_names <- c('ARG4', 'LRR2', 'ARG1')
intersect(r_gene_donors, embeddings_all$genotype)
intersect(r_gene_donors, image_scores$genotype)
r_gene_donors_scores <- filter(image_scores, genotype %in% r_gene_donors) 
r_gene_donors_summary <- r_gene_donors_scores %>% 
  group_by(genotype) %>% 
  summarise(score_mean = mean(score, na.rm = TRUE), 
            score_median = median(score, na.rm = TRUE))
r_gene_donors_exg <- pctd_ne %>% 
  mutate(genotype = str_remove_all(genotype, ' ')) %>%
  left_join(genotype_alignment, join_by(genotype==genotype_idx)) %>%
  mutate(genotype_markers = case_when(is.na(genotype_markers) ~ genotype,
                                      .default = genotype_markers)) %>% 
  filter(!is.na(genotype_markers)) %>%
  select(!c(genotype)) %>% 
  rename(genotype = genotype_markers) %>% 
  group_by(location, genotype) %>% 
  filter(genotype %in% r_gene_donors)
r_gene_donors_exg_summary <- r_gene_donors_exg %>%
  ungroup() %>%
  group_by(genotype) %>% 
  summarise(pctd_mean = mean(pctd, na.rm = TRUE), 
            pctd_median = median(pctd, na.rm = TRUE), 
            n = n())
r_gene_images_ne <- filter(embeddings_all, location=='UNL' & genotype %in% r_gene_donors) %>% 
  pull(image_id) %>% 
  unique()

all_mlm <- read_csv('output/mlm_20260515/most_significant_associations_0.0001.csv')

chr6_embeddings <- all_mlm %>% 
  group_by(embedding) %>% 
  add_count() %>% 
  group_by(CHROM, embedding) %>% 
  summarise(proportionMarkers = n()/mean(n)) %>% 
  filter(CHROM==6 & proportionMarkers > 0.5) %>% 
  pull(embedding)

chr6_hits <- filter(all_mlm, embedding %in% chr6_embeddings & CHROM==6) %>% 
  group_by(embedding) %>% 
  summarise(peak_start = min(POS), 
            peak_end = max(POS), 
            peak_length = peak_end - peak_start, 
            min_pval = min(MLM_P)) %>% 
  filter(58583464 > peak_start & 58583464 < peak_end) %>% 
  mutate(peak_length_mb = peak_length/1e6) %>% 
  arrange(desc(peak_length_mb))

chr6_peaks <- filter(all_mlm, CHROM==6 & 
                            embedding %in% chr6_hits$embedding[1:9]) %>% 
  arrange(embedding, POS)

chr6_sig <- filter(chr6_hits, min_pval < 1.71e-8)

manhattan <- ggplot(chr6_peaks, aes(POS, -log10(MLM_P), color = embedding)) + 
  geom_point() + 
  annotate('rect', xmin = 58582313, xmax = 58584616, ymin = 5.5, ymax = 8.5, color = 'purple', fill = 'transparent')
  # scale_x_continuous(limits = c(5.5e7, 6.5e7))
manhattan

image_scores_al_ga <- read_csv('data/manual/image_scores_al_ga.csv')
image_scores_aamu <- filter(image_scores_al_ga, location=='AAMU') %>% 
  left_join(idx_aamu, join_by(plotNumber), relationship = 'many-to-one')
image_scores_fvsu <- filter(image_scores_al_ga, location=='FVSU') %>% 
  left_join(idx_fvsu, join_by(plotNumber))
image_scores_ne <- read_csv('data/manual/all_image_scores.csv') %>% 
  mutate(location = 'UNL')

image_scores_all <- bind_rows(image_scores_aamu, image_scores_fvsu, image_scores_ne) %>% 
  mutate(genotype = str_remove_all(genotype, ' ')) %>%
  left_join(genotype_alignment, join_by(genotype==genotype_idx)) %>%
  mutate(genotype_markers = case_when(is.na(genotype_markers) ~ genotype,
                                      .default = genotype_markers)) %>% 
  filter(!is.na(genotype_markers)) %>%
  select(!c(genotype)) %>% 
  rename(genotype = genotype_markers) %>%
  inner_join(vcf_sig, join_by(genotype)) %>% 
  mutate(across(all_of(markers), 
                ~case_when(.x %in% c('0/1', '1/0', './.', '.', '0|1') ~ NA, 
                           .x=='1|1' ~ '1/1', 
                           .x=='0|0' ~ '0/0',
                           .default = .x)))

genotypes_intersect <- intersect(blues_unl$genotype, blues_aamu$genotype)
genotypes_intersect <- intersect(genotypes_intersect, blues_fvsu$genotype)  

image_scores_nec <- filter(image_scores_all, location=='UNL' & genotype %in% genotypes_intersect) %>% 
  mutate(location = 'UNL-SAP')

image_scores_all <- bind_rows(image_scores_all, image_scores_nec)

df_bar <- image_scores_all %>% 
  ungroup() %>% 
  filter(!is.na(score)) %>%
  select(score, location, all_of(markers)) %>% 
  pivot_longer(!c(score, location), 
               values_to = 'allele', 
               names_to = 'marker') %>% 
  filter(!is.na(allele)) %>% 
  group_by(marker, allele, location) %>%
  summarise(mean = mean(score), 
            se = sd(score)/sqrt(n()), 
            n = n()) %>% 
  mutate(location = factor(location, levels = c('UNL', 'UNL-SAP', 'AAMU', 'FVSU')))

wilcox.test(score ~ `6:58584404:G:A`, data = image_scores_all, subset = location=='UNL')
wilcox.test(score ~ `6:58584404:G:A`, data = image_scores_all, subset = location=='UNL-SAP')
wilcox.test(score ~ `6:58584404:G:A`, data = image_scores_all, subset = location=='AAMU')
wilcox.test(score ~ `6:58584404:G:A`, data = image_scores_all, subset = location=='FVSU')

pLocusScoreBar <-  df_bar %>% 
  filter(marker == "6:58584404:G:A") %>%
  ggplot(aes(location, mean, fill = allele)) + 
  geom_col(position = position_dodge(width = 0.9)) + 
  geom_errorbar(aes(ymin  = mean - se, ymax = mean + se), position = position_dodge(width = 0.9), width = 0.25) + 
  annotate(geom = 'text', 
           x = c('UNL', 'UNL-SAP', 'AAMU', 'FVSU'), 
           y = rep(2.9, 4),
           label = c('*', '', '****', '****')) +
  scale_x_discrete(name = NULL, 
                   expand = c(0, 0), 
                   label = c('NE', 'NE-C', 'AL', 'GA')) + 
  scale_y_continuous(name = 'Human Ordinal Score', 
                     expand = c(0, 0), 
                     limits = c(0, 3)) +
  scale_fill_manual(name = 'Chr6:58584404', 
                    values = paletteer_d('MoMAColors::Althoff')[1:3], 
                    label = c('G', 'A')) + 
  theme_use + 
  theme(axis.text.x = element_text(angle = 90))
pLocusScoreBar
ggsave('figures/supplemental/pLocusScores.png', height = 2.75, width = 3.25, dpi = 1e3, bg = NULL)
