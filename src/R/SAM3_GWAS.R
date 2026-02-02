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

embeddings <- read_csv('output/sam3_rs_rf_predictors.csv')
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

idx_fvsu <- read_csv('data/fvsu2025/FVSU_SAP_BAP.csv') %>% 
  rename(plotNumber = `Plot No`,
         genotype = Accession, 
         rep = Replication, 
         population = Population) %>% 
  arrange(plotNumber) %>% 
  add_column(range = c(rep(3:23, each = 16), rep(24, 11), rep(25:45, each = 16), rep(46, 11), rep(49:69, each = 16), rep(70, 9), 
                       rep(71:84, each = 16), rep(91:92, each = 16), rep(89:90, each = 16), rep(87:88, each = 16), rep(85, 16), rep(86, 9)), 
             row = c(rep(3:18, times = 8), rep(c(3:5, 7:19), times = 2), rep(3:18, times = 2), rep(c(3:4, 6:19), times = 2), 
                     rep(c(3:15, 17:19), times = 2), rep(3:18, times = 5), 3:13, rep(3:18, times = 18), rep(c(3, 5:19), times = 2), 3:18, 3:13, 
                     rep(3:18, times = 21), 3:11, rep(3:18, times = 16), rep(c(3:15, 17:19), times = 2), rep(3:18, times = 3), 3:11))
  
embeddings_fvsu <- read_csv('output/sam3_embeddings_fvsu.csv') %>% 
  mutate(location = 'FVSU', 
         plotNumber = basename(image_path) %>% 
           str_split_i('_', 1) %>% 
           as.numeric(),
         image_id = basename(image_path) %>%
           str_remove('-05_00_[0-9]\\.(png|npz)') %>%
           str_remove('-05_00\\.jpg')) %>% 
  left_join(idx_fvsu, join_by(plotNumber))

idx_aamu <- read_csv('data/aamu2025/2025 Sorghum_Update_073125-XK_Index.csv') %>% 
  rename(genotype = `Variety Name`, 
         plotNumber = `Plot Number`, 
         block = Block) %>% 
  arrange(plotNumber) %>% 
  add_column(range = c(rep(1:27, times = 32), rep(28:54, times = 32)), 
             row = c(rep(1:16, each = 27), rep(17:32, each = 27), rep(1:16, each = 27), rep(17:32, each = 27))) %>% 
  mutate(poundsOfNitrogenPerAcre = case_when(block %in% c(1, 4) ~ 100, .default = 180))

embeddings_aamu <- read_csv('output/sam3_embeddings_aamu.csv') %>% 
  mutate(location = 'AAMU') %>% 
  mutate(plotNumber = basename(image_path) %>% 
           str_split_i('_', 1) %>% 
           as.numeric(), 
         image_id = basename(image_path) %>% 
           str_remove('-05_00_[0-9]\\.(png|npz)') %>%
           str_remove('-05_00\\.jpg')) %>% 
  left_join(idx_aamu, join_by(plotNumber))

idx_unl <- read_csv('data/ne2025/SbDiv_ne2025_fieldindex.csv') %>% 
  select(!genotype)
pctd <- read_csv('data/final_disease_scores.csv') %>% 
  mutate(image_id = str_remove(image_path, '-05_00_leaf.png'))

embeddings_all <- embeddings %>% 
  mutate(location = 'UNL') %>% 
  left_join(idx_unl, join_by(plotNumber)) %>% 
  mutate(image_id = basename(image_path) %>% 
           str_remove('-05_00_[0-9]\\.(png|npz)') %>%
           str_remove('-05_00\\.jpg')) %>%
  bind_rows(embeddings_fvsu, embeddings_aamu) %>% 
  filter(image_id %in% pctd$image_id) %>% 
  mutate(genotype = str_replace(genotype, 'PI ', 'PI')) %>% 
  filter(!genotype %in% c('Border', 'Check', 'Fill', 'Mixed')) %>% 
  mutate(genotype = case_when(genotype=='1903 AS 4633' ~ 'PI533936', genotype=='194 Kano' ~ 'PI534054', genotype=='2033Z-3' ~ 'PI533970',
                              genotype=='255 Tirter' ~ 'PI533866', genotype=='290 Feterita Shendi 2' ~ 'PI533769', 
                              genotype=='450 Bori Light Brown' ~ 'PI534047', genotype=='54.K.94(Witchweed res.)' ~ 'PI533752', 
                              genotype=='6428 Sian' ~ 'PI534037', genotype=='740 Oua Berr' ~ 'PI533863', genotype=='80M' ~ 'PI656041',
                              genotype=='88V1080' ~ 'PI656006', genotype=='90M' ~ 'PI656012', genotype=='96CD635' ~ 'PI656062',
                              genotype=='A 84' ~ 'PI534099', genotype=='A-106' ~ 'PI534101', genotype=='A-96' ~ 'PI533901',
                              genotype=='A/BTx626' ~ 'PI552856', genotype=='A/BTx635' ~ 'PI561073', genotype=='A/BTxARG-1' ~ 'PI561072', 
                              genotype=='Accho Karuho' ~ 'PI534114', genotype=='Ajabsido' ~ 'PI656015', genotype=='Akwu' ~ 'PI534115', 
                              genotype=='AS 2613 NT' ~ 'PI533943', genotype=='AS 4055 N Kambwa' ~ 'PI533939', genotype=='AS 5826 Holcus' ~ 'PI533938', 
                              genotype=='AS4616 Balangira-Mwanza' ~ 'PI533822', genotype=='AS4660 Kikuma' ~ 'PI533821', 
                              genotype=='BA45 Faria Bonkum' ~ 'PI533838', genotype=='Barking 119' ~ 'PI533985', genotype=='BE 25' ~ 'PI534070',
                              genotype=='BO 36' ~ 'PI534063', genotype=='BOK11' ~ 'PI656002', genotype=='Brawley' ~ 'PI533998',
                              genotype=='BTx2752' ~ 'PI656018', genotype=='BTx2928' ~ 'PI629059', genotype=='BTx399' ~ 'PI655993',
                              genotype=='BTx406' ~ 'PI656020', genotype=='BTx615' ~ 'PI656022', genotype=='BTx638' ~ 'PI574455', 
                              genotype=='BTx640' ~ 'PI642791', genotype=='BTx642' ~ 'PI656021', genotype=='Budy' ~ 'PI534137', 
                              genotype=='Bulfontein White Kafir Corn DL/60/133' ~ 'PI533979',
                              genotype=='Butivori' ~ 'PI576359', genotype=='CE151-262-A1' ~ 'PI656031', genotype=='Chanan Singoo' ~ 'PI533855',
                              genotype=='Chari Uri' ~ 'PI576364', genotype=='CHILTEX' ~ 'PI655984', genotype=='Cholia Talijhari' ~ 'PI533852', 
                              genotype=='COMBINE 7078' ~ 'PI655990', genotype=='COMBINE HEGARI' ~ 'PI659691', 
                              genotype=='COMBINE KAFIR-60' ~ 'PI655988', genotype=='Culum Brick' ~ 'PI533830', genotype=='DAY MILO' ~ 'PI641874',
                              genotype=='Deburr' ~ 'PI533831', genotype=='DEER' ~ 'PI655995', genotype=='DELLA' ~ 'PI566819', 
                              genotype=='Dobbs' ~ 'PI533972', genotype=='Dorado' ~ 'PI656034', genotype=='DWARF YELLOW MILO' ~ 'PI24969',
                              genotype=='EC 18246 (preconverted)' ~ 'PI533845', genotype=='EC 21360 G29' ~ 'PI534104', 
                              genotype=='EC 21361 G30' ~ 'PI534105', genotype=='EC 21428 SB 63' ~ 'PI534108', genotype=='Ex-Mubi' ~ 'PI576434', 
                              genotype=='F.C.I. 4201' ~ 'PI34911', genotype=='F.R. Miller' ~ 'PI534167', 
                              genotype=='Framiola DL/59/1539' ~ 'PI533976', genotype=='Hamaisi 38' ~ 'PI533996', genotype=='HC 6028' ~ 'PI534097',
                              genotype=='Huria White 621' ~ 'PI533986', genotype=='ICSV 1089BF' ~ 'PI656036', genotype=='ICSV 400' ~ 'PI601816',
                              genotype=='ICSV 401' ~ 'PI656037', genotype=='ICSV 745' ~ 'PI576130', genotype=='IS 12623C' ~ 'PI659695', 
                              genotype=='IS 12661' ~ 'PI276837', genotype=='IS 18684' ~ 'PI533758', genotype=='IS 18696' ~ 'PI533800', 
                              genotype=='IS 2319C' ~ 'PI659693', genotype=='IS 3515C' ~ 'PI576399', genotype=='IS 5590C' ~ 'PI659753', 
                              genotype=='IS 7151C' ~ 'PI659696', genotype=='IS 8525(J)' ~ 'PI656083', genotype=='J.A.T.S. #67' ~ 'PI533927', 
                              genotype=='JOCORO' ~ 'PI656039', genotype=='Jonar Tamargundi' ~ 'PI534028', 
                              genotype=='Jowar Red Jankinagar' ~ 'PI576366', genotype=='K.3 Perimanjial Irungu Cholam' ~ 'PI533750',
                              genotype=='K037 Camjin' ~ 'PI533839', genotype=='KA 12 Janjari' ~ 'PI533876', genotype=='KA 15 Yazgar Giwa' ~ 'PI533877',
                              genotype=='KA 21 Gajerar Kaura' ~ 'PI533878', genotype=='KA 24' ~ 'PI534075', genotype=='Kabutuwa' ~ 'PI534144', 
                              genotype=='KANSAS ORANGE' ~ 'PI641824', genotype=='Karad 2-7-11' ~ 'PI533810', genotype=='KAT83369' ~ 'PI656043', 
                              genotype=='Kharuth Waragel' ~ 'PI576390', genotype=='Kireniga 317' ~ 'PI533987', genotype=='Klor' ~ 'PI533910', 
                              genotype=='Kodilib' ~ 'PI533789', genotype=='Kokla' ~ 'PI533911', genotype=='KS19' ~ 'PI655998', 
                              genotype=='Kuyuma' ~ 'PI656044', genotype=='Lambas' ~ 'PI576394', genotype=='M 1' ~ 'PI533871', 
                              genotype=='M35-1' ~ 'PI656047', genotype=='MACIA' ~ 'PI565121', genotype=='Maja Abiad Q2/2/68' ~ 'PI533949', 
                              genotype=='Malwal Aweil' ~ 'PI533962', genotype=='MARTIN' ~ 'PI655987', genotype=='MN 1592 (preconverted)' ~ 'PI533997',
                              genotype=='MN 707 (preconverted)' ~ 'PI533957', genotype=='MN 708 (preconverted)' ~ 'PI576393', 
                              genotype=='Monshal' ~ 'PI533757', genotype=='MR732' ~ 'PI656051', genotype=='Msumbji SB 117' ~ 'PI533869', 
                              genotype=='Mugbash 56/56' ~ 'PI533759',
                              .default = genotype))


