library(tidyverse)
source('src/R/Functions.R')
library(paletteer)
library(ggcorrplot)
library(ape)
source('../parallelgwas/manhattanPlot.R')

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
        axis.line.x.bottom = element_line(color = 'black', linewidth = 0.5),
        axis.line.y.left = element_line(color = 'black', linewidth = 0.5),
        panel.grid = element_blank(), 
        panel.background = element_blank())

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

fi_pctd_features <- read_csv('output/rf/sam3_rs_embedding_pctd_senesced_removed_feature_importances_rf.csv') %>% 
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

rmip_summary <- rmip %>%
  filter(RMIP > 0.2) %>% 
  group_by(SNP) %>% 
  summarise(sig_features = paste0(embedding, collapse = ';'))
# rmip_0.1features <- rmip %>% 
#   ungroup() %>% 
#   filter(RMIP >= 0.10) %>% 
#   distinct(feature) %>% 
#   pull(feature)
# n_features <- length(rmip_0.1features)
# 
# rmip_0.1features <- rmip %>% 
#   filter(feature %in% rmip_0.1features) %>% 
#   ungroup() %>% 
#   mutate(CHROM = str_remove(CHROM, 'Chr') %>% 
#            as.numeric())

select_features <- intersect(head(rmip$embedding, n=20), head(high_fi_features, n=20))
n_features <- length(select_features)
rmip_selected <- rmip %>% 
  filter(embedding %in% select_features) %>% 
  ungroup() %>% 
  mutate(CHROM = str_remove(CHROM, 'Chr') %>% 
           as.numeric(), 
         label = str_c(feature, '\n(', str_to_title(stat), ')'))

plotManhattan(rmip_selected, RMIP, multitrait = TRUE, trait = label, threshold = 0.2, 
              colors = paletteer_d("RColorBrewer::Paired", n_features),
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
  mutate(genotype = str_replace(genotype, 'PI ', 'PI') %>% 
           str_replace('SC ', 'SC')) %>% 
  mutate(genotype = case_when(str_length(genotype)==5 & str_starts(genotype, 'SC') ~ str_replace(genotype, 'SC', 'SC0'), 
                              .default = genotype) %>% 
           str_replace('SC', 'SC '), 
         block = max(block, rep, na.rm = TRUE)) %>% 
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
                              genotype=='Mugbash 56/56' ~ 'PI533759', genotype=='N250B' ~ 'PI656052', genotype=='N290B' ~ 'PI656053', 
                              genotype=='P850029' ~ 'PI656056', genotype=='P898012' ~ 'PI656057', genotype=='P9517' ~ 'PI656058', 
                              genotype=="Nebraska 6350" ~ 'PI533948', genotype=="Nkuli Swaziland" ~ 'PI576340', genotype=="No. 1 Gambela" ~ 'PI533792',
                              genotype=="No. 25 Gobo, Kaichama Ethiopia" ~ 'PI534123', genotype=="No. 35 Mab" ~ 'PI533903', 
                              genotype=="No. 37 Ubi, Abelti Ethiopia" ~ 'PI534127', genotype=="No. 4 Hadoui" ~ 'PI533766', 
                              genotype=="No. 5 Gambela" ~ 'PI533794', genotype=="No. 59 Bekedjie, Kembolcha Ethiopia" ~ 'PI534132', 
                              genotype=="No. 64 Netch Addesho Ethiopia" ~ 'PI534135', genotype=="No. 65 Amelsie, Kimbolcha Ethiopia" ~ 'PI534128',
                              genotype=="No. 69 Mashela Tinguish, Warakul Ethiopi" ~ 'PI534133', genotype=="No. 755 Muzeba" ~ 'PI533915', 
                              genotype=="No. 902 Sorghum arundi" ~ 'PI534145', genotype=="Orange No. l, Baijo" ~ 'PI533902', 
                              genotype=="P 3742" ~ 'PI533954', genotype=="P 3749 Q2/5/62" ~ 'PI533955', genotype=="P-721" ~ 'PI656055', 
                              genotype=="PINK KAFIR" ~ 'PI655972', genotype=="Pinolero 1" ~ 'PI656059', genotype=="PL 47" ~ 'PI534079',
                              genotype=="PLAINSMAN" ~ 'PI655985', genotype=="QL3-TEXAS" ~ 'PI656060', genotype=="QL3(India)" ~ 'PI656061', 
                              genotype=="R TX 431" ~ 'PI655997', genotype=="R1, 21" ~ 'PI534148', genotype=="R1,38" ~ 'PI533921', 
                              genotype=="R3, 80" ~ 'PI534155', genotype=="R9188" ~ 'PI656007', 
                              genotype=="Ramada_(XK: fromTomClemente_UNL_2024)" ~ 'PI651493', genotype=="RCV" ~ 'PI656008', 
                              genotype=="RED AMBER" ~ 'PI17548', genotype=="RED KAFIR" ~ 'PI655976', genotype=="REDBINE-60" ~ 'PI655989', 
                              genotype=="Rexx" ~ 'PI534163', genotype=="RIO" ~ 'PI651496', genotype=="ROX ORANGE" ~ 'PI641836', 
                              genotype=="RTAM428" ~ 'PI656009', genotype=="RTx2536" ~ 'PI656010', genotype=="RTX433" ~ 'PI564164', 
                              genotype=="RTX434" ~ 'PI564165', genotype=="RTX435" ~ 'PI656004', genotype=="RTx436" ~ 'PI561071', 
                              genotype=="RTx437" ~ 'PI629034', genotype=="Safara, Kordafan" ~ 'PI533964', genotype=="SAP-124" ~ 'PI576375', 
                              genotype=="SAP-125" ~ 'PI576376', genotype=="SAP-128" ~ 'PI576426', genotype=="SAP-131" ~ 'PI595739', 
                              genotype=="SAP-133" ~ 'PI595740', genotype=="SAP-134" ~ 'PI595741', genotype=="SAP-135" ~ 'PI576385', 
                              genotype=="SAP-138" ~ 'PI597961', genotype=="SAP-139" ~ 'PI595714', genotype=="SAP-141" ~ 'PI576422', 
                              genotype=="SAP-148" ~ 'PI533965', genotype=="SAP-150" ~ 'PI576425', genotype=="SAP-151" ~ 'PI597957', 
                              genotype=="SAP-154" ~ 'PI595743', genotype=="SAP-157" ~ 'PI595744', genotype=="SAP-158" ~ 'PI597966', 
                              genotype=="SAP-159" ~ 'PI595745', genotype=="SAP-162" ~ 'PI595718', genotype=="SAP-166" ~ 'PI597964', 
                              genotype=="SAP-168" ~ 'PI597968', genotype=="SAP-171" ~ 'PI597973', genotype=="SAP-172" ~ 'PI597976', 
                              genotype=="SAP-173" ~ 'PI597980', genotype=="SAP-175" ~ 'PI597982', genotype=="SAP-197" ~ 'PI597958', 
                              genotype=="SAP-206" ~ 'PI533807', genotype=="SAP-219" ~ 'PI533754', genotype=="SAP-225" ~ 'PI597950',
                              genotype=="SAP-275" ~ 'PI533980', genotype=="SAP-280" ~ 'PI576337', genotype=="SAP-287" ~ 'PI576332',
                              genotype=="SAP-294" ~ 'PI576345', genotype=="SAP-306" ~ 'PI576333', genotype=="SAP-311" ~ 'PI595702', 
                              genotype=="SAP-312" ~ 'PI576339', genotype=="SAP-323" ~ 'PI597952', genotype=="SAP-325" ~ 'PI576373', 
                              genotype=="SAP-340" ~ 'PI595699', genotype=="SAP-342" ~ 'PI576347', genotype=="SAP-347" ~ 'PI576386', 
                              genotype=="SAP-380" ~ 'PI655982', genotype=="SAP-386" ~ 'PI609456', genotype=="SAP-395" ~ 'PI597965',
                              genotype=="SAP-398" ~ 'PI597972', genotype=="SAP-404" ~ 'PI533799', genotype=="SAP-50" ~ 'PI655981', 
                              genotype=="SB-283" ~ 'PI533967', genotype=="SAP-250" ~ 'PI597950', genotype=="SC 0386" ~ 'PI656119', 
                              genotype=="SC 0332" ~ 'PI656118', genotype=="SC 0172" ~ 'PI656117', genotype=="SC 0525" ~ 'PI656101', 
                              genotype=="SC 0498" ~ 'PI656099', genotype=="SC 0301" ~ 'PI656094', genotype=="SC 0145" ~ 'PI656082', 
                              genotype=="SC 0480" ~ 'PI656097', genotype=="SC 1451" ~ 'PI656083', genotype=="SC 1439" ~ 'PI656081', 
                              genotype=="SC 1429" ~ 'PI656080', genotype=="SC 1426" ~ 'PI656079', genotype=="SC 1424" ~ 'PI656078',
                              genotype=="SC 1271" ~ 'PI656076', genotype=="SC 1251" ~ 'PI656075', genotype=="SC 1074" ~ 'PI656073', 
                              genotype=="SC 1019" ~ 'PI656071', genotype=="SC 1047" ~ 'PI656072', genotype=="SC 1476" ~ 'PI656087',
                              genotype=="SC 0621" ~ 'PI656104', genotype=="SC 0639" ~ 'PI656105', genotype=="SC 0695" ~ 'PI656106', 
                              genotype=="SC 0971" ~ 'PI656111', genotype=="SC 1215" ~ 'PI656112', genotype=="SC 1440" ~ 'PI656115', 
                              genotype=="SC 0947" ~ 'PI656121', genotype=="SC 1484" ~ 'PI656088', genotype=="SC 1489" ~ 'PI656089', 
                              genotype=="SC 1494" ~ 'PI656090', genotype=="Segaolane" ~ 'PI656023', genotype=="SC 170-6-17" ~ 'PI656068',
                              genotype=="SC 326-6" ~ 'PI656069', genotype=="SC 748-5" ~ 'PI656070', genotype=="SEPON82" ~ 'PI656024', 
                              genotype=="Shan Qui Red" ~ 'PI656025', genotype=="Sinidyil 177" ~ 'PI533991', genotype=="SO 16" ~ 'PI533785', 
                              genotype=="SOBERANO" ~ 'PI656026', genotype=="Sorghum Soroti" ~ 'PI533833', genotype=="SPUR FETERITA" ~ 'PI655973',
                              genotype=="Standard Blackhull Kafir" ~ 'PI655970', genotype=="STANDARD WHITE MILO" ~ 'PI655971', 
                              genotype=="SUGAR DRIP" ~ 'PI586435', genotype=="SUMAC" ~ 'PI35038', genotype=="SURENO" ~ 'PI561472', 
                              genotype=="SV 34" ~ 'PI533843', genotype=="T 28" ~ 'PI534053', genotype=="TAM2566" ~ 'PI655977', 
                              genotype=="Tambroro 7" ~ 'PI533762', genotype=="TEXAS BLACKHULL KAFIR" ~ 'PI655974', genotype=="Town" ~ 'PI656028',
                              genotype=="Tuery 11" ~ 'PI534139', genotype=="TX 3197 (COMB KAFIR 60) B LINE" ~ 'PI655992', 
                              genotype=="TX 378 (REDLAN) B LINE" ~ 'PI655991', genotype=="TX 399(DALHART RES WHTLND)B LN" ~ 'PI655993', 
                              genotype=="TX2783" ~ 'PI656001', genotype=="Tx2891" ~ 'PI548797', genotype=="Tx2911" ~ 'PI607931', 
                              genotype=="Tx2917" ~ 'PI629040', genotype=="WHEATLAND" ~ 'PI655975', genotype=="WILEY" ~ 'PI655994', 
                              genotype=="Wit Lichtenburg DL/59/1530" ~ 'PI533961', genotype=="WRAY_(XK: fromBIllRooney_TAMU_2024)" ~ 'PI653616',
                              genotype=="ZA 71" ~ 'PI534092', .default = genotype))

vp <- partitionVariance3(embeddings_all, high_fi_features[1], label = high_fi_features[1], 
                         modelStatement = '~ (1|genotype) + (1|location) + (1|location:range) + (1|location:row) + (1|location:block) + (1|genotype:location)')
for(i in 2:length(high_fi_features))
{
  vp <- bind_rows(vp, 
                  partitionVariance3(embeddings_all, high_fi_features[i], label = high_fi_features[i], 
                                     modelStatement = '~ (1|genotype) + (1|location) + (1|location:range) + (1|location:row) + (1|location:block) + (1|genotype:location)'))
}
vp_summary <- vp %>% 
  select(grp, pctVar, label) %>%
  pivot_wider(id_cols = label, 
              values_from = pctVar, 
              names_from = grp) %>% 
  arrange(desc(genotype), desc(location), desc(`genotype:location`), desc(`location:block`), desc(`location:range`), desc(`location:row`), desc(Residual))

vp <- vp %>% 
  mutate(grp = factor(grp, levels = c('Residual', 'location:row', 'location:range', 'location:block', 'location', 'genotype:location', 'genotype')), 
         label = factor(label, levels = vp_summary$label))

vp.plot <- ggplot(vp, aes(label, pctVar, fill = grp)) + 
  geom_col() + 
  scale_x_discrete(expand = c(0, 0), name = 'Embedding') + 
  scale_y_continuous(expand = c(0, 0), labels = ~str_c(.x, '%'), name = 'Variance Explained') + 
  scale_fill_manual(values = paletteer_d('MetBrewer::Archambault', 7, direction = -1), 
                    labels = c('Residual', 'Row', 'Range', 'Block', 'Genotype x Location', 'Location', 'Genotype'),
                    name = NULL) +
  theme_use + 
  theme(axis.text.x = element_text(angle=90))
vp.plot
ggsave('output/high_fi_vp.png', plot = vp.plot, width = 8, height = 4.5, dpi = 1000, bg = 'transparent')

snps_selected <- c('S05_453076', 'S03_68036482', 'S07_60649975', 'S010_56042856')
vcf_sig <- read_tsv('output/selected_sig_snps.recode.vcf', skip = 23)
colnames(vcf_sig) <- c('CHROM', colnames(vcf_sig)[2:11], str_remove(colnames(vcf_sig)[12:730], 'ExPVP_'), str_replace(colnames(vcf_sig)[731:815], 'SC', 'SC '))
indivs_keep <- read_csv('output/sam3_genotypes_keep.txt', col_names = 'genotype')$genotype %>% 
  str_replace('SC', 'SC ') %>% 
  str_remove('ExPVP_')

vcf_sig <- vcf_sig[, c('CHROM', 'POS', 'ID', 'REF', 'ALT', indivs_keep)] %>%
  pivot_longer(cols = !c(CHROM, POS, ID, REF, ALT), 
               names_to = 'genotype', 
               values_to = 'allele') %>% 
  filter(!(allele %in% c('1|0', '0|1'))) %>% 
  select(ID, genotype, allele) %>% 
  pivot_wider(id_cols = genotype, 
              names_from = ID, 
              values_from = allele)

ordinal_scores <- read_csv('data/manual/scores_828.csv') %>% 
  mutate(genotype = str_replace(genotype, 'PI ', 'PI')) %>% 
  left_join(vcf_sig, join_by(genotype))

samplePhenotypesForResampling('output/scores_828_blues.csv', genotype = 'genotype', trait = 'score_average')
samplePhenotypesForResampling('output/scores_813_blues.csv', genotype = 'genotype', trait = 'score_average')

all_farmcpu_hits_828 <- summariseSignals_PANICLE('output/gwas/scores_828/GWAS_score_average_*')
write_csv(all_farmcpu_hits_828, 'output/scores_828_allfarmcpuhits.csv')

all_farmcpu_hits_828 <- read_csv('output/scores_828_allfarmcpuhits.csv')

rmip_828 <- all_farmcpu_hits_828 %>% 
  group_by(SNP, CHROM, POS) %>% 
  summarise(RMIP = n()/100, 
            min_p = min(pval, na.rm = TRUE), 
            mean_effect = mean(effect, na.rm = TRUE)) %>% 
  arrange(desc(RMIP))
plotManhattan(rmip_828, RMIP, multitrait = FALSE, resampling = TRUE, threshold = 0.2, main = 'Mean Anthracnose Severity Ordinal Score \n8/28', colors = paletteer_d("rcartocolor::Prism", 10), theme = theme_use, species = 'sorghum')

all_farmcpu_hits_813 <- summariseSignals_PANICLE('output/gwas/scores_813/GWAS_score_average_*')
write_csv(all_farmcpu_hits_813, 'output/scores_813_allfarmcpuhits.csv')

all_farmcpu_hits_813 <- read_csv('output/scores_813_allfarmcpuhits.csv')

rmip_813 <- all_farmcpu_hits_813 %>% 
  group_by(SNP, CHROM, POS) %>% 
  summarise(RMIP = n()/100, 
            min_p = min(pval, na.rm = TRUE), 
            mean_effect = mean(effect, na.rm = TRUE)) %>% 
  arrange(desc(RMIP))
plotManhattan(rmip_813, RMIP, multitrait = FALSE, resampling = TRUE, threshold = 0.2, main = 'Mean Anthracnose Severity Ordinal Score \n8/13', colors = paletteer_d("rcartocolor::Prism", 10), theme = theme_use, species = 'sorghum')

ordinal_vcf_sig <- read_tsv('output/selected_sig_snps_828.recode.vcf', skip = 23)
colnames(ordinal_vcf_sig) <- c('CHROM', colnames(ordinal_vcf_sig)[2:11], str_remove(colnames(ordinal_vcf_sig)[12:730], 'ExPVP_'), str_replace(colnames(ordinal_vcf_sig)[731:815], 'SC', 'SC '))
indivs_keep <- read_csv('output/sam3_genotypes_keep.txt', col_names = 'genotype')$genotype %>% 
  str_replace('SC', 'SC ') %>% 
  str_remove('ExPVP_')

ordinal_vcf_sig <- ordinal_vcf_sig[, c('CHROM', 'POS', 'ID', 'REF', 'ALT', indivs_keep)] %>%
  pivot_longer(cols = !c(CHROM, POS, ID, REF, ALT), 
               names_to = 'genotype', 
               values_to = 'allele') %>% 
  filter(!(allele %in% c('1|0', '0|1'))) %>% 
  select(ID, genotype, allele) %>% 
  pivot_wider(id_cols = genotype, 
              names_from = ID, 
              values_from = allele)

vcf_combined <- left_join(vcf_sig, ordinal_vcf_sig, join_by(genotype)) %>% 
  mutate(across(!c(genotype), ~as.numeric(str_sub(.x, 1, 1))))

for(snp in snps_selected)
{
  df <- filter(ordinal_scores, !is.na(ordinal_scores[[snp]]))
  model <- lm(score_average ~ df[[snp]], data = df)
  a <- anova(model)
  print(str_c('SNP: ', snp, ' pval: ', a$`Pr(>F)`))
  
  p <- ggplot(df, aes(.data[[snp]], score_average, fill = .data[[snp]])) + 
    geom_boxplot() + 
    scale_x_discrete(name = str_c(snp, ' Allele'), labels = c('REF', 'ALT')) + 
    scale_y_continuous(name = 'Mean Anthracnose Severity Ordinal Score') +
    scale_fill_paletteer_d("rcartocolor::Prism", 
                           guide = NULL) + 
    theme_use
  print(p)
  ggsave(str_c('output/', snp, '_ordinal_score_boxplot.png'), plot = p, dpi = 1000)
}

embeddings_selected <- embeddings_all %>% 
  select(genotype, location, all_of(select_features)) %>%
  left_join(vcf_sig, join_by(genotype)) %>% 
  pivot_longer(starts_with('S'), 
               names_to = 'SNP', 
               values_to = 'allele') %>% 
  filter(!is.na(allele))

boxplot_features <- c('embedding_std_976', 'embedding_mean_119', 'embedding_std_566')
for(feature in boxplot_features)
{
  p <- ggplot(embeddings_selected, aes(allele, .data[[feature]], fill = allele)) + 
    facet_grid(rows = vars(SNP), cols = vars(location)) + 
    geom_boxplot() + 
    scale_x_discrete(name = 'Allele', labels = c('REF', 'ALT')) + 
    scale_fill_paletteer_d('rcartocolor::Prism', guide = NULL) + 
    labs(title = feature) +
    theme_use
  print(p)
}

for(loc in c('AAMU', 'FVSU'))
{
  for(feature in boxplot_features)
  {
    for(snp in snps_selected)
    {
      df <- filter(embeddings_selected, location==loc & SNP==snp)
      df %>% group_by(allele) %>% count() %>% print()
      model <- lm(df[[feature]] ~ allele, data = df)
      a <- anova(model)
      print(str_c('Pval for ', snp, ' allele for ', feature, ' at ', loc, ': ', a$`Pr(>F)`, ' ', (a$`Pr(>F)` < 0.05/12)))
    }
  }
}

gff <- read.gff('data/genotype/Sbicolor_730_v5.1.gene.gff3')
gff_genes <- filter(gff, type=='gene') %>% 
  mutate(CHROM = str_remove(seqid, 'Chr') %>% 
           as.numeric(), 
         gene_id = str_split_i(attributes, ';', 2) %>% 
           str_remove('Name='))
annotation <- read_tsv('data/genotype/Sbicolor_730_v5.1.P14.annotation_info.txt')
defline <- read_tsv('data/genotype/Sbicolor_730_v5.1.P14.defline.txt', col_names = c('transcript_id', 'attribute', 'value')) %>% 
  pivot_wider(id_cols = transcript_id, 
              names_from = attribute, 
              values_from = value) %>% 
  rename(description = defLine, 
         auto_defline = pdef) %>% 
  mutate(gene_id = str_sub(transcript_id, 1,16), 
         description = str_trim(description), 
         auto_defline = str_trim(auto_defline))

annotation_full <- gff_genes %>% 
  left_join(annotation, join_by(gene_id==locusName)) %>% 
  left_join(defline, join_by(gene_id, transcriptName==transcript_id)) %>% 
  rename(transcript_id = transcriptName)

annotation_chr3 <- filter(annotation_full, 
                          seqid=='Chr03' & 
                            (between(start, 68036482 - 80772, 68036482 + 80772) | between(end, 68036482 - 80772, 68036482 + 80772))) %>%
  arrange(start, end)

annotation_chr5 <- filter(annotation_full, 
                         seqid=='Chr05' & 
                           (between(start, 453076 - 2.4e3, 453076 + 2.4e3) | between(end, 453076 - 2.4e3, 453076 + 2.4e3))) %>%
  arrange(start, end)

annotation_chr7 <- filter(annotation_full, 
                          seqid=='Chr07' & 
                            (between(start, 60649975 - 156454, 60649975 + 156454) | between(end, 60649975 - 156454, 60649975 + 156454))) %>%
  arrange(start, end)

annotation_chr10 <- filter(annotation_full, 
                          seqid=='Chr10' & 
                            (between(start, 56042856 - 38034, 56042856 + 38034) | between(end, 56042856 - 38034, 56042856 + 38034))) %>%
  arrange(start, end)


chr3_119_aamu <- filter(embeddings_selected, SNP==snps_selected[2] & location=='AAMU') %>% 
  ggplot(aes(allele, embedding_mean_119, fill = allele)) + 
  geom_boxplot() + 
  scale_x_discrete(name = str_c(snps_selected[2], ' Allele'), labels = c('REF', 'ALT')) + 
  scale_fill_manual(values = paletteer_d('RColorBrewer::Paired', 10)[1:2], 
                    guide = NULL) +  
  theme_use
chr3_119_aamu
ggsave(str_c('output/', snps_selected[2], '_119_aamu_boxplot.png'), dpi = 1000, bg = 'transparent')

chr5_976_fvsu <- filter(embeddings_selected, SNP==snps_selected[1] & location=='FVSU') %>% 
  ggplot(aes(allele, embedding_std_976, fill = allele)) + 
  geom_boxplot() + 
  scale_x_discrete(name = str_c(snps_selected[1], ' Allele'), labels = c('REF', 'ALT')) + 
  scale_fill_manual(values = paletteer_d('RColorBrewer::Paired', 10)[9:10], 
                    guide = NULL) +  
  theme_use
chr5_976_fvsu
ggsave(str_c('output/', snps_selected[1], '_976_fvsu_boxplot.png'), dpi = 1000, bg = 'transparent')

chr7_976_aamu <- filter(embeddings_selected, SNP==snps_selected[3] & location=='AAMU') %>% 
  ggplot(aes(allele, embedding_std_976, fill = allele)) + 
  geom_boxplot() + 
  scale_x_discrete(name = str_c(snps_selected[3], ' Allele'), labels = c('REF', 'ALT')) + 
  scale_fill_manual(values = paletteer_d('RColorBrewer::Paired', 10)[9:10], 
                         guide = NULL) + 
  theme_use
chr7_976_aamu
ggsave(str_c('output/', snps_selected[3], '_976_aamu_boxplot.png'), dpi = 1000, bg = 'transparent')

chr10_119_fvsu <- filter(embeddings_selected, SNP==snps_selected[4] & location=='FVSU') %>% 
  ggplot(aes(allele, embedding_mean_119, fill = allele)) + 
  geom_boxplot() + 
  scale_x_discrete(name = str_c(snps_selected[4], ' Allele'), labels = c('REF', 'ALT')) + 
  scale_fill_manual(values = paletteer_d('RColorBrewer::Paired', 10)[1:2], 
                    guide = NULL) +  
  theme_use
chr10_119_fvsu
ggsave(str_c('output/', snps_selected[4], '_119_fvsu_boxplot.png'), dpi = 1000, bg = 'transparent')
