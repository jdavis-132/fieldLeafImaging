library(tidyverse)
source('src/R/Functions.R')
library(paletteer)
library(cowplot)
library(readxl)
library(glue)


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

all_farmcpu_hits <- read_csv('output/farmcpu_20260515/all_farmcpu_hits.csv')

rmip <- all_farmcpu_hits %>% 
  mutate(embedding = case_when(str_detect(embedding, 'human_scores') ~ 'human_scores', 
                               str_detect(embedding, 'pctd') ~ 'pctd', 
                               .default = embedding)) %>% 
  group_by(SNP, CHROM, POS, embedding) %>% 
  summarise(RMIP = n()/100, 
            min_p = min(FarmCPU_P, na.rm = TRUE), 
            mean_effect = mean(FarmCPU_Effect, na.rm = TRUE)) %>% 
  arrange(desc(RMIP))

rmip_pctd <- read_csv('output/farmcpu_20260515/pctd_farmcpu_hits.csv') %>% 
  mutate(embedding = 'pctd') %>% 
  group_by(SNP, CHROM, POS, embedding) %>% 
  summarise(RMIP = n()/100, 
            min_p = min(FarmCPU_P, na.rm = TRUE), 
            mean_effect = mean(FarmCPU_Effect, na.rm = TRUE)) %>% 
  arrange(desc(RMIP))

rmip_scores <- read_csv('output/farmcpu_20260515/human_scores_farmcpu_hits.csv') %>% 
  mutate(embedding = 'human_score') %>%
  group_by(SNP, CHROM, POS, embedding) %>% 
  summarise(RMIP = n()/100, 
            min_p = min(FarmCPU_P, na.rm = TRUE), 
            mean_effect = mean(FarmCPU_Effect, na.rm = TRUE)) %>% 
  arrange(desc(RMIP))

rmip_all <- bind_rows(rmip, rmip_pctd, rmip_scores)

fi_pctd_features <- read_csv('output/rf_20260507/sam3_human_scores_embedding_feature_importances_rf.csv') %>% 
  pivot_longer(cols = everything(), names_to = 'feature', values_to = 'fi') %>% 
  group_by(feature) %>%
  summarise(avg_fi = mean(fi, na.rm = TRUE)) %>% 
  arrange(desc(avg_fi)) %>% 
  mutate(feature = as.numeric(feature)) %>% 
  mutate(stat = case_when(feature < 1024 ~ 'mean', .default = 'std'), 
         embedding_num = case_when(feature > 1023 ~ feature - 1024, .default = feature))

high_fi <- fi_pctd_features %>% 
  arrange(desc(avg_fi))
high_fi <- high_fi[1:47, ]

high_fi_features <- str_c('embedding', high_fi$stat, high_fi$embedding_num, sep = '_')

annotation <- read_csv('data/sorghum_v5.1_annotation_combined.csv')

validatedRGenes <- read_excel('data/20260413_anthracnoseResistanceCandidates.xlsx')
prevCandidateRGenes <- read_excel('data/20260413_anthracnoseResistanceCandidates.xlsx', sheet = 'candidates')
validatedRGenes_all_diseases <- read_excel('data/20260413_anthracnoseResistanceCandidates.xlsx', sheet = 'validated_all_diseases')
candidates_all_diseases <- read_excel('data/20260413_anthracnoseResistanceCandidates.xlsx', sheet = 'candidates_all_diseases')
kim_candidates <- read_excel('data/anthracnose candidate genes in sorghum_2026 v2.xlsx')

rmip_0.2 <-  filter(rmip_all, RMIP > 0.2)
filter(rmip_0.2, str_detect(embedding, 'embedding')) %>% 
  mutate(stat = str_split_i(embedding, '_', 2) %>% 
           str_to_title() %>% 
           str_replace('Std', 'SD'), 
         feature = str_split_i(embedding, '_', 3), 
         label = str_c(feature, ' (', stat, ')')) %>% 
  select(SNP, CHROM, POS, RMIP, label) %>% 
  write.table('figures/supplemental/all_sig_hits_embeddings.csv', 
            col.names = c('Marker', 'Chromosome', 'Position', 'RMIP', 'Embedding'),
            row.names = FALSE, quote = FALSE, sep = ',')
  
rmip_0.2SNPs <- rmip_0.2 %>% 
  distinct(SNP, CHROM, POS) %>% 
  arrange(CHROM, POS) 

rmip_0.2SNPs %>% 
  ungroup() %>% 
  select(CHROM, POS) %>%
  write_tsv('output/snps_keep.txt', col_names = FALSE)

for(p in 1:nrow(rmip_0.2SNPs))
{
  candidate_genes <- getCandidateGenes(annotations = annnotation, 
                    outdir = 'output/farmcpu_20260515/', 
                    chr = rmip_0.2SNPs$CHROM[p], 
                    pos = rmip_0.2SNPs$POS[p], 
                    window = 2e5)
  knownRGenes <- intersect(validatedRGenes$gene_id_v5, candidate_genes$gene_id)
  proposedRGenes <- intersect(prevCandidateRGenes$gene_id_v5, candidate_genes$gene_id)
  kimCandidates <- intersect(kim_candidates$gene, candidate_genes$gene_id)
  validated_other_diseases <- intersect(validatedRGenes_all_diseases$gene_id_v5, candidate_genes$gene_id)
  candidates_other_diseases <- intersect(candidates_all_diseases$gene_id_v5, candidate_genes$gene_id)
  if(length(knownRGenes) > 0 | length(proposedRGenes) > 0 | length(kimCandidates) > 0 | 
     length(validated_other_diseases) > 0 | length(candidates_all_diseases) > 0)
  {
    print(rmip_0.2SNPs$SNP[p])
    if(length(knownRGenes) > 0)
    {
      print('Known R Genes w/in 200kb: ')
      print(knownRGenes)
    }
    if(length(proposedRGenes) > 0)
    {
      print('Previously Proposed R Genes w/in 200kb: ')
      print(proposedRGenes)
    }
    if(length(kimCandidates) > 0)
    {
      print('Kim Candidate R Genes w/in 200kb: ')
      print(kimCandidates)
    }
    if(length(validated_other_diseases) > 0)
    {
      print('Validated R Genes for other diseases w/in 200kb: ')
      print(validated_other_diseases)
    }
    if(length(candidates_other_diseases) > 0)
    {
      print('Candidate R Genes for other diseases w/in 200kb: ')
      print(candidates_other_diseases)
    }
  }
}

near_known_gene <- tibble(Marker = c("1:65781144:G:GT", "5:60007887:G:C", '9:60035864:A:T', '9:60186716:T:A'), 
                          Gene = c("Sobic.001G329100", "Sobic.005G126200", 'Sobic.009G217900', 'Sobic.009G217900, Sobic.009G221266'), 
                          Embedding = c('839 (SD)', "792 (Mean), 984 (Mean), 793 (SD), 119 (Mean)", '383 (SD), 983 (SD), 326 (SD), 37 (Mean)', 
                                        '734 (Mean)'),
                          Reference = c('Chen et al. (2024)', "Zhu et al. (2013, 2015)", 'Biruma et al. (2012)', 'Biruma et al. (2012)'))
write_csv(near_known_gene, 'figures/supplemental/hits_near_known_Rgenes.csv')

near_proposed_gene <- tibble(Marker = c("2:1961996:T:A", "2:7435977:G:A", "3:59834531:G:A", '7:63087208:C:T', '8:63746589:G:A', '9:60035864:A:T', '9:60186716:T:A', 
                                        '10:60309710:C:G'), 
                             Gene = c("Sobic.002G022600", "Sobic.002G072000", "Sobic.003G203500", 'Sobic.007G163100', 'Sobic.008G174200', 'Sobic.009G221000', 
                                      'Sobic.009G221000', 'Sobic.010G248500'), 
                             Embedding = c("128 (SD)", "793 (SD)", "918 (SD)", '930 (Mean)', '975 (SD)', '383 (SD), 983 (SD), 326 (SD), 37 (Mean)', '734 (Mean)',
                                           '637 (Mean)'),
                             Reference = c("Birhanu et al. (2024)", "Zhang et al. (2025)", "Ahn et al. (2021b)", 'Ahn et al. (2026)', 'Zhang et al. (2025)', 
                                           'Zhang et al. (2025)', 'Zhang et al. (2025)', 'Zhang et al. (2025)'))
write_csv(near_proposed_gene, 'figures/supplemental/hits_near_proposed_Rgenes.csv')

known_candidates <- c('embedding_std_839', 'embedding_mean_792', 'embedding_mean_984', 'embedding_std_793', 'embedding_mean_119', 'embedding_std_383', 
                      'embedding_std_983', 'embedding_std_326', 'embedding_mean_37', 'embedding_mean_734')
proposed_candidates <- c('embedding_std_793', 'embedding_std_918', 'embedding_mean_930', 'embedding_std_975', 'embedding_std_983',
                         'embedding_std_326', 'embedding_mean_37', 'embedding_mean_734', 'embedding_mean_637')

new_candidate_markers <- c('2:1961996:T:A', '2:2510254:C:T', '2:2512849:C:T', '3:12740766:G:C', '7:4717906:C:G', '8:2354126:C:T', '9:62154422:T:C',
                           '9:62272161:C:A', '9:62299247:C:T')
new_candidates_embeddings <- c('embedding_std_128', 'embedding_mean_637', 'embedding_std_930', 'embedding_mean_968', 'embedding_std_132', 'human_score',
                               'embedding_std_793', 'embedding_mean_586', 'embedding_std_128', 'embedding_std_128')
new_candidate_genes <- c('Sobic.002G023200, Sobic.002G023300', 'Sobic.002G027700',  'Sobic.002G027700', 'Sobic.003G133800', 'Sobic.007G045800', 
                         'Sobic.008G027166, Sobic.008G027200, Sobic.008G027300, Sobic.008G027350, Sobic.008G027400, Sobic.008G027500',
                         'Sobic.009G249100', 'Sobic.009G249100', 'Sobic.009G251800')
new_candidate_annotation <- c('similar to Pathogenesis related protein-1', 'RECEPTOR-LIKE PROTEIN 57; AT ortho has LRR', 
                              'RECEPTOR-LIKE PROTEIN 57; AT ortho has LRR', 'LRR', 'RLK', 
                              'LRR; DISEASE RESISTANCE RPP13-LIKE PROTEIN 1 & DISEASE RESISTANCE PROTEIN RPS4-RELATED (AT)',
                              'WRKY TF; similar to Disease resistance protein-like', 'WRKY TF; similar to Disease resistance protein-like',
                              'LRR')

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

blues_unl <- read_csv('output/sam3_blues.csv') %>% 
  mutate(location = 'UNL')
genotypes_unl <- blues_unl$genotype
blues_fvsu <- read_csv('output/sam3_fvsu_blues.csv') %>% 
  mutate(location = 'FVSU')
genotypes_fvsu <- blues_fvsu$genotype
blues_aamu <- read_csv('output/sam3_aamu_blues.csv') %>% 
  mutate(location = 'AAMU')
genotypes_aamu <- blues_aamu$genotype
blues_all <- bind_rows(blues_unl, blues_fvsu, blues_aamu) %>%
  filter(!genotype %in% c('Border', 'Check', 'Fill', 'Mixed')) %>% 
  left_join(vcf_sig, join_by(genotype)) %>% 
  mutate(across(all_of(markers), ~case_when(.x %in% c('0/1', '1/0', './.', '.', '0|1') ~ NA,
                                            .x=='1|1' ~ '1/1',
                                            .x=='0|0' ~ '0/0',
                                            .default = .x))) %>%
  select(c(location, genotype, all_of(c(markers, high_fi_features))))

genotypes_intersect <- intersect(blues_unl$genotype, blues_aamu$genotype)
genotypes_intersect <- intersect(genotypes_intersect, blues_fvsu$genotype)
blues_unl_intersect <- blues_all %>% 
  filter(location=='UNL' & genotype %in% genotypes_intersect) %>% 
  mutate(location = 'UNL-SAP')
blues_all <- bind_rows(blues_all, blues_unl_intersect) %>% 
  mutate(across(all_of(markers), ~factor(.x, levels = c('0/0', '1/1'))))

# ne_field_idx <- read_csv('data/ne2025/SbDiv_ne2025_fieldindex.csv') %>%
#   mutate(block = as.numeric(rep)) %>%
#   select(!rep)%>% 
#   mutate(genotype = str_remove_all(genotype, ' ')) %>%
#   left_join(genotype_alignment, join_by(genotype==genotype_idx)) %>%
#   mutate(genotype_markers = case_when(is.na(genotype_markers) ~ genotype,
#                                       .default = genotype_markers))

genotype_alignment <- read_tsv('data/genotype_conversion_table.tsv', col_names = c('genotype_idx', 'genotype_markers')) %>% 
  mutate(genotype_idx = str_remove_all(genotype_idx, ' '))

image_scores_lg30 <- read_csv('data/manual/20260318_ScoresSorghum_LFGT.csv') %>%
  filter(!(project=='UNL')) %>% 
  mutate(timestamp= as.character(timestamp))
image_scores_rr30 <- read_csv('data/manual/20260319_ScoresSorghum_RRR.csv') %>%
  filter(!(project=='UNL')) %>% 
  mutate(timestamp= as.character(timestamp))
image_scores_lg_192_fvsu <- read_excel('data/fvsu2025/20260518_192scores.xlsx') %>% 
  mutate(project = 'FVSU')
image_scores_lg_186_aamu <- read_excel('data/aamu2025/20260518_186scores.xlsx') %>% 
  mutate(project = 'AAMU') %>% 
  filter(score!='skip') %>% 
  mutate(score = as.numeric(score))
image_scores_rr_1 <- read_csv('data/manual/20260520_RRR_1.csv') %>% 
  filter(project!='sap135images') %>% 
  mutate(project = 'AAMU') %>% 
  mutate(timestamp= as.character(timestamp))
image_scores_rr_2 <- read_csv('data/manual/20260520_RRR_2.csv') %>% 
  mutate(project = 'FVSU') %>% 
  mutate(timestamp= as.character(timestamp))

idx_aamu <- read_csv('data/aamu2025/aamu_field_index.csv')
idx_fvsu <- read_csv('data/fvsu2025/fvsu_field_index.csv') %>%
  mutate(block = str_split_i(rep, fixed(' '), 2) %>%
           as.numeric()) %>%
  select(!c(S.No, rep))

image_scores <- bind_rows(image_scores_lg30, image_scores_rr30, image_scores_lg_186_aamu, image_scores_lg_192_fvsu, image_scores_rr_1, image_scores_rr_2) %>% 
  mutate(plotNumber = str_split_i(image, '_', 1) %>% 
           as.numeric()) %>% 
  rename(location = project)
write_csv(image_scores, 'data/manual/image_scores_al_ga.csv')

image_scores_aamu <- filter(image_scores, location=='AAMU') %>% 
  left_join(idx_aamu, join_by(plotNumber), relationship = 'many-to-one')
image_scores_fvsu <- filter(image_scores, location=='FVSU') %>% 
  left_join(idx_fvsu, join_by(plotNumber))

image_scores <- bind_rows(image_scores_aamu, image_scores_fvsu) %>% 
  mutate(genotype = str_remove_all(genotype, ' ')) %>%
  left_join(genotype_alignment, join_by(genotype==genotype_idx)) %>%
  mutate(genotype_markers = case_when(is.na(genotype_markers) ~ genotype,
                                      .default = genotype_markers)) %>% 
  filter(!is.na(genotype_markers)) %>%
  select(!c(genotype)) %>% 
  rename(genotype = genotype_markers) %>%
  right_join(vcf_sig, join_by(genotype)) %>% 
  mutate(across(all_of(markers), 
                ~case_when(.x %in% c('0/1', '1/0', './.', '.', '0|1') ~ NA, 
                           .x=='1|1' ~ '1/1', 
                           .x=='0|0' ~ '0/0',
                           .default = .x)))
scores_sig_aamu <- c()
scores_sig_fvsu <- c()
for(m in markers)
{
  test_formula <- as.formula(str_c('score ~ `', m, '`'))
  print(m)
  aamu_test <- wilcox.test(test_formula, data = image_scores, subset = location=='AAMU')
  fvsu_test <- wilcox.test(test_formula, data = image_scores, subset = location=='FVSU')
  if(fvsu_test$p.value < 0.05)
  {
    print(str_c('FVSU: ', fvsu_test$p.value))
    scores_sig_fvsu <- c(scores_sig_fvsu, m)
  }
  if(aamu_test$p.value < 0.05)
  {
    print(str_c('AAMU: ', aamu_test$p.value))
    scores_sig_aamu <- c(scores_sig_aamu, m)
  }
}

df_bar <- image_scores %>% 
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
            n = n())

blues_summary <- blues_all %>% 
  pivot_longer(!c(location, starts_with('embedding')), 
               values_to = 'allele', 
               names_to = 'marker') %>% 
  filter(!is.na(allele)) %>% 
  group_by(marker, allele, location) %>% 
  summarise(across(all_of(high_fi_features),
                   .fns = c(mean = ~mean(.x),
                            se = ~sd(.x)/sqrt(n()), 
                            n = ~n()))) %>%
  mutate(location = factor(location, levels = c('UNL', 'UNL-SAP', 'AAMU', 'FVSU'))) 

# test stability
threshold <- 0.05
stable_dir <- c()
sig_unl <- c()
sig_unlsap <- c()
sig_aamu <- c()
sig_fvsu <- c()
for(i in 1:57)
{
  m <- rmip_0.2$SNP[i]
  embedding <- rmip_0.2$embedding[i]
  print(str_c(embedding, m, sep = ':'))
  
  model <- as.formula(str_c(embedding, '~ `', m, '`'))
  unl_test <- wilcox.test(model, data = blues_all, subset = location=='UNL', conf.int = TRUE)
  unl_sap_test <- wilcox.test(model, data = blues_all, subset = location=='UNL-SAP', conf.int = TRUE)
  aamu_test <- wilcox.test(model, data = blues_all, subset = location=='AAMU', conf.int = TRUE)
  fvsu_test <- wilcox.test(model, data = blues_all, subset = location=='FVSU', conf.int = TRUE)
  
  if((unl_test$estimate*aamu_test$estimate > 0) & (unl_test$estimate*fvsu_test$estimate > 0))
  {
    stable_dir <- c(stable_dir, i)
  }
  
  if(unl_test$p.value < threshold){sig_unl <- c(sig_unl, i)}
  if(unl_sap_test$p.value < threshold){sig_unlsap <- c(sig_unlsap, i)}
  if(aamu_test$p.value < threshold){sig_aamu <- c(sig_aamu, i)}
  if(fvsu_test$p.value < threshold){sig_fvsu <- c(sig_fvsu, i)}
}

sig_both <- intersect(sig_aamu, sig_fvsu)

for(i in c(19, 23, 31, 37, 38, 45, 51))
{
  m <- rmip_0.2$SNP[i]
  embedding <- rmip_0.2$embedding[i]
  mean <- str_c(embedding, '_mean')
  se <- str_c(embedding, '_se')
  plot <- filter(blues_summary, marker==m) %>% 
    ggplot(aes(location, .data[[mean]], fill = allele)) + 
      geom_col(position = position_dodge(width = 0.9)) + 
      geom_errorbar(aes(ymin = .data[[mean]] - .data[[se]], ymax = .data[[mean]] + .data[[se]]), 
                    width = 0.25, position = position_dodge(width = 0.9)) + 
      theme_use
  print(plot)
}

dir_switch_sig <- c(19, 31, 51)

select_features <- c('embedding_std_793', 'embedding_std_383', 'embedding_std_326', 'embedding_std_983', 'embedding_mean_734', 
                     'embedding_mean_637', 'embedding_mean_792', 'embedding_mean_984', 'embedding_mean_119', 'embedding_std_128')

# features_ordered <-  c('119 (Mean)', '128 (SD)', '326 (SD)', '383 (SD)', '637 (Mean)', 
#                        '734 (Mean)', '792 (Mean)', '793 (SD)', '983 (SD)', '984 (Mean)')
features_ordered <-  c('119 (Mean)', '128 (SD)', '792 (Mean)', '637 (Mean)', '326 (SD)',
                       '793 (SD)', '383 (SD)', '983 (SD)', '734 (Mean)', '984 (Mean)')
n_features <- length(select_features)
rmip_select <- rmip %>% 
  filter(embedding %in% select_features) %>% 
  ungroup() %>% 
  mutate(CHROM = str_remove(CHROM, 'Chr') %>% 
           as.numeric(), 
         stat = str_split_i(embedding, '_', 2),
         feature = str_split_i(embedding, '_', 3),
         label = str_c(feature, ' (', str_to_title(stat), ')') %>%
         str_replace('Std', 'SD')) %>%
  mutate(label = factor(label, levels = features_ordered))

lrr2_chr5loc <- mean(59834505:59836269) + 348316776
cdl1_chr9loc <- mean(60010749:60014092) + 654779914
cs1a_chr9loc <- mean(60240330:60245571) + 654779914

manhattan <- plotManhattan(rmip_select, RMIP, multitrait = TRUE, trait = label, threshold = 0.2, 
                           colors = paletteer_d("RColorBrewer::Paired", 10),
                           species = 'sorghum', theme = theme_use, chrGap = 8e6) + 
  annotate('point', x = c(lrr2_chr5loc, cdl1_chr9loc, cs1a_chr9loc), y = rep(0, 3), size = 4, color = 'blue', shape = 17)
# manhattan

ggsave('figures/main/figure4/select_embeddings_farmcpu.png', plot = manhattan, width = 5, height = 3.25, dpi = 1000,
       bg = 'transparent')

markers_128 <- rmip_0.2$SNP[which(rmip_0.2$embedding=='embedding_std_128')]
markers_637 <- rmip_0.2$SNP[which(rmip_0.2$embedding=='embedding_mean_637')]
markers_793 <- rmip_0.2$SNP[which(rmip_0.2$embedding=='embedding_std_793')]
markers_984 <- rmip_0.2$SNP[which(rmip_0.2$embedding=='embedding_mean_984')]
markers_highlight <- c('2:1961996:T:A', '9:62272161:C:A', '5:60007887:G:C', '5:60007887:G:C', '10:60309710:C:G')
highligh_embeddings <- c('embedding_std_128', 'embedding_std_128', 'embedding_std_793', 'embedding_mean_984', 'embedding_mean_637')

score_bars <- df_bar %>% 
  filter(marker %in% c('2:1961996:T:A', '5:60007887:G:C', '9:62272161:C:A', '10:60309710:C:G')) %>%
  mutate(marker = factor(marker, levels = c('2:1961996:T:A', '5:60007887:G:C', '9:62272161:C:A', '10:60309710:C:G'))) %>% 
  ggplot(aes(location, mean, fill = allele)) + 
  facet_wrap(vars(marker), nrow = 2) + 
  geom_col(position = position_dodge(width = 0.9)) + 
  geom_errorbar(aes(ymin  = mean - se, ymax = mean + se), position = position_dodge(width = 0.9), width = 0.25) + 
  # annotate(geom = 'text', 
  #          x = c('2:1961996:T:A', '5:60007887:G:C', '9:62272161:C:A', '10:60309710:C:G'), 
  #          y = rep(2.4, 4), 
  #          label = c('****', '****', '****', ''),
  #          size = 3.175) +
  scale_x_discrete(name = NULL, 
                   expand = c(0, 0), 
                   label = c('AL', 'GA')) + 
  scale_y_continuous(name = 'Human Ordinal Score', 
                     expand = c(0, 0)) +
  scale_fill_manual(name = NULL, 
                    values = paletteer_d('ggsci::orange_material', 10)[c(4, 6)], 
                    label = c('REF', 'ALT')) + 
  theme_use + 
  theme(axis.text.x = element_text(angle = 90))
score_bars

ggsave('figures/main/figure4/human_score_bars.svg', plot = score_bars, height = 2.75, width = 1.75, dpi = 1e3, bg = NULL)

embeddings_highlight <- c('embedding_std_128', 'embedding_mean_637', 'embedding_std_793', 'embedding_mean_984')
color_pals <- list(paletteer_d('ggsci::light_blue_material', 10)[c(4, 10)], paletteer_d('ggsci::light_green_material', 10)[c(5, 10)], 
                  paletteer_d('ggsci::red_material')[c(4, 10)], paletteer_d('rcartocolor::Purp', 7)[c(2, 6)])

for(i in c(28, 37, 42, 23, 39))
# for(i in dir_switch_sig)
{
  CHROM <- rmip_0.2$CHROM[i]
  m <- rmip_0.2$SNP[i]
  embedding <- rmip_0.2$embedding[i]
  feature <- str_split_i(embedding, '_', 3)
  stat <- str_split_i(embedding, '_', 2) %>% 
    str_to_title() %>% 
    str_replace('Std', 'SD')
  color_pal <- color_pals[[which(embeddings_highlight==embedding)]]
  # color_pal <- color_pals[[4]]
  mean <- str_c(embedding, '_mean')
  se <- str_c(embedding, '_se')
  
  model <- as.formula(str_c(embedding, '~ `', m, '`'))
  unl_test <- wilcox.test(model, data = blues_all, subset = location=='UNL', conf.int = TRUE)
  unl_sap_test <- wilcox.test(model, data = blues_all, subset = location=='UNL-SAP', conf.int = TRUE)
  aamu_test <- wilcox.test(model, data = blues_all, subset = location=='AAMU', conf.int = TRUE)
  fvsu_test <- wilcox.test(model, data = blues_all, subset = location=='FVSU', conf.int = TRUE)
  
  pvals <- c(unl_test$p.value, unl_sap_test$p.value, aamu_test$p.value, fvsu_test$p.value)
  asterisks <- rep('', 4)
  for(j in 1:4)
  {
    if(pvals[j] < 0.0001){asterisks[j] <- '****'} 
    else if(pvals[j] < 0.001){asterisks[j] <- '***'}
    else if(pvals[j] < 0.01){asterisks[j] <- '**'}
    else if(pvals[j] < 0.05){asterisks[j] <- '*'}
  }
  df <- blues_summary %>% 
    filter(marker==m) 
  asterisk_height <- max(df[[mean]], na.rm = TRUE)
  
  assign(glue('stability_chr{CHROM}_{feature}'), df %>% 
    ggplot(aes(location, .data[[mean]], fill = allele)) + 
      geom_col(position = position_dodge(width = 0.9)) + 
      geom_errorbar(aes(ymin = .data[[mean]] - .data[[se]],
                        ymax = .data[[mean]] + .data[[se]]),
                    position = position_dodge(width = 0.9),
                    width = 0.25) +
      annotate(geom = 'text', 
               x = c('UNL', 'UNL-SAP', 'AAMU', 'FVSU'),
               y = asterisk_height, 
               label = asterisks,
               size = 9, 
               size.unit = 'pt') + 
      scale_x_discrete(name = NULL, 
                       label = c('NE', 'NE-C', 'AL', 'GA'),
                       expand = c(0, 0)) + 
      scale_y_continuous(name = str_c(feature, ' (', stat, ')'), 
                         expand = c(0, 0)) +
      scale_fill_manual(name = str_c('Chr', CHROM, ':\n', str_split_i(m, ':', 2)),
                        values = color_pal, 
                        labels = str_split(m, ':')[3:4]) + 
    theme_use + 
    theme(axis.text.x = element_text(size = 9, color = 'black', margin = margin(0, 0, 0, 0), 
                                     vjust = 0.5, hjust = 0.5, angle = 90)))
  
  print(get(glue('stability_chr{CHROM}_{feature}')))
}

fig4top <- plot_grid(manhattan, score_bars, nrow = 1, rel_widths = c(1, 0.33), labels = 'auto')
fig4bottom <- plot_grid(get('stability_chr5_793'), get('stability_chr5_984'), get('stability_chr9_128'),
                        get('stability_chr2_128'), 
                         get('stability_chr10_637'), 
                        nrow = 1, labels = c('c', 'd', 'e', 'f', 'g'))
fig4 <- plot_grid(fig4top, fig4bottom, ncol = 1, labels = NULL, rel_heights = c(1, 0.75))
ggsave('figures/main/figure4/figure4.svg', plot = fig4, width = 6.5, height = 5, dpi = 1e3, bg = NULL)
  
gxe_bars <- plot_grid(get('stability_chr9_37'), get('stability_chr1_976'), get('stability_chr2_821'), 
                      labels = 'auto', nrow = 1)
ggsave('figures/supplemental/gxe_barplots.svg', plot = gxe_bars, dpi = 1e3, bg = NULL, 
       width = 6.5, height = 2.76)
