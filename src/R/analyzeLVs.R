library(tidyverse)
library(readxl)
source('src/R/Functions.R')

manual_scores <- read_excel('data/manual/DataRecord_2025_combined (1).xlsx', sheet = '828', skip = 1, 
                            col_names = c('plotNumber', 'rep', 'date', 'genotype', 
                                          'score_A', 'score_B', 'score_average', 'notes'))

embeddings_sam <- read_csv('src/sam2/sam2_image_embeddings.csv')
lv_cols <- colnames(embeddings_sam[2:ncol(embeddings_sam)])
# look at distributions
# plot histograms 
for(lv in lv_cols)
{
  printHistogram(embeddings_sam, lv, title = lv)
}

sam_pcs <- getPCScores(embeddings_sam, cols = contains('embedding'))
pcs <- colnames(sam_pcs)[2:21]
for(lv in pcs)
{
  printHistogram(sam_pcs, lv, title = lv)
}

sam_pcs_winsor <- sam_pcs %>% 
  select(image_path, all_of(pcs))
for(pc in pcs)
{
  sam_pcs_winsor <- winsorize(sam_pcs_winsor, pc, 0.01, 0.99)
}

for(lv in pcs)
{
  printHistogram(sam_pcs_winsor, lv, title = lv)
}

embeddings_dino <- read_csv('output/dinov2_features.csv')
embeddings_dino_metadata <- embeddings_dino %>% 
  mutate(plotNumber = str_split_i(image_path, fixed('/'), 6) %>% 
           str_split_i('_', 1) %>% 
           as.numeric()) %>% 
  left_join(manual_scores, join_by(plotNumber)) %>% 
  relocate(image_path, plotNumber, rep, date, genotype, score_A, score_B, score_average)
write_csv(embeddings_dino_metadata, 'output/dinov2_features_metadata.csv')

lv_cols <- colnames(embeddings_dino)[2:ncol(embeddings_dino)]
for(lv in lv_cols)
{
  printHistogram(embeddings_dino, lv, title = lv)
}

dino_pcs <- getPCScores(embeddings_dino, cols = contains('feature'))
pcs <- colnames(dino_pcs)[2:59]
for(lv in pcs)
{
  printHistogram(dino_pcs, lv, title = lv)
}

dino_pcs_winsor <- dino_pcs %>% 
  select(image_path, all_of(pcs))
for(pc in pcs)
{
  dino_pcs_winsor <- winsorize(dino_pcs_winsor, pc, 0.01, 0.99)
}

for(lv in pcs)
{
  printHistogram(dino_pcs_winsor, lv, title = lv)
}
