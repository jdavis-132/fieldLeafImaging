library(tidyverse)
source('src/R/Functions.R')

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

embeddings_dino <- read_csv('dinov2_features.csv')
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
