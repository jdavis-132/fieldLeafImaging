library(tidyverse)
source('src/R/Functions.R')
embeddings_128 <- read_csv('src/autoencoder_no_weighting/embeddings/embeddings.csv')
lv_cols <- colnames(embeddings)[3:258]

embeddings_vae <- read

# plot histograms 
for(lv in lv_cols)
{
  printHistogram(embeddings_128, lv, title = lv)
}

winsorize <- function(data, col, lower_prob, upper_prob)
{
  lower <- quantile(data[[col]], probs = c(lower_prob))
  upper <- quantile(data[[col]], probs = c(upper_prob))
  df <- data
  df[df[[col]] <= lower, col] <- lower
  df[df[[col]] >= upper, col] <- upper
  return(df)
}

embeddings_winsor <- embeddings_128
for(lv in lv_cols)
{
  embeddings_winsor <- winsorize(embeddings_winsor, lv, 0.05, 0.95)
}

# plot histograms 
for(lv in lv_cols)
{
  printHistogram(embeddings_winsor, lv, title = lv)
}

embeddings_sam <- read_csv('src/sam2/sam2_image_embeddings.csv')
lv_cols <- colnames(embeddings_sam[2:ncol(embeddings_sam)])
# look at distributions
# plot histograms 
for(lv in lv_cols)
{
  printHistogram(embeddings_sam, lv, title = lv)
}

sam_pcs <- getPCScores(embeddings_sam, cols = contains('embedding'))
pcs <- colnames(sam_pcs)[2:20]
for(lv in pcs)
{
  printHistogram(sam_pcs, lv, title = lv)
}

embeddings_dino <- read_csv('dinov2_features.csv')
lv_cols <- colnames(embeddings_dino)[2:ncol(embeddings_dino)]
for(lv in lv_cols)
{
  printHistogram(embeddings_dino, lv, title = lv)
}
