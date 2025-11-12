library(tidyverse)
devtools::source_url('https://github.com/jdavis-132/hips/raw/refs/heads/master/src/Functions.R')
source('../widiv-transcriptome/src/Functions.R')
embeddings_128 <- read_csv('src/autoencoder_no_weighting/embeddings/embeddings.csv')
lv_cols <- colnames(embeddings)[3:258]

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

sam_pcs <- getPCScores(embeddings_sam, )

