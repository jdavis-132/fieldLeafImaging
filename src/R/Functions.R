library(tidyverse)
devtools::source_url('https://github.com/jdavis-132/hips/raw/refs/heads/master/src/Functions.R')
source('../widiv-transcriptome/src/Functions.R')

printScreePlot <- function(prcomp_obj, nPCs = ncol(prcomp_obj$rotation))
{
  summary <- summary(prcomp_obj)
  var_explained <- summary$importance %>% 
    as_tibble(rownames = 'metric') %>%
  pivot_longer(!metric, names_to = 'PC', names_prefix = 'PC', values_to = 'value') %>%
  mutate(PC = str_remove(PC, 'V') %>% 
           as.numeric()) %>%
    pivot_wider(id_cols = c(PC), names_from = metric, values_from = value) %>%
    rename(sdev = `Standard deviation`,
           propVar = `Proportion of Variance`,
           cumulativePropVar = `Cumulative Proportion`)

  plot <- ggplot(var_explained, aes(PC, propVar*100)) +
    geom_line() +
    scale_x_continuous(name = 'PC', expand = c(0,0), limits = c(0, nPCs)) +
    scale_y_continuous(name = 'Variance Explained', labels = ~str_c(.x, '%'), expand = c(0, 0)) +
    theme_use
  print(plot)
  return(var_explained)
}

getPCScores <- function(data, cols, rank = 100)
{
  mat <- data %>% 
    select(cols) %>%
    as.matrix()
  
  metadata <- data %>% 
    select(!cols)
  
  pca <- prcomp(
    mat, retx = TRUE, scale = TRUE, rank. = rank)
  printScreePlot(pca)
  pc_scores <- bind_cols(metadata, pca$x)
  
  return(pc_scores)
}
