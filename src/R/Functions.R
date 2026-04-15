library(tidyverse)
devtools::source_url('https://github.com/jdavis-132/hips/raw/refs/heads/master/src/Functions.R')
source('../widiv-transcriptome/src/Functions.R')
source('https://github.com/jschnable/parallelgwas/raw/refs/heads/main/manhattanPlot.R')

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
    select({{ cols }}) %>%
    as.matrix() 
  print(str_c('input matrix columns: ', dim(mat)[2]))
  
  metadata <- data %>% 
    select(!{{ cols }})
  
  pca <- prcomp(
    mat, retx = TRUE, scale = TRUE, rank. = rank)
  printScreePlot(pca)
  pc_scores <- bind_cols(metadata, pca$x)
  
  return(pc_scores)
}

winsorize <- function(data, col, lower_prob, upper_prob)
{
  lower <- quantile(data[[col]], probs = c(lower_prob), na.rm = TRUE)
  upper <- quantile(data[[col]], probs = c(upper_prob), na.rm = TRUE)
  df <- data
  df[which(df[[col]] <= lower), col] <- lower
  df[which(df[[col]] >= upper), col] <- upper
  return(df)
}

partitionVarianceSpATS <- function(data, response, genotype, x, y, covariates = NULL)
{
  x_seg <- floor(length(unique(data[[x]])) / 2) + 1
  y_seg <- floor(length(unique(data[[y]])) / 2) + 1
  spatial_formula <- as.formula(paste('~SAP(', x, ', ', y, ', nseg = c(', x_seg, ', ', y_seg, '))'))
  
  model <- SpATS(response, genotype, genotype.as.random = TRUE, 
                 spatial = spatial_formula,
                 random = covariates, data = data)
  vc <- model$var.comp %>% 
    as_tibble(rownames = 'component') %>% 
    rename(variance = value) %>% 
    mutate(proportion_variance = variance / sum(variance, na.rm = TRUE), 
           response = response)
  return(vc)
}

getSpATSBLUEs <- function(df, response, genotype, x, y, covariates_fixed = NULL, covariates_random = NULL)
{
  x_seg <- floor(length(unique(df)) / 2) + 1
  y_seg <- floor(length(unique(df[[y]])) / 2) + 1
  spatial_formula <- as.formula(paste('~SAP(', x, ', ', y, ', nseg = c(', x_seg, ', ', y_seg, '))'))
  print(response)
  model <- SpATS(response, genotype, genotype.as.random = FALSE, 
                 spatial = spatial_formula,
                 fixed = covariates_fixed, 
                 random = covariates_random, data = df)
  blues <- model$coeff %>% 
    as_tibble(rownames = 'genotype') %>% 
    filter(!str_detect(genotype, x) & !str_detect(genotype, y) & !str_detect(genotype, 'Intercept')) %>% 
    add_row(genotype = sort(df[[genotype]])[1], value = 0) %>%
    rename('{response}' := value)
  return(blues)
}

convertPhenotypesToPLINK <- function(infile, phenotypes)
{
  print(infile)
  plink <- read.csv(infile) %>% 
    dplyr::mutate(FID = '0', 
                  IID = genotype) %>% 
    dplyr::select(FID, IID, any_of(phenotypes)) %>% 
    relocate(FID, IID)
  write_tsv(plink, str_replace(infile, '.csv', '.txt'), quote = 'needed')
}

getBLUEs <- function(df, response, genotype, x, y, covariates_fixed = NULL, covariates_random = NULL)
{
  print(response)
  formula <- str_c(response, ' ~ ', x, ' + ', y, ' + ', genotype)
  if(!is.null(covariates_fixed))
  {
    for(f in covariates_fixed)
    {
      formula <- str_c(formula, ' + ', f)
    }
  }  
  if(!is.null(covariates_random))
  {
    for(r in covariates_random)
    {
      formula <- str_c(formula, ' + (1|', f, ')')
    }
  }
  model <- lm(as.formula(formula), data = df)
  blues <- model$coefficients %>% 
    as_tibble(rownames = 'genotype') %>% 
    filter(!str_detect(genotype, x) & !str_detect(genotype, y) & !str_detect(genotype, 'Intercept')) %>% 
    mutate(genotype = str_remove(genotype, 'genotype')) %>% 
    add_row(genotype = sort(df[[genotype]])[1], value = 0) %>%
    rename('{response}' := value)
  return(blues)
}

getRFPredictability <- function(predictions_df, model_descriptor = NULL)
{
  spearman_r2 <- cor(predictions_df[['label']], predictions_df[['predicted']], use = 'complete.obs', method = 'spearman')^2
  
  plot <- ggplot(predictions_df, aes(label, predicted)) + 
    # geom_bin2d() + 
    geom_point() +
    annotate(geom = 'text', x = 10, y=35, label = str_c('R^2==', spearman_r2), parse = TRUE) +
    # scale_fill_viridis(direction = -1) +
    # guides(fill = guide_colorbar(barwidth = 10)) +
    labs(x = 'Observed Percent Diseased Area\n(ExG Threshold)', 
         y = 'Predicted Percent Disease Area\n(RF)', 
         title = model_descriptor) + 
    theme_use 
  print(plot)
  return(plot)
}

summariseSignals_PANICLE <- function(path)
{
  files <- Sys.glob(path)
  signals <- tibble()
  for(f in files)
  {
    df <- read_csv(f,
                   col_types = 'ccnccnnnc', 
                   col_names = c('SNP', 'CHROM', 'POS', 'REF', 'ALT', 'MAF', 'pval', 'effect', 'method'),
                   skip = 1) %>% 
      mutate(filename = f)
    signals <- bind_rows(signals, df)
  }
  return(signals)
}

getCandidateGenes <- function(annotations, outdir, chr, pos, window)
{
  window_start <- max(0, pos - window)
  window_end <- pos + window
  annotation_filtered <- annotation %>% 
    filter(CHROM==chr &
             (between(start, window_start, window_end) | between(end, window_start, window_end)))
  write_csv(annotation_filtered, str_c(outdir, 'chr', chr, '_', pos, '_candidates_', (window/1e3), 'kb.csv'))
  return(annotation_filtered)
}
