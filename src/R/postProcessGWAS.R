library(tidyverse)
source('src/R/Functions.R')

trait_groups <- c('pctd_rs', 'pctd_rs_pred', 'sam3_rs_high_fi', 'scores_813', 'scores_828')

for(grp in trait_groups)
{
  summary <- read_csv(str_c('output/candidate_info/mlm_20260312/', grp, '/GWAS_summary_by_traits_methods.csv'))
  pmap_files <- Sys.glob(str_c('output/candidate_info/mlm_20260312/', grp, '/GWAS_*_all_results.csv'))
  
  for(f in pmap_files)
  {
    trait <- str_remove(f, str_c('output/candidate_info/mlm_20260312/', grp, '/GWAS_')) %>% 
      str_remove('_all_results.csv')
    
    if(str_detect(trait, 'PC')) {next}
    
    thresh <- summary$Threshold[summary$Trait==trait]
    
    pmap <- read_csv(f)
      
    manhattan <- plotManhattan(pmap, 
                               sig = MLM_P, 
                               multitrait = FALSE, 
                               resampling = FALSE, 
                               threshold = -log10(thresh), 
                               theme = theme_use, 
                               main = str_c(grp, trait, sep = ':'), 
                               species = 'sorghum')
    ggsave(str_c('output/candidate_info/mlm_20260312/', grp, '/', trait, '_manhattan.png'), 
           plot = manhattan, 
           width = 5, 
           height = 2.5, 
           dpi = 300, 
           bg = 'transparent')
    sig_markers <- filter(pmap, MLM_P < thresh)
    write_csv(sig_markers, str_c('output/candidate_info/mlm_20260312/', grp, '/', trait, '_significant_markers.csv'))
  }
}

all_sig_markers <- tibble()
sig_marker_files <- Sys.glob('output/candidate_info/mlm_20260312/*/*_significant_markers.csv')

for(f in sig_marker_files)
{
  df <- read_csv(f)
  if(nrow(df) > 0)
  {
    df <- df %>% 
      mutate(grp = str_split_i(f, fixed('/'), 4), 
             trait = str_split_i(f, fixed('/'), 5) %>% 
               str_remove('_significant_markers.csv'))
    all_sig_markers <- bind_rows(all_sig_markers, df)
  }
}

write_csv(all_sig_markers, 'output/candidate_info/mlm_20260312/all_significant_markers.csv')
plotManhattan(all_sig_markers, sig = MLM_P, multitrait = FALSE, resampling = FALSE, species = 'sorghum')
window_size <- 1e6
peaks <- read_csv('output/candidate_info/mlm_20260312/peaks_all_traits.csv') %>% 
  rowwise() %>% 
  mutate(window_start = case_when(pLength >= window_size ~ pStart, 
                                  .default = max(0, POS - (window_size/2))), 
         window_end = case_when(pLength >= window_size ~ pStop, 
                                .default = POS + (window_size / 2)))
peaks <- peaks %>% 
  ungroup() %>% 
  add_column(traits = c('predicted', 'pctd', 'pctd', 'pctd', 'pctd,predicted', "embedding_mean_108", 
                        'pctd,predicted,embedding_std_251', 'pctd', 'scores_813', 'embedding_mean_986', 
                        'embedding_mean_181', 'embedding_mean_181,embedding_mean_698', 
                        'embedding_mean_210,embedding_mean_619,embedding_mean_698', 
                        'embedding_mean_619,embedding_mean_698', 'embedding_std_793', 'embedding_mean_174', 
                        'scores_813', 'embedding_mean_308', 'embedding_std_251', 'embedding_mean_586'))
write_tsv(peaks, 'output/candidate_info/mlm_20260312/peaks_all_traits.tsv')

peaks <- read_tsv('output/candidate_info/mlm_20260312/peaks_all_traits.tsv')
annotation <- read_csv('data/sorghum_v5.1_annotation_combined.csv')

for(p in 1:nrow(peaks))
{
  getCandidateGenes(annotations = annnotation, 
                    outdir = 'output/candidate_info/mlm_20260312/candidate_genes/', 
                    chr = peaks$CHROM[p], 
                    pos = peaks$POS[p], 
                    window = 1e5)
}

scores_813_log10_6 <- read_csv('output/candidate_info/mlm_20260312/scores_813/GWAS_score_average_all_results.csv') %>% 
  filter(MLM_P < 1e-6)
write_csv(scores_813_log10_6, 'output/candidate_info/mlm_20260312/scores_813_plt1eneg6.csv')

scores_828_log10_6 <- read_csv('output/candidate_info/mlm_20260312/scores_828/GWAS_score_average_all_results.csv') %>%
  filter(MLM_P < 1e-6)
write_csv(scores_828_log10_6, 'output/candidate_info/mlm_20260312/scores_828_plt1eneg6.csv')
