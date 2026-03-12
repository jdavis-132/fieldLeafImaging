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
