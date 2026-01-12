library(tidyverse)
source('src/R/Functions.R')
print(getwd())

# Usage: R CMD BATCH -phenotype_source -embeddings.csv -field_index.csv -LV_prefix -out_prefix -winsor_strength -vcf_genotypes_list.txt LV_quantGen.R
# Assumes plotNumber join column present in field_index
# And image_path column in embeddings where basenames begin with plotNumber_*
# phenotype_source should either be image or manual
# if image, will look for image_path in join but if manual, plotNumber
args <- commandArgs(trailingOnly = FALSE)
phenotype_source <- str_remove(args[length(args)-6], fixed('-'))
embeddings <- str_remove(args[length(args)-5], fixed('-'))
field_index <- str_remove(args[length(args)-4], fixed('-'))
LV_prefix <- str_remove(args[length(args)-3], fixed('-'))
out_prefix <- str_remove(args[length(args) - 2], fixed('-'))
winsor_strength <- as.numeric(str_remove(args[length(args) - 1], fixed('-')))
vcf_genotypes <- str_remove(args[length(args)], fixed('-'))

marker_genotypes <- read_tsv(vcf_genotypes, col_names = TRUE)
marker_genotypes <- colnames(marker_genotypes)[10:ncol(marker_genotypes)]
# join dataframes
df_embeddings <- read_csv(embeddings)
if(phenotype_source == 'image')
{
  df_embeddings <- df_embeddings %>% 
    dplyr::mutate(plotNumber = basename(image_path) %>% 
                    str_split_i('_', 1) %>% 
                    as.numeric()) %>% 
    select(c(image_path, plotNumber, contains(LV_prefix)))
  row_id <- 'image_path'
} else
{
  df_embeddings <- df_embeddings %>% 
    filter(across(contains(LV_prefix), ~!is.na(.x))) %>% 
    select(c(plotNumber, contains(LV_prefix)))
}


df_field_index <- read_csv(field_index) %>% 
  mutate(genotype = str_remove_all(genotype, ' ')) %>% 
  mutate(genotype = case_when(genotype == 'PI655991' ~ "BTx378", 
                              genotype == 'PI655990' ~ 'Comb7078', 
                              genotype == 'PI655977' ~ "RTAM2566", 
                              genotype == 'PI655978' ~ "RTX2737",
                              genotype == 'PI542718' ~ "SanChiSan",
                              genotype == 'PI656023' ~ "Segaolane",  
                              .default = genotype))

df_combined <- left_join(df_embeddings, df_field_index, join_by(plotNumber))

# winsorize to deal with extreme values
lv_cols <- colnames(df_combined)[str_detect(colnames(df_combined), LV_prefix)]

df_winsor <- df_combined
for(lv in lv_cols)
{
  df_winsor <- winsorize(df_winsor, lv, winsor_strength, 1 - winsor_strength)
}

if(phenotype_source=='image')
{
  # calculate PCs
  pcs <- getPCScores(df_winsor, (matches(LV_prefix) & where(~is.numeric(.x) && 
                                        var(.x, na.rm = TRUE) != 0)))
  
  # winsorize PCs
  df_winsorpc <- pcs %>%
    rename_with(~str_c(out_prefix, '_', .x), .cols=contains('PC'))
  pc_cols <- colnames(df_winsorpc)[str_detect(colnames(df_winsorpc), 'PC')]
  for(pc in pc_cols)
  {
    df_winsorpc <- winsorize(df_winsorpc, pc, winsor_strength, 1 - winsor_strength)
  }
  # add to df
  df_winsorpc <- select(df_winsorpc, c(image_path, all_of(pc_cols)))
  
  df <- left_join(df_winsor, df_winsorpc, join_by(image_path))
  write_csv(df, str_c('output/', out_prefix, '_rf_predictors.csv'))
  response_vars <- c(lv_cols, pc_cols)
}else
{
  df <- df_winsor
  response_vars <- lv_cols
}

# broad-sense variance partitioning 
vp <- tibble()
for(v in response_vars)
{
  vp <- bind_rows(vp, partitionVariance3(df, v, v, ' ~ (1|genotype) + (1|range) + (1|row)'))
}

write_csv(vp, str_c('output/', out_prefix, '_vp.csv'))

#BLUEs
blues <- getBLUEs(df, response_vars[1], 'genotype', 'range', 'row')

if(length(response_vars) > 1)
{
  for(v in response_vars[2:length(response_vars)])
  {
    blues <- full_join(blues, 
                       getBLUEs(df, v, 'genotype', 'range', 'row'), 
                       join_by(genotype))
  }
}
blues <- filter(blues, genotype %in% marker_genotypes)
write_csv(blues, str_c('output/', out_prefix, '_blues.csv'))
write.table(unique(blues$genotype), str_c('output/', out_prefix, '_genotypes_keep.txt'), 
            sep = '\t', quote = FALSE, col.names = FALSE, row.names = FALSE)

# Convert blues to plink for h2 in ldak
convertPhenotypesToPLINK(str_c('output/', out_prefix, '_blues.csv'), response_vars)

# split BLUEs dataframe for parallel GWAS
splitDataFrame(blues, response_vars, str_c('output/', out_prefix, '_blues_'), 25)

