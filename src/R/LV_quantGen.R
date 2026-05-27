library(tidyverse)
source('src/R/Functions.R')
print(getwd())

# Usage: R CMD BATCH -phenotype_source -embeddings.csv -field_index.csv -LV_prefix -out_prefix -winsor_strength -vcf_genotypes_list.txt LV_quantGen.R
# Assumes plotNumber join column present in field_index
# And image_path column in embeddings where basenames begin with plotNumber_*
# phenotype_source should either be image or manual
# if image, will look for image_path in join but if manual, plotNumber
args <- commandArgs(trailingOnly = FALSE)
phenotype_source <- str_remove(args[length(args) - 7], fixed('-'))
embeddings <- str_remove(args[length(args) - 6], fixed('-'))
field_index <- str_remove(args[length(args) - 5], fixed('-'))
LV_prefix <- str_remove(args[length(args) - 4], fixed('-')) # pipe sep list of prefixes, used as regex
out_prefix <- str_remove(args[length(args) - 3], fixed('-'))
winsor_strength <- as.numeric(str_remove(args[length(args) - 2], fixed('-')))
genotype_alignment <- str_remove(args[length(args) - 1], fixed('-'))
images_keep <- str_remove(args[length(args)], fixed('-')) # file with list of images to keep, excluding end of basename beginning from '-05_00_[tag].png, one per line

genotype_alignment <- read_tsv(genotype_alignment, col_names = c('genotype_idx', 'genotype_markers'), skip = 0) %>%
  mutate(genotype_idx = str_remove_all(genotype_idx, ' ')) %>%
  distinct()

# join dataframes
df_embeddings <- read_csv(embeddings)
if(phenotype_source == 'image')
{
  images_keep_list <- read_tsv(images_keep, col_names = c('image_id'), skip = 1)
  df_embeddings <- df_embeddings %>% 
    filter(!str_detect(image_path, 'cropped_transparent_bg')) %>% 
    mutate(plotNumber = basename(image_path) %>% 
                    str_split_i('_', 1) %>% 
                    as.numeric(), 
           image_id = basename(image_path) %>%
             str_remove('-05_00_[0-9]\\.(png|npz)') %>%
             str_remove('-05_00\\.jpg') %>% 
             str_remove('-05_00')) %>% 
    filter(image_id %in% images_keep_list$image_id) %>%
    select(c(image_path, plotNumber, matches(paste(LV_prefix, collapse = "|"))))
  row_id <- 'image_path'
} else
{
  df_embeddings <- df_embeddings %>% 
    filter(across(matches(paste(LV_prefix, collapse = "|")), ~!is.na(.x))) %>% 
    select(c(plotNumber, matches(paste(LV_prefix, collapse = "|"))))
}


df_field_index <- read_csv(field_index) %>% 
  mutate(genotype = str_remove_all(genotype, ' '))
  # mutate(genotype = case_when(genotype == 'PI655991' ~ "BTx378", 
  #                             genotype == 'PI655990' ~ 'Comb7078', 
  #                             genotype == 'PI655977' ~ "RTAM2566", 
  #                             genotype == 'PI655978' ~ "RTX2737",
  #                             genotype == 'PI542718' ~ "SanChiSan",
  #                             genotype == 'PI656023' ~ "Segaolane",  
  #                             .default = genotype))

df_combined <- left_join(df_embeddings, df_field_index, join_by(plotNumber))

# winsorize to deal with extreme values
lv_cols <- c()
for(p in unlist(LV_prefix))
{
  lv_cols <- c(lv_cols, colnames(df_combined)[str_detect(colnames(df_combined), p)])
}
# lv_cols <- c("embedding_std_976", "embedding_mean_560", "embedding_mean_174", "embedding_mean_939", "embedding_std_251", "embedding_std_466",
#              "embedding_mean_875", "embedding_std_793", "embedding_mean_191", "embedding_mean_283", "embedding_mean_108", "embedding_mean_768",
#              "embedding_mean_698", "embedding_mean_344", "embedding_mean_119", "embedding_std_244", "embedding_mean_615", "embedding_std_566",
#              "embedding_mean_586", "embedding_mean_122", "embedding_mean_210", "embedding_mean_619", "embedding_std_161", "embedding_mean_308",
             # "embedding_mean_165", "embedding_mean_986", "embedding_mean_197", "embedding_std_617", "embedding_std_783", "embedding_mean_181")
# lv_cols <- c("embedding_std_930", "embedding_std_552", "embedding_std_918", "embedding_mean_637", "embedding_std_976", "embedding_mean_886",
#             "embedding_mean_210", "embedding_std_383", "embedding_std_687", "embedding_mean_984", "embedding_mean_68", "embedding_mean_836",
#             "embedding_mean_586", "embedding_std_968", "embedding_mean_546", "embedding_std_821", "embedding_mean_656", "embedding_mean_968",
#             "embedding_mean_698", "embedding_mean_165", "embedding_mean_37", "embedding_mean_582", "embedding_mean_214", "embedding_mean_792",
#             "embedding_std_606", "embedding_mean_930", "embedding_mean_734", "embedding_mean_108", "embedding_mean_560", "embedding_std_166",
#             "embedding_std_983", "embedding_std_981", "embedding_mean_197", "embedding_std_817", "embedding_std_132", "embedding_std_82",
#             "embedding_mean_950", "embedding_mean_119", "embedding_std_128", "embedding_mean_139", "embedding_std_76", "embedding_std_839",
#             "embedding_std_326", "embedding_mean_842", "embedding_std_567", "embedding_mean_237", "embedding_std_793")

df_winsor <- df_combined
for(lv in lv_cols)
{
  df_winsor <- winsorize(df_winsor, lv, winsor_strength, 1 - winsor_strength)
}

if(phenotype_source=='image')
{
  # calculate PCs
  pcs <- getPCScores(df_winsor, (matches(paste(LV_prefix, collapse = "|")) & where(~is.numeric(.x) &&
                                        isTRUE(var(.x, na.rm = TRUE) != 0))))

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
# vp <- tibble()
# for(v in response_vars)
# {
#   vp <- bind_rows(vp, partitionVariance3(df, v, v, ' ~ (1|genotype) + (1|range) + (1|row)'))
# }
# 
# write_csv(vp, str_c('output/', out_prefix, '_vp.csv'))

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
blues <- blues %>% 
  left_join(genotype_alignment, join_by(genotype==genotype_idx)) %>%
  mutate(genotype_markers = case_when(is.na(genotype_markers) ~ genotype,
	   .default = genotype_markers)) %>% 
  filter(!is.na(genotype_markers)) %>%
  select(!c(genotype)) %>% 
  rename(genotype = genotype_markers)

blues_winsor <- blues
for(v in response_vars)
{
  blues_winsor <- winsorize(blues_winsor, v, winsor_strength, 1 - winsor_strength)
}
write_csv(blues_winsor, str_c('output/', out_prefix, '_blues.csv'))
write.table(unique(blues$genotype), str_c('output/', out_prefix, '_genotypes_keep.txt'), 
            sep = '\t', quote = FALSE, col.names = FALSE, row.names = FALSE)

# Convert blues to plink for h2 in ldak
# convertPhenotypesToPLINK(str_c('output/', out_prefix, '_blues.csv'), response_vars)

# split BLUEs dataframe for parallel GWAS
# splitDataFrame(blues, response_vars, str_c('output/', out_prefix, '_blues_'), 25)

