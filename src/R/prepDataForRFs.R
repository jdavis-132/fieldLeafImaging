library(tidyverse)
source('src/R/Functions.R')
# pctd_crops_unl <- tibble()
# for(i in 1:8)
# {
#   path <- str_c('data/processed/ne2025/device', i, '/pctd_all.csv')
#   pctd_crops_device <- read_csv(path)
#   pctd_crops_unl <- bind_rows(pctd_crops_unl, pctd_crops_device)
# }
# 
# pctd_crops_unl <- pctd_crops_unl %>%
#   select(image_name, ExG_P20_below_threshold_pct) %>%
#   rename(pctd = ExG_P20_below_threshold_pct) %>%
#   mutate(image_id = str_remove(image_name, '-05_00_[0-9]\\.(png|npz)') %>%
#            str_remove('-05_00\\.jpg') %>% 
#            str_remove('-05_00'))
# write_csv(pctd_crops_unl, 'data/processed/ne2025/pctd_crops_all.csv')
idx_unl <- read_csv('data/ne2025/SbDiv_ne2025_fieldindex.csv')
pctd_crops_unl <- read_csv('data/processed/ne2025/pctd_crops_all.csv') %>% 
  mutate(plotNumber = str_split_i(image_name, '_', 1) %>% 
           as.numeric()) %>%
  left_join(idx_unl, join_by(plotNumber))
images_keep_unl <- read_csv('data/ne2025/images_keep_all.csv', col_names = c('image_id'))
pctd_crops_unl_filtered <- filter(pctd_crops_unl, image_id %in% images_keep_unl$image_id)
write_csv(pctd_crops_unl_filtered, 'data/ne2025/pctd_filtered.csv')

hist <- ggplot(pctd_crops_unl_filtered, aes(pctd)) + 
  geom_histogram()
hist

gte25 <- filter(pctd_crops_unl_filtered, pctd >= 25)
write_csv(gte25, 'data/ne2025/gte25pctd/crops_gte25pctd.csv')
senesced_crops <- c('2724_LeafPhotoA_2025-09-09%2010_46_37.811-05_00_0.png', '2724_LeafPhotoA_2025-09-09%2010_46_14.817-05_00_0.png', 
                    '2723_LeafPhotoA_2025-09-09%2010_44_59.309-05_00_2.png', '2723_LeafPhotoA_2025-09-09%2010_44_59.309-05_00_3.png', 
                    '2723_LeafPhotoA_2025-09-09%2010_44_59.309-05_00_1.png', '2723_LeafPhotoA_2025-09-09%2010_44_59.309-05_00_0.png', 
                    '2722_LeafPhotoA_2025-09-09%2010_43_40.267-05_00_0.png', '2722_LeafPhotoA_2025-09-09%2010_43_15.446-05_00_0.png', 
                    '2721_LeafPhotoA_2025-09-09%2010_42_26.993-05_00_2.png', '2721_LeafPhotoA_2025-09-09%2010_42_26.993-05_00_1.png', 
                    '2721_LeafPhotoA_2025-09-09%2010_42_26.993-05_00_0.png', '2714_LeafPhotoA_2025-09-09%2010_33_32.644-05_00_2.png', 
                    '2714_LeafPhotoA_2025-09-09%2010_33_32.644-05_00_1.png', '2714_LeafPhotoA_2025-09-09%2010_33_32.644-05_00_0.png', 
                    '2714_LeafPhotoA_2025-09-09%2010_33_10.614-05_00_1.png', '2714_LeafPhotoA_2025-09-09%2010_33_10.614-05_00_0.png', 
                    '2703_LeafPhotoA_2025-09-09%2010_15_56.392-05_00_3.png', '2703_LeafPhotoA_2025-09-09%2010_15_56.392-05_00_2.png',
                    '2703_LeafPhotoA_2025-09-09%2010_15_56.392-05_00_1.png', '2703_LeafPhotoA_2025-09-09%2010_15_56.392-05_00_0.png', 
                    '2670_LeafPhotoA_2025-09-09%2011_44_17.139-05_00_1.png', '2670_LeafPhotoA_2025-09-09%2011_44_17.139-05_00_3.png', 
                    '2670_LeafPhotoA_2025-09-09%2011_44_17.139-05_00_2.png', '2670_LeafPhotoA_2025-09-09%2011_44_17.139-05_00_0.png',
                    '2281_LeafPhotoA_2025-09-08%2016_59_13.027-05_00_3.png', '2281_LeafPhotoA_2025-09-08%2016_59_13.027-05_00_2.png', 
                    '2281_LeafPhotoA_2025-09-08%2016_59_13.027-05_00_1.png', '2281_LeafPhotoA_2025-09-08%2016_59_13.027-05_00_0.png', 
                    '1904_LeafPhotoA_2025-09-09%2014_39_09.961-05_00_2.png', '1904_LeafPhotoA_2025-09-09%2014_39_09.961-05_00_1.png', 
                    '1904_LeafPhotoA_2025-09-09%2014_39_09.961-05_00_0.png', '1896_LeafPhotoA_2025-09-09%2012_57_03.176-05_00_3.png', 
                    '1896_LeafPhotoA_2025-09-09%2012_57_03.176-05_00_2.png', '1896_LeafPhotoA_2025-09-09%2012_57_03.176-05_00_1.png', 
                    '1896_LeafPhotoA_2025-09-09%2012_57_03.176-05_00_0.png', '1738_LeafPhotoA_2025-09-09%2009_51_15.610-05_00_3.png', 
                    '1738_LeafPhotoA_2025-09-09%2009_51_15.610-05_00_2.png', '1738_LeafPhotoA_2025-09-09%2009_51_15.610-05_00_1.png', 
                    '1738_LeafPhotoA_2025-09-09%2009_51_15.610-05_00_0.png', '1735_LeafPhotoA_2025-09-09%2009_46_06.290-05_00_1.png', 
                    '1735_LeafPhotoA_2025-09-09%2009_46_06.290-05_00_0.png', '1707_LeafPhotoA_2025-09-09%2009_00_57.756-05_00_0.png', 
                    '2974_LeafPhotoA_2025-09-09%2016_22_34.046-05_00_2.png', '2974_LeafPhotoA_2025-09-09%2016_22_34.046-05_00_1.png',
                    '2974_LeafPhotoA_2025-09-09%2016_22_34.046-05_00_0.png', '2749_LeafPhotoA_2025-09-09%2011_20_42.639-05_00_2.png', 
                    '2749_LeafPhotoA_2025-09-09%2011_20_42.639-05_00_1.png', '2749_LeafPhotoA_2025-09-09%2011_20_42.639-05_00_0.png', 
                    '2740_LeafPhotoA_2025-09-09%2011_08_17.094-05_00_3.png', '2731_LeafPhotoA_2025-09-09%2010_56_46.860-05_00_1.png') %>% 
  str_replace('%20', ' ')

broken_crops <- c('2722_LeafPhotoA_2025-09-09%2010_43_15.446-05_00_2.png', '2722_LeafPhotoA_2025-09-09%2010_43_15.446-05_00_1.png', 
                  '2719_LeafPhotoA_2025-09-09%2010_39_28.624-05_00_1.png', '2719_LeafPhotoA_2025-09-09%2010_39_28.624-05_00_0.png', 
                  '2717_LeafPhotoA_2025-09-09%2010_36_42.002-05_00_3.png', '2717_LeafPhotoA_2025-09-09%2010_36_42.002-05_00_0.png', 
                  '2717_LeafPhotoA_2025-09-09%2010_36_42.002-05_00_1.png', '2717_LeafPhotoA_2025-09-09%2010_36_27.604-05_00_2.png', 
                  '2717_LeafPhotoA_2025-09-09%2010_36_27.604-05_00_1.png', '2717_LeafPhotoA_2025-09-09%2010_36_27.604-05_00_0.png', 
                  '2714_LeafPhotoA_2025-09-09%2010_33_10.614-05_00_2.png', '2712_LeafPhotoA_2025-09-09%2010_30_21.843-05_00_3.png', 
                  '2712_LeafPhotoA_2025-09-09%2010_30_21.843-05_00_2.png', '2709_LeafPhotoA_2025-09-09%2010_26_09.764-05_00_3.png', 
                  '2709_LeafPhotoA_2025-09-09%2010_26_09.764-05_00_2.png', '2709_LeafPhotoA_2025-09-09%2010_26_09.764-05_00_1.png', 
                  '2678_LeafPhotoA_2025-09-09%2011_04_41.551-05_00_2.png', '2678_LeafPhotoA_2025-09-09%2011_04_41.551-05_00_1.png', 
                  '2671_LeafPhotoA_2025-09-09%2011_45_24.152-05_00_2.png', '2671_LeafPhotoA_2025-09-09%2011_45_24.152-05_00_1.png', 
                  '2670_LeafPhotoA_2025-09-09%2011_44_02.746-05_00_3.png', '2670_LeafPhotoA_2025-09-09%2011_44_02.746-05_00_2.png', 
                  '2670_LeafPhotoA_2025-09-09%2011_44_02.746-05_00_1.png', '2668_LeafPhotoA_2025-09-09%2011_41_41.343-05_00_2.png', 
                  '2668_LeafPhotoA_2025-09-09%2011_41_41.343-05_00_1.png', '2660_LeafPhotoA_2025-09-09%2011_32_52.703-05_00_3.png', 
                  '2660_LeafPhotoA_2025-09-09%2011_32_52.703-05_00_2.png', '2514_LeafPhotoA_2025-09-09 10_11_35.245-05_00_0.png', 
                  '2179_LeafPhotoA_2025-09-09%2012_23_49.478-05_00_1.png', '2179_LeafPhotoA_2025-09-09%2012_23_49.478-05_00_0.png', 
                  '2170_LeafPhotoA_2025-09-09%2012_08_21.167-05_00_0.png', '2169_LeafPhotoA_2025-09-09%2012_05_55.711-05_00_3.png', 
                  '2169_LeafPhotoA_2025-09-09%2012_05_55.711-05_00_0.png', '2168_LeafPhotoA_2025-09-09%2012_03_54.274-05_00_2.png', 
                  '2168_LeafPhotoA_2025-09-09%2012_03_54.274-05_00_1.png', '2168_LeafPhotoA_2025-09-09%2012_03_54.274-05_00_0.png', 
                  '2074_LeafPhotoA_2025-09-08%2012_09_18.987-05_00_0.png', '2051_LeafPhotoA_2025-09-08%2011_43_37.507-05_00_0.png',
                  '1896_LeafPhotoA_2025-09-09%2012_57_23.415-05_00_3.png', '1747_LeafPhotoA_2025-09-09%2010_05_14.099-05_00_2.png',
                  '1744_LeafPhotoA_2025-09-09%2010_01_05.958-05_00_0.png', '1743_LeafPhotoA_2025-09-09%2009_59_20.258-05_00_3.png', 
                  '1743_LeafPhotoA_2025-09-09%2009_59_20.258-05_00_2.png', '1743_LeafPhotoA_2025-09-09%2009_59_20.258-05_00_1.png', 
                  '1743_LeafPhotoA_2025-09-09%2009_59_20.258-05_00_0.png', '1742_LeafPhotoA_2025-09-09%2009_58_06.549-05_00_1.png', 
                  '1742_LeafPhotoA_2025-09-09%2009_58_06.549-05_00_0.png', '1740_LeafPhotoA_2025-09-09%2009_55_10.339-05_00_2.png', 
                  '1740_LeafPhotoA_2025-09-09%2009_55_10.339-05_00_1.png', '1740_LeafPhotoA_2025-09-09%2009_55_10.339-05_00_0.png', 
                  '1739_LeafPhotoA_2025-09-09%2009_53_10.390-05_00_3.png', '1739_LeafPhotoA_2025-09-09%2009_53_10.390-05_00_2.png', 
                  '1734_LeafPhotoA_2025-09-09%2009_44_18.298-05_00_2.png', '1734_LeafPhotoA_2025-09-09%2009_44_18.298-05_00_1.png',  
                  '1734_LeafPhotoA_2025-09-09%2009_44_18.298-05_00_0.png', '1733_LeafPhotoA_2025-09-09%2009_43_21.749-05_00_2.png', 
                  '1733_LeafPhotoA_2025-09-09%2009_43_21.749-05_00_1.png', '1733_LeafPhotoA_2025-09-09%2009_43_21.749-05_00_0.png', 
                  '1733_LeafPhotoA_2025-09-09%2009_42_56.742-05_00_0.png', '1728_LeafPhotoA_2025-09-09%2009_30_58.151-05_00_3.png', 
                  '1728_LeafPhotoA_2025-09-09%2009_30_58.151-05_00_2.png', '1719_LeafPhotoA_2025-09-09%2009_17_27.215-05_00_3.png', 
                  '1719_LeafPhotoA_2025-09-09%2009_17_27.215-05_00_2.png', '1712_LeafPhotoA_2025-09-09%2009_07_36.217-05_00_1.png', 
                  '1712_LeafPhotoA_2025-09-09%2009_07_36.217-05_00_0.png', '1711_LeafPhotoA_2025-09-09%2009_06_57.472-05_00_2.png', 
                  '1711_LeafPhotoA_2025-09-09%2009_06_57.472-05_00_1.png', '6043_LeafPhotoA_2025-09-09%2016_04_58.355-05_00_0.png',
                  '6020_LeafPhotoA_2025-09-09%2016_43_52.229-05_00_0.png', '2974_LeafPhotoA_2025-09-09%2016_22_34.046-05_00_3.png', 
                  '2914_LeafPhotoA_2025-09-09%2016_33_56.135-05_00_3.png', '2749_LeafPhotoA_2025-09-09%2011_20_58.643-05_00_1.png', 
                  '2749_LeafPhotoA_2025-09-09%2011_20_58.643-05_00_0.png', '2748_LeafPhotoA_2025-09-09%2011_19_58.528-05_00_2.png', 
                  '2740_LeafPhotoA_2025-09-09%2011_08_33.741-05_00_2.png', '2740_LeafPhotoA_2025-09-09%2011_08_33.741-05_00_1.png', 
                  '2740_LeafPhotoA_2025-09-09%2011_08_17.094-05_00_2.png', '2740_LeafPhotoA_2025-09-09%2011_08_17.094-05_00_1.png', 
                  '2740_LeafPhotoA_2025-09-09%2011_08_17.094-05_00_0.png', '2738_LeafPhotoA_2025-09-09%2011_05_39.627-05_00_3.png', 
                  '2738_LeafPhotoA_2025-09-09%2011_05_39.627-05_00_2.png', '2738_LeafPhotoA_2025-09-09%2011_05_39.627-05_00_1.png', 
                  '2737_LeafPhotoA_2025-09-09%2011_04_37.350-05_00_2.png', '2737_LeafPhotoA_2025-09-09%2011_04_37.350-05_00_1.png', 
                  '2737_LeafPhotoA_2025-09-09%2011_04_37.350-05_00_0.png', '2731_LeafPhotoA_2025-09-09%2010_56_46.860-05_00_0.png',
                  '2729_LeafPhotoA_2025-09-09%2010_55_13.189-05_00_1.png', '2729_LeafPhotoA_2025-09-09%2010_55_13.189-05_00_0.png') %>% 
  str_replace('%20', ' ')

crops_remove <- union(broken_crops, senesced_crops)
write_csv(tibble(image_name = crops_remove), 'data/ne2025/crops_remove_broken_or_senesced.csv')

crops_keep <- setdiff(pctd_crops_unl_filtered$image_name, crops_remove)
write_csv(tibble(image_name = crops_keep), 'data/ne2025/image_crops_keep.csv')

pctd_unl_rs <- filter(pctd_crops_unl_filtered, image_name %in% crops_keep)

human_scores <- read_csv('data/manual/all_image_scores.csv')  %>%
  select(image_id, username, score, genotype) %>%
  pivot_wider(id_cols = c(image_id, genotype),
              names_from = username, 
              values_from = score) %>%
  mutate(mean_score = (Libia + Ruben)/2) %>% 
  select(image_id, mean_score, genotype)

ae1_embeddings <- read_csv('models/autoencoder_20260107_143328_standard_lr0.001_bs32_l1/embeddings.csv')
ae2_embeddings <- read_csv('models/autoencoder_20260107_214228_standard_lr0.001_bs32_l1/embeddings.csv')
ae3_embeddings <- read_csv('models/autoencoder_20260108_012555_standard_lr0.001_bs32_l1_attention/embeddings.csv')
ae4_embeddings <- read_csv('models/autoencoder_20260108_183222_standard_lr0.001_bs32_l1_attention/embeddings.csv')
ae5_embeddings <- read_csv('models/autoencoder_20260109_034107_standard_lr0.001_bs32_l1_attention/embeddings.csv')
ae6_embeddings <- read_csv('models/autoencoder_20260109_205057_standard_lr0.001_bs32_l1/embeddings.csv')
ae7_embeddings <- read_csv('models/autoencoder_20260110_045517_standard_lr0.001_bs32_disease_weighted_l1/embeddings.csv')
dinov2_embeddings <- read_csv('output/dinov2_features.csv')
sam3_embeddings <- read_csv('output/sam3_embeddings.csv')
sam3_mean <- sam3_embeddings %>% 
  select(c(image_path, starts_with('embedding_mean')))
sam3_std <- sam3_embeddings %>% 
  select(c(image_path, starts_with('embedding_std')))

embedding_sets <- list(ae1_embeddings, ae2_embeddings, ae3_embeddings, ae4_embeddings, ae5_embeddings, ae6_embeddings, ae7_embeddings, 
                       dinov2_embeddings, sam3_embeddings, sam3_mean, sam3_std)
LV_prefix <- c(rep('latent_dim', 7), 'feature', rep('embedding', 3))
models <- c('ae1', 'ae2', 'ae3', 'ae4', 'ae5', 'ae6', 'ae7', 'dinov2', 'sam3', 'sam3_mean', 'sam3_std')
winsor_strength <- 0.01

for(i in 1:length(embedding_sets))
{
  lv_cols <- colnames(embedding_sets[[i]])[str_detect(colnames(embedding_sets[[i]]), LV_prefix[i])]
  # clean up image path for matching
  df_embeddings <- embedding_sets[[i]] %>%
    mutate(image_id = basename(image_path) %>% 
             str_remove('-05_00_[0-9]\\.(png|npz)') %>%
             str_remove('-05_00\\.jpg') %>% 
             str_remove('-05_00'), 
           image_name = basename(image_path) %>% 
             str_replace('.npz', '.png')) %>% 
    select(c(image_name, image_id, all_of(lv_cols)))
  
  # winsorize
  df_winsor <- df_embeddings
  for(lv in lv_cols)
  {
    df_winsor <- winsorize(df_winsor, lv, winsor_strength, 1 - winsor_strength)
  }
  
  pcs <- getPCScores(df_winsor, (matches(LV_prefix) & where(~is.numeric(.x) &&
                                          isTRUE(var(.x, na.rm = TRUE) != 0))))

  # winsorize PCs
  df_winsorpc <- pcs %>%
    rename_with(~str_c(models[i], '_', .x), .cols=contains('PC'))
    pc_cols <- colnames(df_winsorpc)[str_detect(colnames(df_winsorpc), 'PC')]
  for(pc in pc_cols)
  {
    df_winsorpc <- winsorize(df_winsorpc, pc, winsor_strength, 1 - winsor_strength)
  }
  # add to df
  df_winsorpc <- select(df_winsorpc, c(image_name, image_id, all_of(pc_cols)))

  df <- left_join(df_winsor, df_winsorpc, join_by(image_name, image_id))

  df_scores <- human_scores %>%
    left_join(df, join_by(image_id), relationship = 'one-to-many')

  write_csv(df_scores, str_c('output/', models[i], '_human_scores_rf.csv'))

  df_pctd <- pctd_crops_unl_filtered %>%
    left_join(df, join_by(image_name, image_id), relationship = 'one-to-one')
  write_csv(df_pctd, str_c('output/', models[i], '_pctd_crops_rf.csv'))

  df_pctd_rs <- pctd_unl_rs %>%
    left_join(df, join_by(image_name, image_id), relationship = 'one-to-many')
  write_csv(df_pctd_rs, str_c('output/', models[i], '_pctd_crops_rs_rf.csv'))
}



