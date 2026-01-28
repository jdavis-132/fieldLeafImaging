library(tidyverse)

pctd_final <- read_csv('data/final_disease_scores.csv')
pctd_ne <- pctd_final %>% 
  filter(str_detect(image_path, '2025-09-0')) %>% 
  rename(pctd = ExG_P20_disease_pct) %>% 
  mutate(plotNumber = str_split_i(image_path, '_', 1) %>% 
           as.numeric())

ordinal_scores_828 <- read_csv('data/manual/scores_828.csv')

combined <- full_join(ordinal_scores_828, pctd_ne, join_by(plotNumber))

corplot <- ggplot(combined, aes(score_average, pctd)) + 
  geom_point() + 
  labs(x = 'Average Ordinal Score (8/28)', 
       y = 'PCT Disease (VI Threshold)')
corplot

combined_gte_50 <- filter(combined, pctd >= 50)

senesced <- c('1286_LeafPhotoA_2025-09-08 16_01_17.344-05_00_leaf.png', '1286_LeafPhotoA_2025-09-08 16_01_45.976-05_00_leaf.png', 
              '1316_LeafPhotoA_2025-09-08 09_34_38.201-05_00_leaf.png', '1622_LeafPhotoA_2025-09-09 09_56_01.433-05_00_leaf.png',
              '2281_LeafPhotoA_2025-09-08 16_59_13.027-05_00_leaf.png', '2721_LeafPhotoA_2025-09-09 10_42_26.993-05_00_leaf.png', 
              '2722_LeafPhotoA_2025-09-09 10_43_15.446-05_00_leaf.png', '2740_LeafPhotoA_2025-09-09 11_08_17.094-05_00_leaf.png')

high_score_low_pctd <- combined %>% 
  filter(score_average > 4.5 & pctd < 5) %>% 
  arrange(desc(score_average), pctd, image_path)
# first 10 images in this set looked healthy, so keep them 

write_csv(pctd_ne, 'data/ne2025/pctd_all.csv')
pctd_ne_filt <- filter(pctd_ne, !(image_path %in% senesced))
write_csv(pctd_ne_filt, 'data/ne2025/pctd_senesced_removed.csv')

pctd_fvsu <- filter(pctd_final, str_starts(image_path, '25') & str_detect(image_path, '2025-10'))
write_csv(pctd_fvsu, 'data/fvsu2025/pctd_all.csv')
pctd_aamu <- filter(pctd_final, !(image_path %in% c(pctd_ne$image_path, pctd_fvsu$image_path)))
write_csv(pctd_aamu, 'data/aamu2025/pctd_all.csv')

images_keep_ne <- basename(pctd_ne$image_path) %>% 
  str_remove('-05_00_leaf.png') %>% 
  str_remove('-05_00_masked.png')
write.table(images_keep_ne, 'data/ne2025/images_keep_all.csv', quote = FALSE, row.names = FALSE, col.names = FALSE)

images_keep_ne_senesced_removed <- basename(pctd_ne_filt$image_path) %>% 
  str_remove('-05_00_leaf.png') %>% 
  str_remove('-05_00_masked.png')
write.table(images_keep_ne_senesced_removed, 'data/ne2025/images_keep_senesced_removed.csv', quote = FALSE, row.names = FALSE, col.names = FALSE)
