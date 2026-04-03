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

# write_csv(pctd_ne, 'data/ne2025/pctd_all.csv')
pctd_ne_filt <- filter(pctd_ne, !(image_path %in% senesced))
# write_csv(pctd_ne_filt, 'data/ne2025/pctd_senesced_removed.csv')

pctd_fvsu <- filter(pctd_final, str_starts(image_path, '25') & str_detect(image_path, '2025-10'))
# write_csv(pctd_fvsu, 'data/fvsu2025/pctd_all.csv')
pctd_aamu <- filter(pctd_final, !(image_path %in% c(pctd_ne$image_path, pctd_fvsu$image_path)))
# write_csv(pctd_aamu, 'data/aamu2025/pctd_all.csv')

images_keep_ne <- basename(pctd_ne$image_path) %>% 
  str_remove('-05_00_leaf.png') %>% 
  str_remove('-05_00_masked.png')
# write.table(images_keep_ne, 'data/ne2025/images_keep_all.csv', quote = FALSE, row.names = FALSE, col.names = FALSE)

images_keep_ne_senesced_removed <- basename(pctd_ne_filt$image_path) %>% 
  str_remove('-05_00_leaf.png') %>% 
  str_remove('-05_00_masked.png')
# write.table(images_keep_ne_senesced_removed, 'data/ne2025/images_keep_senesced_removed.csv', quote = FALSE, row.names = FALSE, col.names = FALSE)

pctd_quantiles <- quantile(pctd_ne_filt$pctd, probs = c(0, 0.2, 0.4, 0.6, 0.8, 1))
quant1 <- filter(pctd_ne_filt, pctd < pctd_quantiles[2])
quant2 <- filter(pctd_ne_filt, (pctd_quantiles[2] <= pctd) & (pctd < pctd_quantiles[3]))
quant3 <- filter(pctd_ne_filt, (pctd_quantiles[3] <= pctd) & (pctd < pctd_quantiles[4]))
quant4 <- filter(pctd_ne_filt, (pctd_quantiles[4] <= pctd) & (pctd < pctd_quantiles[5]))
quant5 <- filter(pctd_ne_filt, pctd >= pctd_quantiles[5])

# imageNESample200 <- c(sample(quant1$image_path, 40), sample(quant2$image_path, 40), sample(quant3$image_path, 40),
#                     sample(quant4$image_path, 40), sample(quant5$image_path, 40))
# write.table(imageNESample200, 'output/ne2025_images_to_score_200.csv', quote = FALSE, row.names = FALSE, col.names = FALSE)
imageNESample200 <- read_csv('output/ne2025_images_to_score_200.csv', col_names = c('image_path')) %>% 
  mutate(image_id = str_remove(image_path, '-05_00_leaf.png') %>% 
           str_remove('-05_00_masked.png'), 
         plotNumber = str_split_i(image_path, '_', 1) %>% 
           as.numeric())
ne_field_idx <- read_csv('data/ne2025/SbDiv_ne2025_fieldindex.csv')
imageNESample200 <- left_join(imageNESample200, ne_field_idx, join_by(plotNumber))

plotsSample200 <- unique(imageNESample200$plotNumber)
genotypesSample200 <- unique(imageNESample200$genotype)

quant1_filt <- filter(quant1, !(image_path %in% imageNESample200$image_path))
quant2_filt <- filter(quant2, !(image_path %in% imageNESample200$image_path))
quant3_filt <- filter(quant3, !(image_path %in% imageNESample200$image_path))
quant4_filt <- filter(quant4, !(image_path %in% imageNESample200$image_path))
quant5_filt <- filter(quant5, !(image_path %in% imageNESample200$image_path))

proposed800Sample <- c(sample(quant1_filt$image_path, 160), sample(quant2_filt$image_path, 160), sample(quant3_filt$image_path, 160),
                                           sample(quant4_filt$image_path, 160), sample(quant5_filt$image_path, 160))
proposed1kSample <- c(proposed800Sample, imageNESample200$image_path)
# we want reps of: 
##  genotypes
##  w/in plot
proposed1kSampleMetadata <- tibble(image_path = proposed1kSample) %>% 
  mutate(image_id = str_remove(image_path, '-05_00_leaf.png') %>% 
           str_remove('-05_00_masked.png'), 
         plotNumber = str_split_i(image_path, '_', 1) %>% 
           as.numeric()) %>% 
  left_join(ne_field_idx, join_by(plotNumber))
proposedSampleSummaryGenotypes <- proposed1kSampleMetadata %>% 
  group_by(genotype) %>% 
  summarise(plots = n_distinct(plotNumber), 
            images = n())
proposedSampleSummaryPlots <- proposed1kSampleMetadata %>% 
  group_by(plotNumber) %>% 
  summarise(images = n())
write.table(proposed800Sample, 'output/ne2025_images_to_score_800.csv', quote = FALSE, row.names = FALSE, col.names = FALSE)



pctd_quantiles_fvsu <- quantile(pctd_fvsu$ExG_P20_disease_pct, probs = c(0, 0.33, 0.66, 1))
quant1_fvsu <- filter(pctd_fvsu, ExG_P20_disease_pct < pctd_quantiles_fvsu[2])
quant2_fvsu <- filter(pctd_fvsu, (ExG_P20_disease_pct >= pctd_quantiles_fvsu[2]) & (ExG_P20_disease_pct < pctd_quantiles_fvsu[3]))
quant3_fvsu <- filter(pctd_fvsu, ExG_P20_disease_pct >= pctd_quantiles_fvsu[3])

imageFVSUSample30 <- c(sample(quant1_fvsu$image_path, 10), sample(quant2_fvsu$image_path, 10), sample(quant3_fvsu$image_path, 10))
# write.table(imageFVSUSample30, 'output/fvsu2025_images_to_score_30.csv', quote = FALSE, row.names = FALSE, col.names = FALSE)

pctd_quantiles_aamu <- quantile(pctd_aamu$ExG_P20_disease_pct, probs = c(0, 0.33, 0.66, 1))
quant1_aamu <- filter(pctd_aamu, ExG_P20_disease_pct < pctd_quantiles_aamu[2])
quant2_aamu <- filter(pctd_aamu, (ExG_P20_disease_pct >= pctd_quantiles_aamu[2]) & (ExG_P20_disease_pct < pctd_quantiles_aamu[3]))
quant3_aamu <- filter(pctd_aamu, ExG_P20_disease_pct >= pctd_quantiles_aamu[3])

imageAAMUSample30 <- c(sample(quant1_aamu$image_path, 10), sample(quant2_aamu$image_path, 10), sample(quant3_aamu$image_path, 10))
# write.table(imageAAMUSample30, 'output/aamu2025_images_to_score_30.csv', quote = FALSE, row.names = FALSE, col.names = FALSE)
