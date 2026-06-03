library(tidyverse)
source('src/R/Functions.R')

ne_raw_images <- read_csv('data/ne2025/all_raw_images.csv', col_names = 'image_path') %>% 
  mutate(location = 'NE')
al_raw_images <- read_csv('data/aamu2025/all_raw_images.csv', col_names = 'image_path') %>% 
  mutate(location = 'AL')
ga_raw_images <- read_csv('data/fvsu2025/all_raw_images.csv', col_names = 'image_path') %>%
  mutate(location = 'GA')

all_raw_images <- bind_rows(ne_raw_images, al_raw_images, ga_raw_images) %>% 
  mutate(image_id = basename(image_path) %>%
           str_split_remove_i(fixed('-'), 4), 
         plotNumber = str_split_i(image_id, '_', 1) %>% 
           as.numeric(),
         device = case_when(location=='NE' ~ str_split_i(image_path, fixed('/'), 3) %>% 
                              str_remove('device') %>%
                              as.numeric()))
idx_ne <- read_csv('data/ne2025/SbDiv_ne2025_fieldindex.csv')
idx_al <- read_csv('data/aamu2025/aamu_field_index.csv')
idx_ga <- read_csv('data/fvsu2025/fvsu_field_index.csv')

pctd_ne <- read_csv('data/ne2025/pctd_all.csv') %>% 
  mutate(image_id = str_split_remove_i(image_path, fixed('-'), 4)) %>% 
  rename(percentUnhealthy = pctd) %>% 
  select(image_id, percentUnhealthy) %>% 
  # segmentation failed for me but somehow still have pctd? remove these pctd values
  filter(!(image_id %in% c('1030_LeafPhotoA_2025-09-08 10_05_24.735', 
                           '1009_LeafPhotoA_2025-09-08 09_39_52.913',
                           '2771_LeafPhotoA_2025-09-09 15_51_03.374')))

pctd_al <- read_csv('data/aamu2025/pctd_all.csv') %>% 
  mutate(image_id = str_split_remove_i(image_path, fixed('-'), 4)) %>% 
  rename(percentUnhealthy = ExG_P20_disease_pct) %>% 
  select(image_id, percentUnhealthy)
pctd_ga <- read_csv('data/fvsu2025/pctd_all.csv') %>% 
  mutate(image_id = str_split_remove_i(image_path, fixed('-'), 4)) %>% 
  rename(percentUnhealthy = ExG_P20_disease_pct) %>% 
  select(image_id, percentUnhealthy)

segmentation_ne <- read_csv('data/processed/ne2025/segmentation_methods.csv') 
segmentation_al <- read_csv('data/processed/aamu2025/segmentation_methods.csv')
segmentation_ga <- read_csv('data/processed/fvsu2025/segmentation_methods.csv') %>% 
  mutate(image_id = basename(image_path) %>% 
           str_split_remove_i(fixed('-'), 4)) %>% 
  select(image_id, segmentation_method) %>%
  distinct()

images_keep_ne <- read_csv('data/ne2025/images_keep_all.csv', col_names = c('image_id'))
images_keep_al <- read_csv('data/aamu2025/image_ids_keep.txt', col_names = c('image_id'))
images_keep_ga <- read_csv('data/fvsu2025/image_ids_keep.txt', col_names = c('image_id'))

images_ne <- filter(all_raw_images, location=='NE') %>% 
  left_join(idx_ne, join_by(plotNumber)) %>% 
  mutate(block = as.numeric(rep)) %>% 
  select(!rep) %>% 
  left_join(pctd_ne, join_by(image_id)) %>% 
  left_join(segmentation_ne, join_by(image_path)) %>% 
  mutate(excluded = case_when(!(image_id %in% images_keep_ne$image_id) 
                              & segmentation_method!='Failed' ~ TRUE, .default = FALSE)) %>% 
  # initially failed but segmented on second run
  mutate(segmentation_method = case_when(image_id %in% c('1635_LeafPhotoA_2025-09-09 10_20_03.665',
                                                         '2602_LeafPhotoA_2025-09-09 10_55_46.992') ~ 'SAM3',
                                         .default = segmentation_method))

images_al <- filter(all_raw_images, location=='AL') %>% 
  left_join(idx_al, join_by(plotNumber)) %>% 
  left_join(pctd_al, join_by(image_id)) %>%
  left_join(segmentation_al, join_by(image_path)) %>% 
  mutate(excluded = case_when(!(image_id %in% images_keep_al$image_id) 
                              & segmentation_method!='Failed' ~ TRUE, .default = FALSE))
images_ga <- filter(all_raw_images, location=='GA') %>% 
  left_join(idx_ga, join_by(plotNumber)) %>% 
  mutate(block = as.numeric(rep)) %>% 
  select(!c(S.No, rep)) %>% 
  left_join(pctd_ga, join_by(image_id)) %>% 
  left_join(segmentation_ga, join_by(image_id)) %>% 
  mutate(excluded = case_when(!(image_id %in% images_keep_ga$image_id) 
                              & segmentation_method!='Failed' ~ TRUE, .default = FALSE))

genotype_alignment <- read_tsv('data/genotype_conversion_table.tsv', col_names = c('genotype_idx', 'genotype_markers'))


images <- bind_rows(images_ne, images_al, images_ga) %>% 
  mutate(genotype = str_remove_all(genotype, ' ')) %>%
  left_join(genotype_alignment, join_by(genotype==genotype_idx)) %>%
  mutate(genotype_markers = case_when(is.na(genotype_markers) ~ genotype,
                                      .default = genotype_markers)) %>% 
  filter(!is.na(genotype_markers)) %>%
  select(!c(genotype)) %>% 
  rename(genotype = genotype_markers) %>% 
  mutate(poundsOfNitrogenPerAcre = case_when(location=='NE' ~ 90, 
                                             location=='GA' ~ NA, 
                                             .default = poundsOfNitrogenPerAcre))
ne_human_scores <- read_csv('data/manual/all_image_scores.csv') %>% 
  select(image_id, username, comment, score) %>% 
  mutate(username = case_when(username=='Libia' ~ 'A', 
                              username=='Ruben' ~ 'B'), 
         location = 'NE') %>%
  pivot_wider(id_cols = c(image_id, location), 
              names_from = username, 
              values_from = c(score, comment))
al_ga_human_scores <- read_csv('data/manual/image_scores_al_ga.csv') %>% 
  mutate(location = case_when(location=='AAMU' ~ 'AL',
                              location=='FVSU' ~ 'GA'), 
         username = case_when(username=='Libia' ~ 'A', 
                              username=='Ruben' ~ 'B'), 
         image_id = str_split_remove_i(image, fixed('-'), 4)) %>% 
  select(image_id, username, comment, score, location) %>% 
  distinct() %>% 
  pivot_wider(id_cols = c(image_id, location), 
              names_from = username, 
              values_from = c(score, comment))

human_scores <- bind_rows(ne_human_scores, al_ga_human_scores)

metadata <- images %>% 
  left_join(human_scores, join_by(image_id, location))
write_csv(metadata, 'data/image_metadata.csv')

blues_ne <- read_csv('output/sam3_blues.csv') %>% 
  mutate(location = 'NE')
blues_al <- read_csv('output/sam3_aamu_blues.csv') %>% 
  mutate(location = 'AL')
blues_ga <- read_csv('output/sam3_fvsu_blues.csv') %>% 
  mutate(location = 'GA')
blues_pctd <- read_csv('output/pctd_blues.csv') %>% 
  rename(percentUnhealthy = pctd)
blues_scores <- read_csv('output/human_scores_blues.csv') %>% 
  rename(human_score = score)

blues <- blues_ne %>% 
  left_join(blues_pctd, join_by(genotype)) %>% 
  left_join(blues_scores, join_by(genotype)) %>% 
  bind_rows(blues_al, blues_ga) %>% 
  relocate(location, genotype, human_score, percentUnhealthy)
write_csv(blues, 'data/blues_all.csv')
