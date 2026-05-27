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
  select(image_id, percentUnhealthy)

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

images_ne <- filter(all_raw_images, location=='NE') %>% 
  left_join(idx_ne, join_by(plotNumber)) %>% 
  mutate(block = as.numeric(rep)) %>% 
  select(!rep) %>% 
  left_join(pctd_ne, join_by(image_id)) %>% 
  left_join(segmentation_ne, join_by(image_path))
images_al <- filter(all_raw_images, location=='AL') %>% 
  left_join(idx_al, join_by(plotNumber)) %>% 
  left_join(pctd_al, join_by(image_id)) %>%
  left_join(segmentation_al, join_by(image_path))
images_ga <- filter(all_raw_images, location=='GA') %>% 
  left_join(idx_ga, join_by(plotNumber)) %>% 
  mutate(block = as.numeric(rep)) %>% 
  select(!c(S.No, rep)) %>% 
  left_join(pctd_ga, join_by(image_id)) %>% 
  left_join(segmentation_ga, join_by(image_id))

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
  left_join(human_scores, join_by(image_id, location)) %>% 
  mutate(excluded = is.na(percentUnhealthy))
write_csv(metadata, 'data/image_metadata.csv')

# figure out why this does not have length 567 as calculated for variables.tex last week
# 5 - need to fix images that initially failed but DO have masks
image_ids_exclude <- metadata$image_id[which(metadata$excluded)]
