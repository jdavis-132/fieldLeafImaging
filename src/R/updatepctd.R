library(tidyverse)
source('../widiv-transcriptome/src/Functions.R')
pctd_original <- read_csv('data/manual/ne2025_exg_percent_disease_area.csv')
images_drop <- read_csv('data/manual/final_removed.csv') %>% 
  mutate(image_name = str_remove(image_name, '_overlay') %>% 
           str_replace('_masked', '_leaf.png'))
images_newscores <- read_csv('data/manual/updatedscores(removed19).csv')[2:5] %>% 
  rename(score_average = ExG_P20_disease_pct)

pctd_updated <- pctd_original %>% 
  mutate(image_path = str_split_remove_i(image_name, '_', 1)) %>% 
  filter(!(image_path  %in% c(images_drop$image_name, images_newscores$image_name))) %>% 
  select(!image_path)

images_newscores <- mutate(images_newscores, image_name = str_c(device, image_name, sep = '_'))

pctd_updated <- bind_rows(pctd_updated, images_newscores)

write_csv(pctd_updated, 'data/manual/ne2025_exg_percent_disease_area_updated.csv')
