library(tidyverse)
library(readxl)
library(paletteer)
source('../widiv-transcriptome/src/Functions.R')

field_map <- read_csv('data/ne2025/SbDiv_ne2025_fieldindex.csv')

strip_widths <- read_csv('output/stripSegmentation/ne2025/all_strip_measurements.csv') %>% 
  mutate(device = str_split_i(image_name, '_', 1) %>%
           str_remove('device') %>%
           as.numeric(),
         plotNumber = str_split_i(image_name, '_', 2) %>% 
           as.numeric()) %>% 
  left_join(field_map, join_by(plotNumber==plot), relationship = 'many-to-one')
