library(tidyverse)

all_fvsu_images <- read_tsv('data/fvsu2025/image_files.txt', col_names = 'image_path') %>% 
  mutate(subdir = str_split_i(image_path, fixed('/'), 8), 
         image_id = str_split_i(image_path, fixed('/'), 9) %>% 
           str_remove('-05_00.jpg')) #%>%
  # add_count(image_id) %>% 
  # arrange(desc(n), image_id, subdir) %>% 
  # filter(n > 1) %>% 
  # group_by(image_id) %>% 
  # summarise(subdirs = unique(subdir) %>% 
  #             toString())

unique_image_ids <- unique(all_fvsu_images$image_id)
potential_problem_images <- unique_image_ids[c(447, 1541)]
