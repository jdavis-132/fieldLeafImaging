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

images_keep <- all_fvsu_images %>% 
  group_by(image_id) %>% 
  summarise(image_path = first(image_path)) %>% 
  pull(image_path)
write.table(images_keep, 'data/fvsu2025/unique_images.txt', quote = FALSE, sep = '\t', row.names = FALSE, col.names = FALSE)
write.table(unique_image_ids, 'data/fvsu2025/unique_image_ids.txt', quote = FALSE, sep = '\t', row.names = FALSE, col.names = FALSE)
