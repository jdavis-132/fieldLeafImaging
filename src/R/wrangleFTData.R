library(tidyverse)
library(lubridate)
source('src/R/Functions.R')

r1 <- read_csv('data/manual/2025-09-02-03-00-07_2025_SbDiv_FieldBook_Rep1_table.csv') %>% 
  mutate(rep = 1)
r2 <- read_csv('data/manual/2025-09-02-04-31-31_2025_SbDiv_FieldBook_Rep2_table.csv') %>% 
  mutate(rep = 2)

ft_data <- bind_rows(r1, r2) %>% 
  filter(!is.na(FLOWERING))
notes_drop <- c("Variation in flowering", "dirs", "missed to collect at right time", "not captured, right time",
                "missed to capture on right time", "missed to capture at right time", "missed to capture in time",
                "difficult to capture the right stagey7", "DIRS(difficult to identify right stage)", "data captured late",
                "DIRS", "missed to capture at the right time", "missed at right time", "DiRS", "missed to capture",
                "missed capture at the right time", "missed on right time", "missed")
ft_data <- ft_data %>% 
  filter(!(Notes %in% notes_drop)) %>% 
  rename(range = Range, 
         row = Row, 
         plotNumber = `Plot ID`, 
         genotype = Genotype) %>% 
  mutate(FLOWERING_DATE = mdy(FLOWERING))

planting_date <- mdy('06/10/2025')

ft_data <- ft_data %>% 
  mutate(days_to_flower = FLOWERING_DATE - planting_date) %>%
  select(plotNumber, days_to_flower)

write_csv(ft_data, 'data/manual/SbDiv_ne2025_FT_clean.csv')
