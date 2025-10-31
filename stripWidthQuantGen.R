library(tidyverse)
library(readxl)
library(paletteer)
library(cowplot)
source('../widiv-transcriptome/src/Functions.R')

field_map <- read_csv('data/ne2025/SbDiv_ne2025_fieldindex.csv')

strip_widths <- read_csv('output/stripSegmentation/ne2025/all_strip_measurements.csv') %>% 
  mutate(device = str_split_i(image_name, '_', 1) %>%
           str_remove('device'),
         plotNumber = str_split_i(image_name, '_', 2) %>% 
           as.numeric(), 
         strip_id = as.factor(strip_id)) %>% 
  left_join(field_map, join_by(plotNumber==plot), relationship = 'many-to-one')

# look at vp on a strip by strip basis
phenos <- c('mean_height_px', 'median_height_px', 'min_height_px', 'max_height_px', 'height_range_px', 'area_px')

vp_strips <- tibble()
for (p in phenos)
{
  vp_strips <- bind_rows(vp_strips, 
                         partitionVariance3(strip_widths, p, p, '~ (1|genotype) + (1|row) + (1|range) + (1|device) + (1|strip_id) + (1|sam_score)'))
}

vp_strips_nonresidual <- filter(vp_strips, grp!='Residual') %>% 
  mutate(grp = factor(grp, levels = c('device', 'strip_id', 'sam_score', 'row', 'range', 'genotype')))
vp_strips.plot <- ggplot(vp_strips_nonresidual, aes(label, pctVar, fill = grp)) + 
  geom_col() + 
  theme_use 
vp_strips.plot

# combine across strips
leaf_widths <- strip_widths %>% 
  group_by(image_name, device, genotype, row, range) %>% 
  summarise(across(c(all_of(phenos), sam_score), ~sum(.x, na.rm = TRUE)))

vp_leaf <- tibble()
for (p in phenos)
{
  vp_leaf <- bind_rows(vp_leaf, 
                         partitionVariance3(leaf_widths, p, p, '~ (1|genotype) + (1|row) + (1|range) + (1|device) + (1|sam_score)'))
}

vp_leaf_nonresidual <- filter(vp_leaf, grp!='Residual') %>% 
  mutate(grp = factor(grp, 
                      levels = c('sam_score', 'device', 'row', 'range', 'genotype'), 
                      labels = c('Segmentation Score', 'Device', 'Row', 'Range', 'Genotype')), 
         label = factor(label, 
                        levels = c('max_height_px', 'median_height_px', 'mean_height_px', 'area_px', 'height_range_px', 'min_height_px'), 
                        labels = c('Maximum \nWidth', 'Median \nWidth', 'Mean \nWidth', 'Area', 'Width \nRange', 'Minimum \nWidth')))
vp_leaf.plot <- ggplot(vp_leaf_nonresidual, aes(label, pctVar, fill = grp)) + 
  geom_col() + 
  scale_y_continuous(name = 'Variance Explained', 
                     breaks = seq(0, 100, by = 10), labels = ~str_c(.x, '%'), expand = c(0, 0)) +
  scale_fill_paletteer_d("nationalparkcolors::Redwoods", direction = -1) +
  labs(x = NULL, fill = NULL) +
  theme_use + 
  theme(text = element_text(color = 'black', size = 16),
        axis.line.x.bottom = element_line(color = 'black'),
        axis.line.y.left = element_line(color = 'black'))
vp_leaf.plot

genotype_ids <- read_tsv('data/sorgRNAseq_811.fam', col_names = FALSE) #%>% 
  select(X2) %>% 
  rename(vcfName = X2) %>% 
  mutate(phenoName = str_replace(vcfName, 'PI', 'PI ')) %>% 
  mutate(phenoName = case_when(
                               # vcfName=='ElMota' ~ 'PI 656035',
                               vcfName=='BTx378' ~ 'PI 655991', 
                               # vcfName=='RTx2917' ~ 'PI 629040',
                               # vcfName=='RTX2737' ~ 'PI 655978', 
                               # vcfName=='SanChiSan' ~ 'PI 542718',
                               # vcfName=='PI655993' ~ 'BTx399', 
                               vcfName=='RTx430' ~ 'PI 655996', 
                               # vcfName=='BTx3197' ~ 'PI 655992', 
                               .default = phenoName))
leaf_widths_gwas <- ungroup(leaf_widths) %>% 
  left_join(genotype_ids, join_by(genotype==phenoName), relationship = 'many-to-one') %>% 
  filter(!is.na(vcfName)) %>% 
  select(vcfName, all_of(phenos)) %>% 
  rename(genotype = vcfName)

blues <- tibble()
for (p in phenos)
{
  model <- lm()
}

write_csv(leaf_widths_gwas, 'leaf_widths_gwasphenos.csv')
write_tsv(as.data.frame(unique(leaf_widths_gwas$genotype)), 'genotypes_prelimgwas.txt', col_names = FALSE)

# field data: how does it compare?
sap2021 <- read_excel('SAPMerged2021_v2.3.xlsx')
sap_vp <- partitionVariance3(sap2021, 'LeafWidth', 'Leaf Width \n(Manual)', '~ (1|Row) + (1|Column) + (1|PINumber)') %>% 
  filter(grp!='Residual') %>% 
  mutate(grp = factor(grp, levels = c('Row', 'Column', 'PINumber'), labels = c('Row', 'Range', 'Genotype')))

colors <- paletteer_d("nationalparkcolors::Redwoods", 5, direction = -1)[3:5]
sap_vp.plot <- ggplot(sap_vp, aes(label, pctVar, fill = grp)) + 
  geom_col() + 
  scale_y_continuous(name = 'Variance Explained', 
                     breaks = seq(0, 100, by = 10), labels = ~str_c(.x, '%'), expand = c(0, 0), limits = c(0, 75)) +
  scale_fill_manual(values = colors) +
  labs(x = NULL, fill = NULL) +
  theme_use + 
  theme(text = element_text(color = 'black', size = 16),
        axis.line.x.bottom = element_line(color = 'black'),
        axis.line.y.left = element_line(color = 'black'), 
        legend.position = 'none')
sap_vp.plot

# leaf_width_combined <- left_join(leaf_widths, sap2021, join_by(genotype==PINumber)) %>% 
#   filter(str_detect(genotype, 'PI') & !is.na(LeafWidth))
# 
# corr <- ggplot(filter(leaf_width_combined, LeafWidth < 20), aes(LeafWidth, mean_height_px)) + 
#   geom_bin_2d()
# corr
plots <- align_plots(vp_leaf.plot, sap_vp.plot, aligh = 'v', axis = 'l')
fig <- plot_grid(plots[[1]], sap_vp.plot, rel_widths=c(0.8, 0.2))
fig
