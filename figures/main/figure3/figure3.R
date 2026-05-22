library(tidyverse)
source('src/R/Functions.R')
library(ggrastr)
library(paletteer)


theme_use <- theme_minimal() +
  theme(axis.text.x = element_text(size = 9, color = 'black', margin = margin(0, 0, 0, 0), 
                                   vjust = 0.5, hjust = 0.5),
        axis.text.y = element_text(size = 9, color = 'black', vjust = 0.5, hjust = 0.5),
        legend.text = element_text(size = 9, color = 'black', vjust = 0.5, hjust = 0.5),
        plot.title = element_text(size = 9, color = 'black', vjust = 0, hjust = 0.5),
        plot.subtitle = element_text(size = 9, color = 'black', vjust = 0, hjust = 0.5),
        text = element_text(size = 9, color = 'black'),
        legend.position = 'top',
        line = element_line(color = 'black', linewidth = 1),
        axis.ticks = element_line(color = 'black', linewidth = 0.5),
        axis.line.x.bottom = element_line(color = 'black', linewidth = 0.5),
        axis.line.y.left = element_line(color = 'black', linewidth = 0.5),
        panel.grid = element_blank(), 
        panel.background = element_blank())

mlm_mean768pmap <- read_csv('output/mlm_20260515/GWAS_embedding_mean_768_all_results.csv')
mostSigMaxPval <- quantile(mlm_mean768pmap$MLM_P, probs = c(0.00001))
mostSigSNPs <- filter(mlm_mean768pmap, MLM_P < mostSigMaxPval)

regionChr <- 6
regionStart <- 58560985
regionEnd <- 61826680
peakSNP <- 59555405
pLocusCoordinates <- 58582313:58584616
regionSNPS <- filter(mostSigSNPs, CHROM==6 & between(POS, regionStart, regionEnd))
pLocusID <- 'Sobic.006G226800'
peakSNPInPLocus <- 58584404

# custom manhattan plotting because we don't want a threshold + we want to color based on if the pval meets a quantile threshold
chromLength <- tibble(max_bp = c(85112863, 79114963, 80873341, 71215609, 77058072, 
                                 62713908, 68911884, 65779274, 63277606, 62870657), 
                      CHROM = 1:10) %>% 
  arrange(CHROM) %>%
  mutate(bp_add = (lag(cumsum(max_bp), default = 0) + (CHROM - 1)*1e7))
n_chromosomes <- length(chromLength$max_bp)
last_chr_len <- chromLength$max_bp[n_chromosomes]
ylab <-  expression(-log[10]~'(p)')

mlm_mean768pmap <- mlm_mean768pmap %>% 
  mutate(colorCategory = case_when(MLM_P < mostSigMaxPval ~ 'mostSigSNPs', 
                                   !(MLM_P < mostSigMaxPval) & ((CHROM %% 2) == 0) ~ 'evenCHROM', 
                                   .default = 'oddCHROM')) %>% 
  mutate(colorCategory = factor(colorCategory, levels = c('oddCHROM', 'evenCHROM', 'mostSigSNPs'))) %>% 
  full_join(chromLength, join_by(CHROM)) %>%
  rowwise() %>%
  mutate(loc = POS + bp_add, 
         MLM_P = -1*log10(MLM_P))

yrange <- mlm_mean768pmap %>% 
  ungroup() %>%
  summarise(yrange = (max(MLM_P, na.rm = TRUE) - min(MLM_P, na.rm = TRUE)))
yrange <- as.numeric(yrange[1, 1])
ylim <- mlm_mean768pmap %>% 
  ungroup() %>%
  summarise(ylim = (max(MLM_P, na.rm = TRUE) + 0.1*yrange))
ylim <- as.numeric(ylim[1, 1])

xlimit <- last_chr_len + max(chromLength$bp_add)
x_axis_set <- chromLength %>% 
  arrange(CHROM) %>% 
  mutate(center = (bp_add + lead(bp_add, default = xlimit))/2)

colors <- c('black', 'darkgrey', paletteer_d("RColorBrewer::Paired", 10)[10])

manhattan <- ggplot(mlm_mean768pmap, aes(loc, MLM_P, color = colorCategory)) + 
  ggrastr::rasterise(geom_point(), dpi = 1000) + 
  annotate('point', x = mean(pLocusCoordinates) + chromLength$bp_add[6], y = 0, size = 6, color = colors[3], shape = 17) +
  scale_x_continuous(labels = chromLength$CHROM, 
                     breaks = x_axis_set$center, 
                     limits = c(0, xlimit), 
                     expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0), 
                     limits = c(0, ylim)) +
  coord_cartesian(clip = "off") + 
  scale_color_manual(values = colors) + 
  labs(x = 'Chromosome', y = ylab, color = NULL) + 
  theme_use + 
  theme(legend.position = 'none')
ggsave('figures/main/figure3/manhattan.png', width = 6.5, height = 3, dpi = 1000, 
       bg = 'transparent')

blues <- read_csv('output/sam3_blues.csv')

histogram <- ggplot(blues, aes(embedding_mean_768)) + 
  geom_histogram(fill = colors[3], color = colors[3]) + 
  annotate('point', x = -0.001370542, y = 15, size = 3) + 
  annotate('point', x = 0.2182325, y = 15, size = 3) +
  scale_x_continuous(name = 'Mean 768 Value', 
                     expand = c(0, 0)) + 
  scale_y_continuous(name = 'Genotypes', 
                     expand = c(0, 0)) + 
  theme_use
histogram
ggsave('figures/main/figure3/histogram_blues.svg', plot = histogram, width = 4.8, height = 3, units = 'in', dpi = 1e3, bg = 'transparent')

vcf_sig <- read_tsv('output/all_sig_snps.recode.vcf', skip = 98)
colnames(vcf_sig) <- c('CHROM', str_replace(colnames(vcf_sig)[2:length(colnames(vcf_sig))], 'SC', 'SC '))
vcf_sig <- vcf_sig %>%
  select(!c(QUAL, INFO, FORMAT, FILTER)) %>%
  mutate(ID = str_c(CHROM, POS, REF, ALT, sep = ':')) %>%
  pivot_longer(cols = !c(CHROM, POS, ID, REF, ALT), 
               names_to = 'genotype', 
               values_to = 'allele') %>% 
  select(ID, genotype, allele) %>% 
  pivot_wider(id_cols = genotype, 
              names_from = ID, 
              values_from = allele)
markers <- colnames(vcf_sig)[2:ncol(vcf_sig)]

blues_unl <- blues %>% 
  mutate(location = 'UNL')
blues_fvsu <- read_csv('output/sam3_fvsu_blues.csv') %>% 
  mutate(location = 'FVSU')
blues_aamu <- read_csv('output/sam3_aamu_blues.csv') %>% 
  mutate(location = 'AAMU')
blues_all <- bind_rows(blues_unl, blues_fvsu, blues_aamu) %>% 
  filter(!genotype %in% c('Border', 'Check', 'Fill', 'Mixed')) %>% 
  left_join(vcf_sig, join_by(genotype)) %>% 
  mutate(across(all_of(markers), ~case_when(.x %in% c('0/1', '1/0', './.', '.', '0|1') ~ NA,
                                            .x=='1|1' ~ '1/1',
                                            .x=='0|0' ~ '0/0',
                                            .default = .x))) %>%
  select(c(location, genotype, all_of(c('6:58584404:G:A', 'embedding_mean_768')))) %>% 
  mutate(`6:58584404:G:A` = factor(`6:58584404:G:A`, levels = c("0/0", "1/1"))) %>%
  filter(!is.na(`6:58584404:G:A`)) 

genotypes_intersect <- intersect(blues_unl$genotype, blues_aamu$genotype)
genotypes_intersect <- intersect(genotypes_intersect, blues_fvsu$genotype)
blues_unl_intersect <- filter(blues_all, location=='UNL' & genotype %in% genotypes_intersect) %>% 
  mutate(location = 'UNL-SAP')
blues_all <- bind_rows(blues_all, blues_unl_intersect)

wilcox.test(embedding_mean_768 ~ `6:58584404:G:A`, data = blues_all, subset = (location=='UNL'))
wilcox.test(embedding_mean_768 ~ `6:58584404:G:A`, data = blues_all, subset = (location=='UNL-SAP'))
wilcox.test(embedding_mean_768 ~ `6:58584404:G:A`, data = blues_all, subset = (location=='AAMU'))
wilcox.test(embedding_mean_768 ~ `6:58584404:G:A`, data = blues_all, subset = (location=='FVSU'))


blues_summary <- blues_all %>% 
  filter(!is.na(`6:58584404:G:A`)) %>% 
  group_by(`6:58584404:G:A`, location) %>% 
  summarise(mean = mean(embedding_mean_768, na.rm = TRUE),
            se = sd(embedding_mean_768, na.rm = TRUE)/sqrt(n()), 
            n = n()) %>%
  mutate(location = factor(location, levels = c('UNL', 'UNL-SAP', 'AAMU', 'FVSU')))

barplot <- ggplot(blues_summary, aes(location, mean, fill = `6:58584404:G:A`)) + 
  geom_col(position = position_dodge(width = 0.9)) + 
  geom_errorbar(aes(ymin  = mean - se, ymax = mean + se), 
                position = position_dodge(width = 0.9), width = 0.25) + 
  annotate(geom = 'text',
           x = c('UNL', 'UNL-SAP', 'AAMU', 'FVSU'),
           y = rep(0.15, 4),
           label = c('****', '*', '**', '****')) +
  scale_x_discrete(name = NULL, 
                   expand = c(0, 0),
                   labels = c('NE', 'NE-C', 'AL', 'GA')) + 
  scale_y_continuous(name = 'Mean 768 Value', 
                     expand = c(0, 0)) +
  scale_fill_manual(name = 'Chr6:\n58584404', 
                    values = paletteer_d('RColorBrewer::Paired', 10)[9:10], 
                    label = c('G', 'A')) + 
  theme_use +
  theme(axis.text.x = element_text(angle = 90))
barplot

ggsave('figures/main/figure3/barplot_external_envs.svg', plot = barplot, width = 1.6, height = 3.25, dpi = 1e3, bg = 'transparent')
# aamu_boxplot <- ggplot(aamu_df, aes(`6:58584404:G:A`, embedding_mean_768, fill = `6:58584404:G:A`)) + 
#   geom_boxplot() + 
#   annotate('text', x = c('0/0', '1/1'), y = c(0.5, 0.5), label = c('n=2976', 'n=1306')) +
#   scale_fill_manual(values = paletteer_d("RColorBrewer::Paired", 10)[9:10]) + 
#   scale_x_discrete(name = 'Allele', labels = c('REF', 'ALT'), 
#                    expand = c(0, 0)) + 
#   scale_y_continuous(name = '768 Mean', 
#                      expand = c(0, 0), 
#                      limits = c(-0.35, 0.55)) + 
#   labs(title = 'AL') +
#   theme_use + 
#   theme(legend.position = 'none')
# aamu_boxplot
# ggsave('figures/main/figure3/aamu_boxplot.svg', plot = aamu_boxplot, width = 2.7, height = 2, dpi = 1000, bg = 'transparent')
