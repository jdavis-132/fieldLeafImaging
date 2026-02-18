library(tidyverse)
kmer_names <- read_tsv('data/kmers_table.names', col_names = c('genotype_kmer')) %>% 
  mutate(PI_num = str_remove(genotype_kmer, 'PI ') %>% 
           str_remove('ExPVP_PI') %>% 
           str_remove('PI_') %>% 
           str_remove('PI'))
reseq_genotypes <- read_tsv('data/sorg_reseq_genotypes.txt', col_names = FALSE) %>% 
  pivot_longer(10:931, values_to = 'genotype', names_to = 'idx') %>% 
  pull(genotype)
exclude_genotypes <- read_tsv('data/SamplesToExclude.txt', col_names = FALSE) %>%
  pull(X1)

reseq_genotypes_filt <- setdiff(reseq_genotypes, exclude_genotypes) %>% 
  sort()
reseq_names <- tibble(genotype_reseq = reseq_genotypes_filt) %>% 
  mutate(PI_num = str_remove(genotype_reseq, 'PI ') %>% 
          str_remove('ExPVP_PI') %>% 
          str_remove('PI_') %>% 
          str_remove('PI') %>% 
          str_remove('_1'))
  
kmer_genotype_alignment <- inner_join(kmer_names, reseq_names, join_by(PI_num))

write_csv(kmer_genotype_alignment, 'data/ne2025/kmer_genotype_alignment.csv')
