library(tidyverse)
reseq_genotypes <- read_tsv('data/sorg_reseq_genotypes.txt', col_names = FALSE) %>% 
  pivot_longer(10:931, values_to = 'genotype', names_to = 'idx') %>% 
  pull(genotype)
exclude_genotypes <- read_tsv('data/SamplesToExclude.txt', col_names = FALSE) %>%
  pull(X1)

reseq_genotypes_filt <- setdiff(reseq_genotypes, exclude_genotypes) %>% 
  sort()


sam3_embeddings <- read_csv('output/sam3_rf_predictors.csv')
field_index <- read_csv('data/ne2025/SbDiv_ne2025_fieldindex.csv')
combined <- left_join(field_index, sam3_embeddings, join_by(plotNumber)) %>% 
  select(genotype.x) %>% 
  rename(genotype_idx = genotype.x) %>% 
  mutate(genotype_reseq = str_replace(genotype_idx, 'PI ', 'PI')) %>% 
  mutate(genotype_reseq = case_when(genotype_reseq %in% c("Border", "BTx399", "Check", "Fill", "Mixed", "PI153852", "PI276837",
                                                          "PI533751", "PI533755", "PI533763", "PI533767", "PI533772", "PI533773",
                                                          "PI533775", "PI533779", "PI533780", "PI533781", "PI533787", "PI533788",
                                                          "PI533796", "PI533801", "PI533805", "PI533815", "PI533817", "PI533827",
                                                          "PI533841", "PI533845", "PI533854", "PI533859", "PI533865", "PI533866",
                                                          "PI533883", "PI533886", "PI533889", "PI533890", "PI533891", "PI533896",
                                                          "PI533905", "PI533910", "PI533926", "PI533933", "PI533938", "PI533939",
                                                          "PI533943", "PI533948", "PI533952", "PI533963", "PI533996", "PI533998",
                                                          "PI534000", "PI534003", "PI534009", "PI534018", "PI534024", "PI534039",
                                                          "PI534048", "PI534056", "PI534064", "PI534068", "PI534071", "PI534087",
                                                          "PI534088", "PI534093", "PI534095", "PI534096", "PI534098", "PI534104",
                                                          "PI534121", "PI534123", "PI534125", "PI534127", "PI534129", "PI534130",
                                                          "PI534135", "PI534139", "PI534143", "PI534156", "PI534158", "PI534161",
                                                          "PI534166", "PI561072", "PI561472", "PI574397", "PI574455", "PI576331",
                                                          "PI576335", "PI576336", "PI576344", "PI576346", "PI576349", "PI576350",
                                                          "PI576353", "PI576356", "PI576365", "PI576366", "PI576367", "PI576385",
                                                          "PI576391", "PI576404", "PI576420", "PI576427", "PI576430", "PI585291",
                                                          "PI585295", "PI595712", "PI595729", "PI595730", "PI595743", "PI597944",
                                                          "PI597954", "PI597956", "PI597960", "PI597969", "PI597977", "PI601816",
                                                          "PI603983", "PI605723", "PI605724", "PI605725", "PI605726", "PI606788",
                                                          "PI613610", "PI613611", "PI613612", "PI613613", "PI619402", "PI642793",
                                                          "PI655976", "PI655997", "PI656013", "PI656014", "PI656030", "PI656032",
                                                          "PI656043", "PI656063", "PI656084", "PI656090", "PI656110", "PI656113",
                                                          "PI656114", "PI656115", "PI656116", "PI656117", "PI656119", "PI659692",
                                                          "PI659693", "SC0190", "SC0235", "SC0249", "SC0276", "SC0287", "SC0412",
                                                          "SC0438", "SC0463", "SC0521", "SC0547", "SC0604", "SC0615", "SC0640", 
                                                          "SC0652", "SC0759", "SC0837", "SC0965", "SC0999", "SC1024", "SC1178" ) ~ NA, 
                                    genotype_reseq=="PI543243" ~ "ExPVP_PI543243",
                                    genotype_reseq=="PI543247" ~ "ExPVP_PI543247",
                                    genotype_reseq=="PI544069" ~ "ExPVP_PI544069",
                                    genotype_reseq=="PI554646" ~ "ExPVP_PI554646",
                                    genotype_reseq=="PI554648" ~ "ExPVP_PI554648",
                                    genotype_reseq=="PI554649" ~ "ExPVP_PI554649",
                                    genotype_reseq=="PI554650" ~ "ExPVP_PI554650",
                                    genotype_reseq=="PI554654" ~ "ExPVP_PI554654",
                                    genotype_reseq=="PI555457" ~ "ExPVP_PI555457",
                                    genotype_reseq=="PI561926" ~ "ExPVP_PI561926",
                                    genotype_reseq=="PI562623" ~ "ExPVP_PI562623",
                                    genotype_reseq=="PI574398" ~ "ExPVP_PI574398", 
                                    genotype_reseq=="PI574407" ~ "ExPVP_PI574407", 
                                    genotype_reseq=="PI594355" ~ "ExPVP_PI594355", 
                                    genotype_reseq=="PI596332" ~ "ExPVP_PI596332", 
                                    genotype_reseq=="PI596567" ~ "ExPVP_PI596567",
                                    genotype_reseq=="PI601415" ~ "ExPVP_PI601415", 
                                    genotype_reseq=="PI601553" ~ "ExPVP_PI601553",
                                    genotype_reseq=="PI601554" ~ "ExPVP_PI601554", 
                                    genotype_reseq=="PI601555" ~ "ExPVP_PI601555", 
                                    genotype_reseq=="PI601556" ~ "ExPVP_PI601556", 
                                    genotype_reseq=="PI601557" ~ "ExPVP_PI601557", 
                                    genotype_reseq=="PI601716" ~ "ExPVP_PI601716", 
                                    genotype_reseq=="PI601721" ~ "ExPVP_PI601721", 
                                    genotype_reseq=="PI601743" ~ "ExPVP_PI601743", 
                                    genotype_reseq=="PI601744" ~ "ExPVP_PI601744", 
                                    genotype_reseq=="PI601756" ~ "ExPVP_PI601756", 
                                    genotype_reseq=="PI602599" ~ "ExPVP_PI602599", 
                                    genotype_reseq=="PI602600" ~ "ExPVP_PI602600", 
                                    .default = genotype_reseq))

ne2025_reseq_genotype_alignment <- combined %>% 
  select(genotype_idx, genotype_reseq)
write_csv(ne2025_reseq_genotype_alignment, 'data/ne2025/genotype_alignment_reseq.csv')
write.table(unique(ne2025_reseq_genotype_alignment$genotype_reseq), 'data/ne2025/reseq_genotypes_to_keep.txt', quote = FALSE, 
            sep = '\t', row.names = FALSE, col.names = FALSE)
