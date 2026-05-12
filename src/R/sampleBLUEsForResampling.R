library(tidyverse)

high_fi_features <- c("embedding_std_930", "embedding_std_552", "embedding_std_918", "embedding_mean_637", "embedding_std_976", "embedding_mean_886",
                      "embedding_mean_210", "embedding_std_383", "embedding_std_687", "embedding_mean_984", "embedding_mean_68", "embedding_mean_836",
                      "embedding_mean_586", "embedding_std_968", "embedding_mean_546", "embedding_std_821", "embedding_mean_656", "embedding_mean_968",
                      "embedding_mean_698", "embedding_mean_165", "embedding_mean_37", "embedding_mean_582", "embedding_mean_214", "embedding_mean_792",
                      "embedding_std_606", "embedding_mean_930", "embedding_mean_734", "embedding_mean_108", "embedding_mean_560", "embedding_std_166",
                      "embedding_std_983", "embedding_std_981", "embedding_mean_197", "embedding_std_817", "embedding_std_132", "embedding_std_82",
                      "embedding_mean_950", "embedding_mean_119", "embedding_std_128", "embedding_mean_139", "embedding_std_76", "embedding_std_839",
                      "embedding_std_326", "embedding_mean_842", "embedding_std_567", "embedding_mean_237", "embedding_std_793")

blues <- read_csv('output/sam3_blues.csv')

samples <- select(blues, c(genotype, all_of(high_fi_features)))

n_samples <- 100
n_obs <- as.integer(nrow(blues_select))
obs_set_na <- as.integer((1 - 0.9)*n_obs)

for(trait in high_fi_features)
{
  for(i in 1:n_samples)
  {
    samples <- samples %>% 
      mutate('{trait}_{i}':= replace(.data[[trait]], sample(1:n_obs, obs_set_na), NA))
  }
  samples <- select(samples, !any_of(trait))
}

write_csv(samples, 'output/sam3_blues_highfi_samples.csv')

