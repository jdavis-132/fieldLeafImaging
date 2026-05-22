library(tidyverse)

args <- commandArgs(trailingOnly = FALSE)
path <- str_remove(args[length(args) - 2], fixed('-'))
threshold <- as.numeric(str_remove(args[length(args) - 1], fixed('-')))
out <- str_remove(args[length(args)], fixed('-'))

summariseSignals_PANICLE <- function(path)
{
  files <- Sys.glob(path)
  signals <- tibble()
  for(f in files)
  {
    df <- read_csv(f,
                   col_types = 'ccnccnnnc', 
                   col_names = c('SNP', 'CHROM', 'POS', 'REF', 'ALT', 'MAF', 'pval', 'effect', 'method'),
                   skip = 1) %>% 
      mutate(filename = f)
    signals <- bind_rows(signals, df)
  }
  return(signals)
}

pmap <- summariseSignals_PANICLE(path)
signals <- filter(pmap, pval < threshold)
write_csv(signals, out)
