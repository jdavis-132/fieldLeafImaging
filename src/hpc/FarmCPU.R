library(rMVP)
library(readr)
library(data.table)
library(dplyr)
library(tidyverse)

args <- commandArgs(trailingOnly = FALSE)
pheno_file <- str_remove(args[length(args)-3], fixed('-'))
prefix <- str_remove(args[length(args)-2], fixed('-'))
TOTAL_MARKERS <- as.numeric(str_remove(args[length(args)-1], fixed('-')))
EFFECTIVE_MARKERS <- as.numeric(str_remove(args[length(args)], fixed('-'))) 
geno_stem <- str_c(prefix, '.mvp')

rMVP::MVP.Data(fileBed=prefix,
               fileKin=TRUE,
               filePC=TRUE,
               filePhe = pheno_file,
               sep.phe = ',',
               priority="memory",
               maxLine=10000,
               out=geno_stem
)

effective_ratio <- EFFECTIVE_MARKERS/TOTAL_MARKERS
pheno <- read_tsv(str_c(geno_stem, '.phe'))
genotype <- attach.big.matrix(str_c(geno_stem, ".geno.desc"))
map <- read.table(str_c(geno_stem, ".geno.map"), header = TRUE)
Kinship <- attach.big.matrix(str_c(geno_stem, ".kin.desc"))
Covariates_PC <- bigmemory::as.matrix(attach.big.matrix(str_c(geno_stem, ".pc.desc")))

Sys.setenv(OMP_NUM_THREADS=1)

for(i in 2:ncol(pheno))
{
  imMVP<-MVP(phe = pheno[,c(1,i)], geno = genotype, map = map, K=Kinship,
               nPC.FarmCPU = 3, maxLoop = 10, method = "FarmCPU", p.threshold = (0.05/EFFECTIVE_MARKERS), 
               threshold = (0.05/effective_ratio),
               file.output = 'pmap.signal', ncpus=15)
}

