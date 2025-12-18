library(rMVP)
library(data.table)
library(tidyverse)

args <- commandArgs(trailingOnly = FALSE)
pheno_file <- str_remove(args[length(args)-3], fixed('-'))
prefix <- str_remove(args[length(args)-2], fixed('-'))
TOTAL_MARKERS <- as.numeric(str_remove(args[length(args)-1], fixed('-')))
EFFECTIVE_MARKERS <- as.numeric(str_remove(args[length(args)], fixed('-'))) 
geno_stem <- str_c('data/', prefix, '.mvp')

rMVP::MVP.Data(fileVCF=str_c('data/', prefix,'.vcf'),
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
  imMVP<-MVP(phe = pheno[,c(1,i)], geno = genotype, map = map, K=Kinship, CV.MLM=Covariates_PC,
               nPC.MLM = 3, maxLine=10000, method = "MLM",
               file.output = 'pmap', vc.method="BRENT", ncpus=15)
}

