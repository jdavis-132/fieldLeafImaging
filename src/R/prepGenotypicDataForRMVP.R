library(rMVP)
library(tidyverse)

MVP.Data(fileVCF=infile,
         fileKin=TRUE,
         filePC=TRUE,
         priority="memory",
         maxLine=10000,
         out=outPrefix
)
