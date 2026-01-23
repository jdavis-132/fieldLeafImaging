library(rMVP)
library(readr)
library(data.table)
library(dplyr)
library(tidyverse)

MVP.FarmCPU_local <- function (phe, geno, map, CV = NULL, ind_idx = NULL, mrk_idx = NULL, 
          P = NULL, method.sub = "reward", method.sub.final = "reward", 
          method.bin = c("EMMA", "static", "FaST-LMM"), bin.size = c(5e+05, 
                                                                     5e+06, 5e+07), bin.selection = seq(10, 100, 10), memo = "MVP.FarmCPU", 
          Prior = NULL, ncpus = 2, maxLoop = 10, maxLine = 5000, threshold.output = 0.01, 
          converge = 1, iteration.output = FALSE, p.threshold = NA, 
          QTN.threshold = 0.01, bound = NULL, verbose = TRUE) 
{
  if (!is.big.matrix(geno)) 
    stop("genotype should be in 'big.matrix' format.")
  if (sum(is.na(phe[, 2])) != 0) 
    stop("NAs are not allowed in phenotype.")
  if (nrow(map) != ncol(geno) & nrow(map) != nrow(geno)) 
    stop("The number of markers in genotype and map doesn't match!")
  mrk_bycol <- ncol(geno) == nrow(map)
  if (is.null(ind_idx)) {
    if (nrow(phe) != ncol(geno) && nrow(phe) != nrow(geno)) 
      stop("number of individuals does not match in phenotype and genotype.")
    n <- ifelse(nrow(phe) == ncol(geno), ncol(geno), nrow(geno))
  }
  else {
    n <- length(ind_idx)
    if (nrow(phe) != n) 
      stop("number of individuals does not match in phenotype and genotype.")
  }
  echo = TRUE
  nm = nrow(map)
  if (!is.null(CV)) {
    CV = as.matrix(CV)
    if (nrow(CV) != n) 
      stop("number of individuals does not match in phenotype and fixed effects.")
    if (sum(is.na(CV)) != 0) 
      stop("NAs are not allowed in fixed effects.")
    CV.index <- apply(CV, 2, function(x) length(table(x)) > 
                        1)
    CV <- CV[, CV.index, drop = FALSE]
    npc = ncol(CV)
  }
  else {
    npc = 0
  }
  method.bin = match.arg(method.bin)
  map <- as.matrix(map)
  suppressWarnings(max.chr <- max(as.numeric(map[, 2]), na.rm = TRUE))
  if (is.infinite(max.chr)) 
    max.chr <- 0
  map.xy.index <- which(!as.numeric(map[, 2]) %in% c(0:max.chr))
  if (length(map.xy.index) != 0) {
    chr.xy <- unique(map[map.xy.index, 2])
    for (i in 1:length(chr.xy)) {
      map[map[, 2] == chr.xy[i], 2] <- max.chr + i
    }
  }
  map[, 1] = 1:nrow(map)
  suppressWarnings(map <- matrix(as.numeric(map), nrow(map)))
  if (sum(is.na(map[, 3]) != 0)) 
    stop("Non-digital characters or NAs are not allowed in map for FarmCPU")
  if (!is.na(p.threshold)) 
    QTN.threshold = max(p.threshold, QTN.threshold)
  name.of.trait = colnames(phe)[2]
  if (!is.null(memo)) 
    name.of.trait = paste(memo, ".", name.of.trait, sep = "")
  theLoop = 0
  theConverge = 0
  seqQTN.save = c(0)
  seqQTN.pre = c(-1)
  isDone = FALSE
  name.of.trait2 = name.of.trait
  while (!isDone) {
    theLoop = theLoop + 1
    # logging.log(paste("Current loop: ", theLoop, " out of maximum of ", 
    #                   maxLoop, sep = ""), "\n", verbose = verbose)
    spacer = "0"
    if (theLoop > 9) {
      spacer = ""
    }
    if (iteration.output) {
      name.of.trait2 = paste("Iteration_", spacer, theLoop, 
                             ".", name.of.trait, sep = "")
    }
    myPrior = FarmCPU.Prior(GM = map, P = P, Prior = Prior)
    if (theLoop <= 2) {
      myBin = FarmCPU.BIN(Y = phe[, c(1, 2)], GDP = geno, 
                          GDP_index = ind_idx, GM = map, CV = CV, P = myPrior, 
                          method = method.bin, b = bin.size, s = bin.selection, 
                          theLoop = theLoop, bound = bound, ncpus = ncpus, 
                          verbose = verbose)
    }
    else {
      myBin = FarmCPU.BIN(Y = phe[, c(1, 2)], GDP = geno, 
                          GDP_index = ind_idx, GM = map, CV = theCV, P = myPrior, 
                          method = method.bin, b = bin.size, s = bin.selection, 
                          theLoop = theLoop, ncpus = ncpus, verbose = verbose)
    }
    seqQTN = myBin$seqQTN
    if (theLoop == 2) {
      if (!is.na(p.threshold)) {
        if (min(myPrior, na.rm = TRUE) > p.threshold) {
          seqQTN = NULL
          # logging.log("Top snps have little effect, set seqQTN to NULL!", 
          #             "\n", verbose = verbose)
        }
      }
      else {
        if (min(myPrior, na.rm = TRUE) > 0.01/nm) {
          seqQTN = NULL
          # logging.log("Top snps have little effect, set seqQTN to NULL!", 
          #             "\n", verbose = verbose)
        }
      }
    }
    if (theLoop == 2 && is.null(seqQTN)) {
      P = myGLM$P[, ncol(myGLM$P)]
      P[P == 0] <- min(P[P != 0], na.rm = TRUE) * 0.01
      results = cbind(myGLM$B, myGLM$S, P)
      colnames(results) = c("effect", "se", "p")
      if (!is.null(mrk_idx)) 
        results <- results[mrk_idx, ]
      break
    }
    if (!any(is.null(seqQTN.save)) && theLoop > 1) {
      if (!(0 %in% seqQTN.save || -1 %in% seqQTN.save) && 
          !is.null(seqQTN)) {
        seqQTN <- union(seqQTN, seqQTN.save)
      }
    }
    if (theLoop != 1) {
      seqQTN.p = myPrior[seqQTN]
      if (theLoop == 2) {
        index.p = seqQTN.p < QTN.threshold
        seqQTN.p = seqQTN.p[index.p]
        seqQTN = seqQTN[index.p]
        seqQTN.p = seqQTN.p[!is.na(seqQTN)]
        seqQTN = seqQTN[!is.na(seqQTN)]
      }
      else {
        index.p = seqQTN.p < QTN.threshold
        index.p[seqQTN %in% seqQTN.save] = TRUE
        seqQTN.p = seqQTN.p[index.p]
        seqQTN = seqQTN[index.p]
        seqQTN.p = seqQTN.p[!is.na(seqQTN)]
        seqQTN = seqQTN[!is.na(seqQTN)]
      }
    }
    myRemove = FarmCPU.Remove(GDP = geno, GDP_index = ind_idx, 
                              GM = map, seqQTN = seqQTN, seqQTN.p = seqQTN.p, 
                              threshold = 0.7)
    seqQTN = myRemove$seqQTN
    theConverge = length(intersect(seqQTN, seqQTN.save))/length(union(seqQTN, 
                                                                      seqQTN.save))
    circle = (length(union(seqQTN, seqQTN.pre)) == length(intersect(seqQTN, 
                                                                    seqQTN.pre)))
    if (is.null(seqQTN.pre)) {
      circle = FALSE
    }
    else {
      if (seqQTN.pre[1] == 0) 
        circle = FALSE
      if (seqQTN.pre[1] == -1) 
        circle = FALSE
    }
    # logging.log("seqQTN:", "\n", verbose = verbose)
    if (is.null(seqQTN)) {
      # logging.log("NULL", "\n", verbose = verbose)
    }
    else {
      # logging.log(seqQTN, "\n", verbose = verbose)
    }
    if (theLoop == maxLoop) {
      # logging.log(paste("Total number of possible QTNs in the model is: ", 
      #                   length(seqQTN), sep = ""), "\n", verbose = verbose)
    }
    isDone = ((theLoop >= maxLoop) | (theConverge >= converge) | 
                circle)
    seqQTN.pre = seqQTN.save
    seqQTN.save = seqQTN
    rm(myBin)
    gc()
    theCV = CV
    if (!is.null(myRemove$bin)) {
      theCV = cbind(CV, myRemove$bin)
    }
    myGLM = FarmCPU.LM(y = as.matrix(phe[, 2]), GDP = geno, GDP_index = ind_idx, 
                       GDP_mrk_bycol = mrk_bycol, w = theCV, maxLine = maxLine, 
                       ncpus = ncpus, npc = npc, verbose = verbose)
    if (!is.null(seqQTN)) {
      if (ncol(myGLM$P) != (npc + length(seqQTN) + 1)) 
        stop("wrong dimensions.")
    }
    if (!isDone) {
      myGLM = FarmCPU.SUB(GM = map, GLM = myGLM, QTN = map[myRemove$seqQTN, 
                                                           , drop = FALSE], method = method.sub)
    }
    else {
      myGLM = FarmCPU.SUB(GM = map, GLM = myGLM, QTN = map[myRemove$seqQTN, 
                                                           , drop = FALSE], method = method.sub.final)
    }
    if (!is.null(mrk_idx)) {
      myGLM$P[-mrk_idx] = NA
      myGLM$B[-mrk_idx] = NA
      myGLM$S[-mrk_idx] = NA
    }
    P = myGLM$P[, ncol(myGLM$P)]
    P[P == 0] <- min(P[P != 0], na.rm = TRUE) * 0.01
    results = cbind(myGLM$B, myGLM$S, P)
    colnames(results) = c("effect", "se", "p")
    if (isDone && !is.null(mrk_idx)) 
      results <- results[mrk_idx, ]
  }
  return(results)
}

MVP_local <- function (phe, geno, map, K = NULL, nPC.GLM = NULL, nPC.MLM = NULL, 
          nPC.FarmCPU = NULL, CV.GLM = NULL, CV.MLM = NULL, CV.FarmCPU = NULL, 
          REML = NULL, maxLine = 10000, ncpus = detectCores(logical = FALSE), 
          vc.method = c("BRENT", "EMMA", "HE"), method = c("GLM", 
                                                           "MLM", "FarmCPU"), maf = NULL, p.threshold = NA, QTN.threshold = 0.01, 
          method.bin = "static", bin.size = c(5e+05, 5e+06, 5e+07), 
          bin.selection = seq(10, 100, 10), maxLoop = 10, permutation.threshold = FALSE, 
          permutation.rep = 100, memo = NULL, outpath = getwd(), col = c("#4197d8", 
                                                                         "#f8c120", "#413496", "#495226", "#d60b6f", "#e66519", 
                                                                         "#d581b7", "#83d3ad", "#7c162c", "#26755d"), file.output = TRUE, 
          file.type = "jpg", dpi = 300, threshold = 0.05, verbose = TRUE) 
{
  if (is.logical(file.output)) {
    if (file.output == TRUE) {
      file.output <- c("pmap", "pmap.signal", "plot", 
                       "log")
    }
    else if (file.output == FALSE) {
      file.output <- c()
    }
  }
  for (mt in method) {
    if (!mt %in% c("GLM", "MLM", "FarmCPU")) 
      stop("Unknow method: ", mt)
  }
  # logging.outpath <- NULL
  if ("log" %in% file.output) {
    # logging.outpath <- outpath
  }
  # logging.initialize("MVP", logging.outpath)
  MVP.Version(width = 65, verbose = verbose)
  time_start <- Sys.time()
  # logging.log("Start:", format(time_start, format = "%F %T %Z"), 
              "\n", verbose = verbose)
  # if ("log" %in% file.output) {
    # logging.log("The log has been output to the file:", 
                # get("logging.file", envir = package.env), "\n", 
                # verbose = verbose)
  }
  vc.method <- match.arg(vc.method)
  if (nrow(phe) != ncol(geno) & nrow(phe) != nrow(geno)) 
    stop("The number of individuals in phenotype and genotype doesn't match!")
  if (nrow(map) != ncol(geno) & nrow(map) != nrow(geno)) 
    stop("The number of markers in genotype and map doesn't match!")
  if (!is.big.matrix(geno)) 
    stop("genotype should be in 'big.matrix' format.")
  map <- as.data.frame(map)
  for (i in 1:ncol(map)) {
    if (is.factor(map[, i])) 
      map[, i] <- as.character.factor(map[, i])
  }
  na.index <- NULL
  if (!is.null(CV.GLM)) {
    CV.GLM <- as.matrix(CV.GLM)
    if (nrow(CV.GLM) != nrow(phe)) 
      stop("The number of individuals in covariates and phenotype doesn't match!")
    na.index <- c(na.index, which(is.na(CV.GLM), arr.ind = TRUE)[, 
                                                                 1])
  }
  if (!is.null(CV.MLM)) {
    CV.MLM <- as.matrix(CV.MLM)
    if (nrow(CV.MLM) != nrow(phe)) 
      stop("The number of individuals in covariates and phenotype doesn't match!")
    na.index <- c(na.index, which(is.na(CV.MLM), arr.ind = TRUE)[, 
                                                                 1])
  }
  if (!is.null(CV.FarmCPU)) {
    CV.FarmCPU <- as.matrix(CV.FarmCPU)
    if (nrow(CV.FarmCPU) != nrow(phe)) 
      stop("The number of individuals in covariates and phenotype doesn't match!")
    na.index <- c(na.index, which(is.na(CV.FarmCPU), arr.ind = TRUE)[, 
                                                                     1])
  }
  na.index <- unique(na.index)
  MrkByCol <- nrow(phe) == nrow(geno)
  m <- ifelse(MrkByCol, ncol(geno), nrow(geno))
  n <- nrow(phe)
  # logging.log(paste("Input data has", n, "individuals and", 
  #                   m, "markers"), "\n", verbose = verbose)
  # logging.log(paste("Markers are detected to be stored by", 
  #                   ifelse(MrkByCol, "column", "row")), "\n", verbose = verbose)
  # logging.log("Analyzed trait:", colnames(phe)[2], "\n", verbose = verbose)
  # logging.log("Number of threads used:", ncpus, "\n", verbose = verbose)
  hpclib <- grepl("mkl", sessionInfo()$LAPACK) | grepl("openblas", 
                                                       sessionInfo()$LAPACK) | eval(parse(text = "!inherits(try(Revo.version,silent=TRUE),'try-error')"))
  if (!hpclib) {
    # logging.log("No high performance math library detected! The computational efficiency would be greatly reduced\n", 
    #             verbose = verbose)
  }
  else {
    if (grepl("mkl", sessionInfo()$LAPACK) | eval(parse(text = "!inherits(try(Revo.version,silent=TRUE),'try-error')"))) {
      # logging.log("Math Kernel Library is detected, nice job!\n", 
      #             verbose = verbose)
    }
    else {
      # logging.log("OpenBLAS Library is detected, nice job!\n", 
      #             verbose = verbose)
    }
  }
  seqTaxa <- which(!is.na(phe[, 2]))
  if (length(na.index) != 0) 
    seqTaxa <- intersect(seqTaxa, c(1:n)[-na.index])
  if (length(seqTaxa) == 0) 
    stop("no effective individuals left due to missings")
  if (length(seqTaxa) == n) 
    seqTaxa <- NULL
  if (!is.null(seqTaxa)) {
    # logging.log("Total", n - length(seqTaxa), "individuals are removed due to missings", 
    #             "\n", verbose = verbose)
    phe = phe[seqTaxa, ]
    if (!is.null(K)) {
      K = K[seqTaxa, seqTaxa]
    }
    if (!is.null(CV.GLM)) {
      CV.GLM = CV.GLM[seqTaxa, , drop = FALSE]
    }
    if (!is.null(CV.MLM)) {
      CV.MLM = CV.MLM[seqTaxa, , drop = FALSE]
    }
    if (!is.null(CV.FarmCPU)) {
      CV.FarmCPU = CV.FarmCPU[seqTaxa, , drop = FALSE]
    }
    if (length(seqTaxa) < n * 0.8) {
      # logging.log("Re-build memory-mapping file for remaining individuals", 
      #             "\n", verbose = verbose)
      if (!MrkByCol) {
        geno <- deepcopy(geno, cols = seqTaxa)
      }
      else {
        geno <- deepcopy(geno, rows = seqTaxa)
      }
      seqTaxa <- NULL
    }
  }
  # logging.log("Calculate allele frequency...", "\n", verbose = verbose)
  marker_freq <- BigRowMean(geno@address, MrkByCol, threads = ncpus, 
                            geno_ind = seqTaxa)/2
  map$MAF <- ifelse(marker_freq > 0.5, 1 - marker_freq, marker_freq)
  geno_marker_index <- NULL
  map_sub <- map
  if (!is.null(maf)) {
    if (length(maf) != 1) 
      stop("maf should be a value")
    if (maf <= 0 || maf >= 0.5) 
      stop("maf should be at the range of 0-0.5")
    geno_marker_index <- which(map$MAF >= maf)
    if (length(geno_marker_index) == 0) 
      stop(paste("MAFs of all markers are smaller than the threshold", 
                 maf))
    if (length(geno_marker_index) == 1) 
      stop(paste("only 1 marker left on the given MAF threshold", 
                 maf))
    if (length(geno_marker_index) == m) 
      geno_marker_index <- NULL
  }
  if (!is.null(geno_marker_index)) {
    # logging.log("Total", m - length(geno_marker_index), 
    #             "markers are removed at MAF threshold", maf, "\n", 
    #             verbose = verbose)
    remmp <- (length(geno_marker_index) < m * 0.8)
    m <- length(geno_marker_index)
    map_sub <- map[geno_marker_index, ]
    marker_freq <- marker_freq[geno_marker_index]
    if (remmp) {
      if (!is.null(seqTaxa)) {
        # logging.log("Re-build memory-mapping file for remaining individuals and markers", 
        #             "\n", verbose = verbose)
        if (MrkByCol) {
          geno <- deepcopy(geno, rows = seqTaxa, cols = geno_marker_index)
        }
        else {
          geno <- deepcopy(geno, cols = seqTaxa, rows = geno_marker_index)
        }
        seqTaxa <- NULL
      }
      else {
        logging.log("Re-build memory-mapping file for remaining markers", 
                    "\n", verbose = verbose)
        if (MrkByCol) {
          geno <- deepcopy(geno, cols = geno_marker_index)
        }
        else {
          geno <- deepcopy(geno, rows = geno_marker_index)
        }
      }
      geno_marker_index <- NULL
    }
  }
  glm.results <- NULL
  mlm.results <- NULL
  farmcpu.results <- NULL
  glm.run <- "GLM" %in% method
  mlm.run <- "MLM" %in% method
  farmcpu.run <- "FarmCPU" %in% method
  nPC <- suppressWarnings(max(nPC.GLM, nPC.MLM, nPC.FarmCPU, 
                              na.rm = TRUE))
  if (nPC <= 0) {
    nPC <- NULL
  }
  else if (nPC < 3) {
    nPC <- 3
  }
  if (!is.null(K)) {
    K <- as.matrix(K)
  }
  if (!is.null(nPC) | "MLM" %in% method) {
    if (is.null(K)) {
      K <- MVP.K.VanRaden(M = geno, ind_idx = seqTaxa, 
                          mrk_idx = geno_marker_index, mrk_freq = marker_freq, 
                          mrk_bycol = MrkByCol, maxLine = maxLine, cpu = ncpus, 
                          verbose = verbose, checkNA = FALSE)
    }
    logging.log("Eigen Decomposition on GRM", "\n", verbose = verbose)
    eigenK <- eigen(K, symmetric = TRUE)
    if (!is.null(nPC)) {
      ipca <- eigenK$vectors[, 1:nPC]
      logging.log("Deriving PCs successfully", "\n", verbose = verbose)
    }
    if (("MLM" %in% method) & vc.method == "BRENT") {
      K <- NULL
      gc()
    }
    if (!"MLM" %in% method) {
      rm(eigenK)
      rm(K)
      gc()
    }
  }
  if (!is.null(nPC)) {
    if (glm.run) {
      if (!is.null(CV.GLM)) {
        logging.log("Number of provided covariates of GLM:", 
                    ncol(CV.GLM), "\n", verbose = verbose)
        if (!is.null(nPC.GLM)) {
          logging.log("Number of PCs included:", nPC.GLM, 
                      "\n", verbose = verbose)
          CV.GLM <- cbind(ipca[, 1:nPC.GLM], CV.GLM)
        }
      }
      else if (!is.null(nPC.GLM)) {
        logging.log("Number of PCs included in GLM:", 
                    nPC.GLM, "\n", verbose = verbose)
        CV.GLM <- ipca[, 1:nPC.GLM, drop = FALSE]
      }
    }
    if (mlm.run) {
      if (!is.null(CV.MLM)) {
        logging.log("Number of provided covariates of MLM:", 
                    ncol(CV.MLM), "\n", verbose = verbose)
        if (!is.null(nPC.MLM)) {
          logging.log("Number of PCs included:", nPC.MLM, 
                      "\n", verbose = verbose)
          CV.MLM <- cbind(ipca[, 1:nPC.MLM], CV.MLM)
        }
      }
      else if (!is.null(nPC.MLM)) {
        logging.log("Number of PCs included in MLM:", 
                    nPC.MLM, "\n", verbose = verbose)
        CV.MLM <- ipca[, 1:nPC.MLM, drop = FALSE]
      }
    }
    if (farmcpu.run) {
      if (!is.null(CV.FarmCPU)) {
        logging.log("Number of provided covariates of FarmCPU:", 
                    ncol(CV.FarmCPU), "\n", verbose = verbose)
        if (!is.null(nPC.FarmCPU)) {
          logging.log("Number of PCs included:", nPC.FarmCPU, 
                      "\n", verbose = verbose)
          CV.FarmCPU <- cbind(ipca[, 1:nPC.FarmCPU], 
                              CV.FarmCPU)
        }
      }
      else if (!is.null(nPC.FarmCPU)) {
        logging.log("Number of PCs included in FarmCPU:", 
                    nPC.FarmCPU, "\n", verbose = verbose)
        CV.FarmCPU <- ipca[, 1:nPC.FarmCPU]
      }
    }
  }
  else {
    if (glm.run) {
      if (!is.null(CV.GLM)) {
        logging.log("Number of provided covariates of GLM:", 
                    ncol(CV.GLM), "\n", verbose = verbose)
      }
    }
    if (mlm.run) {
      if (!is.null(CV.MLM)) {
        logging.log("Number of provided covariates of MLM:", 
                    ncol(CV.MLM), "\n", verbose = verbose)
      }
    }
    if (farmcpu.run) {
      if (!is.null(CV.FarmCPU)) {
        logging.log("Number of provided covariates of FarmCPU:", 
                    ncol(CV.FarmCPU), "\n", verbose = verbose)
      }
    }
  }
  logging.log("-------------------------GWAS Start-------------------------", 
              "\n", verbose = verbose)
  if (glm.run) {
    logging.log("General Linear Model (GLM) Start...", "\n", 
                verbose = verbose)
    glm.results <- MVP.GLM(phe = phe, geno = geno, CV = CV.GLM, 
                           ind_idx = seqTaxa, mrk_idx = geno_marker_index, 
                           mrk_bycol = MrkByCol, maxLine = maxLine, cpu = ncpus, 
                           verbose = verbose)
    gc()
    colnames(glm.results) <- c("Effect", "SE", paste(colnames(phe)[2], 
                                                     "GLM", sep = "."))
    z = glm.results[, 1]/glm.results[, 2]
    lambda = median(z^2, na.rm = TRUE)/qchisq(1/2, df = 1, 
                                              lower.tail = FALSE)
    logging.log("Genomic inflation factor (lambda):", round(lambda, 
                                                            4), "\n", verbose = verbose)
    if ("pmap" %in% file.output) {
      logging.log("Writing results to local file", "\n", 
                  verbose = verbose)
      write.csv(x = cbind(map_sub, glm.results), file = file.path(outpath, 
                                                                  paste(colnames(phe)[2], ".GLM.", memo, ifelse(is.null(memo), 
                                                                                                                "csv", ".csv"), sep = "")), row.names = FALSE)
    }
  }
  if (mlm.run) {
    logging.log("Mixed Linear Model (MLM) Start...", "\n", 
                verbose = verbose)
    mlm.results <- MVP.MLM(phe = phe, geno = geno, K = K, 
                           eigenK = eigenK, CV = CV.MLM, ind_idx = seqTaxa, 
                           mrk_idx = geno_marker_index, mrk_bycol = MrkByCol, 
                           maxLine = maxLine, cpu = ncpus, vc.method = vc.method, 
                           verbose = verbose)
    gc()
    colnames(mlm.results) <- c("Effect", "SE", paste(colnames(phe)[2], 
                                                     "MLM", sep = "."))
    z = mlm.results[, 1]/mlm.results[, 2]
    lambda = median(z^2, na.rm = TRUE)/qchisq(1/2, df = 1, 
                                              lower.tail = FALSE)
    logging.log("Genomic inflation factor (lambda):", round(lambda, 
                                                            4), "\n", verbose = verbose)
    if ("pmap" %in% file.output) {
      logging.log("Writing results to local file", "\n", 
                  verbose = verbose)
      write.csv(x = cbind(map_sub, mlm.results), file = file.path(outpath, 
                                                                  paste(colnames(phe)[2], ".MLM.", memo, ifelse(is.null(memo), 
                                                                                                                "csv", ".csv"), sep = "")), row.names = FALSE)
    }
  }
  if (farmcpu.run) {
    logging.log("FarmCPU Start...", "\n", verbose = verbose)
    farmcpu.results <- MVP.FarmCPU_local(phe = phe, geno = geno, 
                                   map = map[, 1:3], CV = CV.FarmCPU, ind_idx = seqTaxa, 
                                   mrk_idx = geno_marker_index, maxLine = maxLine, 
                                   ncpus = ncpus, memo = "MVP.FarmCPU", p.threshold = p.threshold, 
                                   QTN.threshold = QTN.threshold, method.bin = method.bin, 
                                   bin.size = bin.size, bin.selection = bin.selection, 
                                   maxLoop = maxLoop, verbose = verbose)
    colnames(farmcpu.results) <- c("Effect", "SE", paste(colnames(phe)[2], 
                                                         "FarmCPU", sep = "."))
    z = farmcpu.results[, 1]/farmcpu.results[, 2]
    lambda = median(z^2, na.rm = TRUE)/qchisq(1/2, df = 1, 
                                              lower.tail = FALSE)
    logging.log("Genomic inflation factor (lambda):", round(lambda, 
                                                            4), "\n", verbose = verbose)
    if ("pmap" %in% file.output) {
      logging.log("Writing results to local file", "\n", 
                  verbose = verbose)
      write.csv(x = cbind(map_sub, farmcpu.results), file = file.path(outpath, 
                                                                      paste(colnames(phe)[2], ".FarmCPU.", memo, ifelse(is.null(memo), 
                                                                                                                        "csv", ".csv"), sep = "")), row.names = FALSE)
    }
  }
  MVP.return <- list(map = map_sub, glm.results = glm.results, 
                     mlm.results = mlm.results, farmcpu.results = farmcpu.results)
  if (permutation.threshold) {
    i = 1
    for (i in 1:permutation.rep) {
      index = 1:nrow(phe)
      index.shuffle = sample(index, length(index), replace = FALSE)
      myY.shuffle = phe
      myY.shuffle[, 2] = myY.shuffle[index.shuffle, 2]
      myPermutation = MVP.GLM(phe = myY.shuffle[, c(1, 
                                                    2)], geno = geno, ind_idx = seqTaxa, mrk_idx = geno_marker_index, 
                              maxLine = maxLine, cpu = ncpus)
      pvalue = min(myPermutation[, 3], na.rm = TRUE)
      if (i == 1) {
        pvalue.final = pvalue
      }
      else {
        pvalue.final = c(pvalue.final, pvalue)
      }
    }
    permutation.cutoff = sort(pvalue.final)[ceiling(permutation.rep * 
                                                      0.05)]
    threshold = permutation.cutoff * m
  }
  logging.log(paste0("Significant level: ", formatC(threshold/m, 
                                                    format = "e", digits = 2)), "\n", verbose = verbose)
  if ("pmap.signal" %in% file.output) {
    if (glm.run) {
      index <- which(glm.results[, ncol(glm.results)] < 
                       threshold/m)
      if (length(index) != 0) {
        write.csv(x = cbind.data.frame(map_sub, glm.results)[index, 
        ], file = file.path(outpath, paste(colnames(phe)[2], 
                                           ".GLM_signals.", memo, ifelse(is.null(memo), 
                                                                         "csv", ".csv"), sep = "")), row.names = FALSE)
      }
    }
    if (mlm.run) {
      index <- which(mlm.results[, ncol(mlm.results)] < 
                       threshold/m)
      if (length(index) != 0) {
        write.csv(x = cbind.data.frame(map_sub, mlm.results)[index, 
        ], file = file.path(outpath, paste(colnames(phe)[2], 
                                           ".MLM_signals.", memo, ifelse(is.null(memo), 
                                                                         "csv", ".csv"), sep = "")), row.names = FALSE)
      }
    }
    if (farmcpu.run) {
      index <- which(farmcpu.results[, ncol(farmcpu.results)] < 
                       threshold/m)
      if (length(index) != 0) {
        write.csv(x = cbind.data.frame(map_sub, farmcpu.results)[index, 
        ], file = file.path(outpath, paste(colnames(phe)[2], 
                                           ".FarmCPU_signals.", memo, ifelse(is.null(memo), 
                                                                             "csv", ".csv"), sep = "")), row.names = FALSE)
      }
    }
  }
  if ("plot" %in% file.output) {
    logging.log("---------------------Visualization Start--------------------", 
                "\n", verbose = verbose)
    logging.log("Phenotype distribution Plotting", "\n", 
                verbose = verbose)
    MVP.Hist(memo = memo, outpath = outpath, file.output = TRUE, 
             phe = phe, file.type = file.type, col = col, dpi = dpi)
    plot3D <- FALSE
    if (!is.null(nPC)) {
      MVP.PCAplot(ipca[, 1:3], col = col, plot3D = plot3D, 
                  file.output = TRUE, file.type = file.type, outpath = outpath, 
                  memo = ifelse(is.null(memo), colnames(phe)[2], 
                                paste(colnames(phe)[2], memo, sep = ".")), 
                  dpi = dpi, )
    }
    MVP.Report(MVP.return, col = col, plot.type = c("c", 
                                                    "m", "q", "d"), file.output = TRUE, file.type = file.type, 
               outpath = outpath, memo = memo, chr.den.col = c("darkgreen", 
                                                               "yellow", "red"), dpi = dpi, threshold = threshold/m, 
    )
    if (sum(c(is.null(glm.results), is.null(mlm.results), 
              is.null(farmcpu.results))) < 2) {
      MVP.Report(MVP.return, col = col, plot.type = c("m", 
                                                      "q"), multracks = TRUE, file.output = TRUE, 
                 file.type = file.type, outpath = outpath, memo = memo, 
                 dpi = dpi, threshold = threshold/m)
    }
  }
  time_end <- Sys.time()
  if (length(file.output) > 0) {
    # logging.log("Results are stored at Working Directory:", 
                # outpath, "\n", verbose = verbose)
  }
  # logging.log("End:", format(time_end, format = "%F %T %Z"), 
  #             "\n", verbose = verbose)
  time_diff <- as.numeric(time_end) - as.numeric(time_start)
  h <- time_diff%/%3600
  m <- (time_diff%%3600)%/%60
  s <- ((time_diff%%3600)%%60)
  index <- which(c(h, m, s) != 0)
  num <- c(h, m, s)[index]
  num <- round(num, 0)
  char <- c("h", "m", "s")[index]
  logging.log("Total running time:", paste(num, char, sep = "", 
                                           collapse = ""), "\n", verbose = verbose)
  print_accomplished(width = 60, verbose = verbose)
  return(invisible(MVP.return))
}

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


{
  imMVP<-MVP(phe = as.matrix(phe1), geno = genotype, map = map, K=Kinship,
               nPC.FarmCPU = 3, maxLoop = 10, method = "FarmCPU", p.threshold = (0.05/EFFECTIVE_MARKERS), 
               threshold = (0.05/effective_ratio),
               file.output = 'pmap.signal', ncpus=15)
}


