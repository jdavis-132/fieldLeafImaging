#!/usr/bin/bash
#SBATCH --job-name=summariseSignals
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=06:30:00
#SBATCH --partition=schnablelab,jclarke,batch,guest
#SBATCH --output=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.out
#SBATCH --error=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/schnablelab/jdavis132/fieldLeafImaging
ml anaconda
conda activate GWAS

Rscript src/summariseSignals_PANICLE.R -out/gwas/mlm_panicle/sam3/*_all_results.csv -6.31e-07 -out/gwas/mlm_panicle/sam3/signals.csv
#Rscript src/summariseSignals_PANICLE.R -out/gwas/mlm_reseq/sam3/*_all_results.csv -6.31e-07 -out/gwas/mlm_reseq/sam3/signals.csv
