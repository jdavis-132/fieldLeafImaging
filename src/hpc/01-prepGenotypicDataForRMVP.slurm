#!/usr/bin/bash
#SBATCH --array=1-3
#SBATCH --job-name=prepGenotypicData
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=0:30:00
#SBATCH --partition=schnablelab,jclarke,batch,guest
#SBATCH --output=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.out
#SBATCH --error=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

ml anaconda
conda activate GWAS

cd $WORK/fieldLeafImaging
prefix=$(sed -n "${SLURM_ARRAY_TASK_ID}p" data/phenotype_prefixes.txt)

R CMD BATCH -data/${prefix}.vcf -data/${prefix}.mvp src/prepGenotypicDataForRMVP.R log/prepGenotypicDataForRMVP.Rout
