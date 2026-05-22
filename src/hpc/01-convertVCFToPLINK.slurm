#!/usr/bin/bash
#SBATCH --array=5
#SBATCH --job-name=convertVCFToPLINK
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=1:05:00
#SBATCH --partition=schnablelab,jclarke,batch,guest
#SBATCH --output=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.out
#SBATCH --error=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

ml plink2
ml anaconda
conda activate GWAS

cd $WORK/fieldLeafImaging
prefix=$(sed -n "${SLURM_ARRAY_TASK_ID}p" data/phenotype_prefixes.txt)

plink2 --vcf data/${prefix}.vcf --make-bed --out data/${prefix} --const-fid
