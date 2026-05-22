#!/usr/bin/bash
#SBATCH --job-name=calculateSNPKinship
#SBATCH --array=5
#SBATCH --mem-per-cpu=6G
#SBATCH --time=1:35:00
#SBATCH --partition=schnablelab,jclarke,batch,guest
#SBATCH --output=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.out
#SBATCH --error=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd $WORK/fieldLeafImaging
prefix=$(sed -n "${SLURM_ARRAY_TASK_ID}p" data/phenotype_prefixes.txt)

./src/ldak6.1.beta --calc-kins-direct data/${prefix} --bfile data/${prefix} --ignore-weights YES --power -1
