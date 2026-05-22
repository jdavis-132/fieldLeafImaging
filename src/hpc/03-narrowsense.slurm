#!/usr/bin/bash
#SBATCH --array=1-125
#SBATCH --job-name=narrowsense
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=5G
#SBATCH --time=01:35:00
#SBATCH --partition=schnablelab,jclarke,batch,guest
#SBATCH --output=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.out
#SBATCH --error=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd $WORK/fieldLeafImaging

prefix1=$(sed -n "5p" data/phenotype_prefixes.txt)
#prefix2=$(sed -n "2p" data/phenotype_prefixes.txt)
#prefix3=$(sed -n "3p" data/phenotype_prefixes.txt)

m=$((SLURM_ARRAY_TASK_ID + 999))

./src/ldak6.1.beta --reml out/h2/${prefix1}_${m} --pheno data/${prefix1}_blues.txt --mpheno ${m} --grm data/${prefix1} --ignore-weights YES --power -1 --max-threads 16 --constrain YES
#./src/ldak6.1.beta --reml out/h2/${prefix2}_${SLURM_ARRAY_TASK_ID} --pheno data/${prefix2}_blues.txt --mpheno ${SLURM_ARRAY_TASK_ID} --grm data/${prefix2} --ignore-weights YES --power -1 --max-threads 16 --constrain YES
#./src/ldak6.1.beta --reml out/h2/${prefix3}_${SLURM_ARRAY_TASK_ID} --pheno data/${prefix3}_blues.txt --mpheno ${SLURM_ARRAY_TASK_ID} --grm data/${prefix3} --ignore-weights YES --power -1 --max-threads 16 --constrain YES
