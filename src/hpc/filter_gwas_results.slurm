#!/usr/bin/bash
#SBATCH --array=1
#SBATCH --job-name=filter_mlm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=12:30:00
#SBATCH --partition=schnablelab,jclarke,batch,guest
#SBATCH --output=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.out
#SBATCH --error=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/schnablelab/jdavis132/fieldLeafImaging
ml anaconda
conda activate GWAS

python src/filter_gwas_results.py 'out/gwas/mlm/GWAS_embedding_*_all_results.csv' out/gwas/mlm/most_significant_associations_0.0001.csv
