#!/usr/bin/bash
#SBATCH --array=1
#SBATCH --job-name=aggregate_farmcpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=04:30:00
#SBATCH --partition=schnablelab,jclarke,batch,guest
#SBATCH --output=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.out
#SBATCH --error=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/schnablelab/jdavis132/fieldLeafImaging
ml anaconda
conda activate GWAS

#python src/aggregate_farmcpu_results.py 'out/gwas/farmcpu/GWAS_embedding_*_significant.csv' out/gwas/farmcpu/all_farmcpu_hits.csv
python src/aggregate_farmcpu_results.py 'out/gwas/farmcpu/GWAS_pctd*_significant.csv' out/gwas/farmcpu/pctd_farmcpu_hits.csv
python src/aggregate_farmcpu_results.py 'out/gwas/farmcpu/GWAS_score*_significant.csv' out/gwas/farmcpu/human_scores_farmcpu_hits.csv
