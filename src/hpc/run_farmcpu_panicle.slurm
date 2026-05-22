#!/usr/bin/bash
#SBATCH --array=1-10
#SBATCH --job-name=farmcpu
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --partition=schnablelab,jclarke,batch,guest
#SBATCH --output=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.out
#SBATCH --error=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/schnablelab/jdavis132/fieldLeafImaging
ml anaconda
conda activate GWAS
USE_OPENMP=1

phenotype_files=data/phenotype_files_farmcpu.txt
p=$(sed -n "${SLURM_ARRAY_TASK_ID}p;${SLURM_ARRAY_TASK_ID}q" ${phenotype_files})
g=human_scores.vcf
#p=scores_828_blues_score_average_${SLURM_ARRAY_TASK_ID}.csv

cd /scratch/
mkdir paniclefarmcpu_${SLURM_ARRAY_TASK_ID}
cd paniclefarmcpu_${SLURM_ARRAY_TASK_ID}
mkdir results

cp /work/schnablelab/jdavis132/fieldLeafImaging/data/farmcpu_phenotypes/${p} .
cp /work/schnablelab/jdavis132/fieldLeafImaging/data/${g} .
cp /work/schnablelab/jdavis132/PANICLE/scripts/run_GWAS.py .

ls -lht

pushd /scratch/paniclefarmcpu_${SLURM_ARRAY_TASK_ID}
python run_GWAS.py \
  --phenotype ${p} \
  --genotype ${g} \
  --phenotype-id-column genotype \
  --methods FarmCPU \
  --n-pcs 5 \
  --compute-effective-tests \
  --outputs significant_marker_pvalues \
  --outputdir results \
  --include-standard-errors

cp results/* /work/schnablelab/jdavis132/fieldLeafImaging/out/gwas/farmcpu/

