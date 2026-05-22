#!/usr/bin/bash
#SBATCH --array=1-21
#SBATCH --job-name=mlm
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
USE_OPENMP=1
#vcf_files=data/prefix_vcf_filt.txt
#phenotype_files=data/phenotype_files_mlm.txt
#p=$(sed -n "${SLURM_ARRAY_TASK_ID}p;${SLURM_ARRAY_TASK_ID}q" ${phenotype_files})
p=sam3_blues_${SLURM_ARRAY_TASK_ID}.csv
#g=$(sed -n "${SLURM_ARRAY_TASK_ID}p;${SLURM_ARRAY_TASK_ID}q" ${vcf_files})
#g=${g}.vcf
g=sam3.vcf
prefix="${p%%_blues*}"

cd /scratch/
mkdir paniclemlm_${SLURM_ARRAY_TASK_ID}
cd paniclemlm_${SLURM_ARRAY_TASK_ID}
mkdir results

cp /work/schnablelab/jdavis132/fieldLeafImaging/data/blues/${p} .
cp /work/schnablelab/jdavis132/fieldLeafImaging/data/${g} .
cp /work/schnablelab/jdavis132/fieldLeafImaging/src/run_GWAS.py .
ls -lht

pushd /scratch/paniclemlm_${SLURM_ARRAY_TASK_ID}

python run_GWAS.py \
  --phenotype ${p} \
  --genotype ${g} \
  --phenotype-id-column genotype \
  --methods MLM \
  --n-pcs 5 \
  --compute-effective-tests \
  --outputs all_marker_pvalues \
  --outputdir results \
  --include-standard-errors
cp results/* /work/schnablelab/jdavis132/fieldLeafImaging/out/gwas/mlm/

