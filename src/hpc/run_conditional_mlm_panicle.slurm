#!/usr/bin/bash
#SBATCH --array=1-2
#SBATCH --job-name=conditional_mlm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=schnablelab,jclarke,batch,guest
#SBATCH --output=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.out
#SBATCH --error=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

set -euo pipefail

cd /work/schnablelab/jdavis132/fieldLeafImaging
ml anaconda
conda activate GWAS

datasets=(condition_exg condition_human_exg)
dataset="${datasets[$((SLURM_ARRAY_TASK_ID - 1))]}"
base="output/conditional_panicle/${dataset}"

phenotype="${base}/${dataset}_embedding_phenotypes.csv"
covariates="${base}/${dataset}_severity_covariates.csv"
traits_file="${base}/${dataset}_traits.txt"
genotype="${base}/${dataset}.vcf.gz"
outdir="out/gwas/conditional_mlm/${dataset}"

if [[ ! -f "${genotype}" ]]; then
  echo "Missing filtered VCF: ${genotype}"
  echo "Run output/conditional_panicle/filter_vcf_commands.sh before submitting this job."
  exit 1
fi

traits="$(paste -sd, "${traits_file}")"
covariate_columns="$(head -n 1 "${covariates}" | tr ',' '\n' | grep -v '^genotype$' | paste -sd,)"

mkdir -p "${outdir}"

python src/hpc/run_GWAS.py \
  --phenotype "${phenotype}" \
  --genotype "${genotype}" \
  --format vcf \
  --phenotype-id-column genotype \
  --covariates "${covariates}" \
  --covariate-id-column genotype \
  --covariate-columns "${covariate_columns}" \
  --traits "${traits}" \
  --methods MLM \
  --n-pcs 5 \
  --compute-effective-tests \
  --outputs all_marker_pvalues significant_marker_pvalues \
  --outputdir "${outdir}" \
  --include-standard-errors
