#!/usr/bin/bash
#SBATCH --array=1-45
#SBATCH --job-name=mlm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=5G
#SBATCH --time=01:00:00
#SBATCH --partition=schnablelab,jclarke,batch,guest
#SBATCH --output=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.out
#SBATCH --error=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/schnablelab/jdavis132/fieldLeafImaging
ml anaconda
conda activate GWAS
USE_OPENMP=1

# access from lines in files
#pheno_file_list="data/mlm_phenofiles.txt"
#genostem_list="data/mlm_genostems.txt"
#totalmarker_list="data/mlm_totalmarkers.txt"
#effectivemarker_list="data/mlm_effectivemarkers.txt"

#p=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${pheno_file_list})
#g=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${genostem_list})
#total_markers=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${totalmarker_list})
#effective_markers=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${effectivmarker_list})

# set static values if using
p=sam3_blues_${SLURM_ARRAY_TASK_ID}.csv
g=sam3
total_markers=167175
effective_markers=49957

mkdir out/gwas/${g}

cd /scratch/
mkdir mlm_${SLURM_ARRAY_TASK_ID}
cd mlm_${SLURM_ARRAY_TASK_ID}

cp /work/schnablelab/jdavis132/fieldLeafImaging/data/${p} .
cp /work/schnablelab/jdavis132/fieldLeafImaging/data/${g}.* .
cp /work/schnablelab/jdavis132/fieldLeafImaging/src/MLMGWAS.R .

ls -lht

pushd /scratch/mlm_${SLURM_ARRAY_TASK_ID}
R CMD BATCH -${p} -${g} -${total_markers} -${effective_markers} MLMGWAS.R mlm_${SLURM_ARRAY_TASK_ID}.Rout

cp *MLM*.csv /work/schnablelab/jdavis132/fieldLeafImaging/out/gwas/${g}/
cp *.Rout /work/schnablelab/jdavis132/fieldLeafImaging/log/
