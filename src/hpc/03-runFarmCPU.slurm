#!/usr/bin/bash
#SBATCH --array=1
#SBATCH --job-name=farmcpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=5G
#SBATCH --time=16:00:00
#SBATCH --partition=schnablelab,jclarke,batch,guest
#SBATCH --output=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.out
#SBATCH --error=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/schnablelab/jdavis132/fieldLeafImaging
ml anaconda
conda activate GWAS
USE_OPENMP=1

pheno_file_list="data/mlm_phenofiles.txt"
#genostem_list="data/mlm_genostems.txt"
#totalmarker_list="data/mlm_totalmarkers.txt"
#effectivemarker_list="data/mlm_effectivemarkers.txt"

p=`sed -n "${SLURM_ARRAY_TASK_ID}"p ${pheno_file_list} |  awk '{print $1}'`
#g=`sed -n "${SLURM_ARRAY_TASK_ID}"p ${genostem_list} |  awk '{print $1}'`
#total_markers=`sed -n "${SLURM_ARRAY_TASK_ID}"p ${totalmarker_list} |  awk '{print $1}'`
#effective_markers=`sed -n "${SLURM_ARRAY_TASK_ID}"p ${effectivemarker_list} |  awk '{print $1}'`

# set static value
g=sam3
total_markers=167175
effective_markers=49957

mkdir out/gwas/farmcpu/${g}

cd /scratch/
mkdir farmcpu_${SLURM_ARRAY_TASK_ID}
cd farmcpu_${SLURM_ARRAY_TASK_ID}

cp /work/schnablelab/jdavis132/fieldLeafImaging/data/${p} .
cp /work/schnablelab/jdavis132/fieldLeafImaging/data/${g}.* .
cp /work/schnablelab/jdavis132/fieldLeafImaging/src/FarmCPU.R .

ls -lht

pushd /scratch/farmcpu_${SLURM_ARRAY_TASK_ID}
R CMD BATCH -${p} -${g} -${total_markers} -${effective_markers} FarmCPU.R FarmCPU_${SLURM_ARRAY_TASK_ID}.Rout

cp *signals.csv /work/schnablelab/jdavis132/fieldLeafImaging/out/gwas/farmcpu/${g}/
cp *.Rout /work/schnablelab/jdavis132/fieldLeafImaging/log/

