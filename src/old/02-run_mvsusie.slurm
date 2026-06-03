#!/usr/bin/bash
#SBATCH --array=2-14,16-21
#SBATCH --job-name=mvsusie
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=00:30:00
#SBATCH --partition=schnablelab,jclarke,batch,guest
#SBATCH --output=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.out
#SBATCH --error=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/schnablelab/jdavis132/fieldLeafImaging
ml anaconda
conda activate mvsusie
#source ../mvsusie-py/.venv/bin/activate
export PYTHONPATH=../mvsusie-py/src
USE_OPENMP=1

peak_file=data/peaks_all_traits.tsv
chr=$(awk -F'\t' -v n=${SLURM_ARRAY_TASK_ID} -v i=2 'NR==n {print $i}' ${peak_file})
start=$(awk -F'\t' -v n=${SLURM_ARRAY_TASK_ID} -v i=10 'NR==n {print $i}' ${peak_file})
end=$(awk -F'\t' -v n=${SLURM_ARRAY_TASK_ID} -v i=11 'NR==n {print $i}' ${peak_file})
traits=$(awk -F'\t' -v n=${SLURM_ARRAY_TASK_ID} -v i=12 'NR==n {print $i}' ${peak_file})

python /work/schnablelab/jdavis132/mvsusie-py/scripts/run_mvsusie.py --gwas-results /work/schnablelab/jdavis132/fieldLeafImaging/out/gwas/mlm_reseq/all_traits --ld out/mvsusie --out-dir out/mvsusie/ --chr ${chr} --start ${start} --end ${end} --traits ${traits}
