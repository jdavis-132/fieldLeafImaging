#!/usr/bin/bash
#SBATCH --array=3-21
#SBATCH --job-name=computeLDRegion
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
USE_OPENMP=1

peak_file=data/peaks_all_traits.csv
chr=$(awk -F',' -v n=${SLURM_ARRAY_TASK_ID} -v i=2 'NR==n {print $i}' ${peak_file})
start=$(awk -F',' -v n=${SLURM_ARRAY_TASK_ID} -v i=10 'NR==n {print $i}' ${peak_file})
end=$(awk -F',' -v n=${SLURM_ARRAY_TASK_ID} -v i=11 'NR==n {print $i}' ${peak_file})
vcf=data/sam3_rs.vcf

echo ${start}
python /work/schnablelab/jdavis132/mvsusie-py/scripts/compute_LD_matrix.py ${vcf} --out-dir out/mvsusie/ --chr ${chr} --start ${start} --end ${end}
