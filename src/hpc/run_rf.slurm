#!/usr/bin/bash
#SBATCH --array=2-66
#SBATCH --job-name=RF
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=35:30:00
#SBATCH --partition=schnablelab,jclarke,batch,guest
#SBATCH --output=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.out
#SBATCH --error=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd $WORK/fieldLeafImaging
ml anaconda
conda activate random_forest

model=$(sed -n "${SLURM_ARRAY_TASK_ID}p" data/model_list.txt)
predictor_prefix=$(sed -n "${SLURM_ARRAY_TASK_ID}p" data/predictor_prefix_list.txt)
label=$(sed -n "${SLURM_ARRAY_TASK_ID}p" data/label_file_list.txt)
label_col=$(sed -n "${SLURM_ARRAY_TASK_ID}p" data/label_col_list.txt)
echo ${model}
echo ${predictor_prefix}
echo ${label}

python src/main5CV.py data/rf_20260507/${model}_${label}_rf.csv --predictor_prefix ${predictor_prefix} --label ${label_col} --output_prefix out/rf_20260507/${model}_${label}_${predictor_prefix}_
