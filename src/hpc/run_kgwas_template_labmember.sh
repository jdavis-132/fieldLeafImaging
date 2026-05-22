#!/bin/bash
#SBATCH --array=2
#SBATCH --job-name=kgwas_custom
#SBATCH --partition=schnablelab,jclarke,batch,guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work/schnablelab/jdavis132/fieldLeafImaging/log/kgwas_%a.out
#SBATCH --error=/work/schnablelab/jdavis132/fieldLeafImaging/log/kgwas_%a.err

# ===================================================================
# Template: Run k-mer GWAS using pre-computed sorghum k-mer table
# ===================================================================
#
# INSTRUCTIONS:
# 1. Copy this script to your own directory
# 2. Edit the two variables below:
#    - PHENOTYPE_FILE: path to your phenotype file
#    - TRAIT_NAME: a short name for your trait (no spaces)
# 3. Submit: sbatch run_kgwas_template_labmember.sh
#
# PHENOTYPE FILE FORMAT (tab-separated, with header):
#   accession_id	phenotype_value
#   BTx643	62
#   BTx645	64.5
#   PI153844	67.5
#
# Sample names must match those in the k-mer table (937 sorghum accessions).
# See: /work/schnablelab/waqarali/kgwas_sorghum_merged/results/kmers_table/kmers_table.names
# ===================================================================

# ---- EDIT THESE TWO LINES ----
PHENOTYPE_FILE_LIST="/work/schnablelab/jdavis132/fieldLeafImaging/data/blues/high_fi_kgwas_pheno_files.txt"
PHENOTYPE_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${PHENOTYPE_FILE_LIST})
TRAIT_NAME=$(echo "${PHENOTYPE_FILE}" | sed 's|sam3_rs_high_fi_blues_||g' | sed 's|\.tsv||g')
PHENOTYPE_FILE="/work/schnablelab/jdavis132/fieldLeafImaging/data/blues/${PHENOTYPE_FILE}"
# ---- END EDITS ----

# Paths to pre-computed resources (DO NOT CHANGE)
KGWAS_DIR="/work/schnablelab/waqarali/kgwas_sorghum_merged"
KMERS_TABLE="${KGWAS_DIR}/results/kmers_table/kmers_table"
KGWAS_SCRIPT="${KGWAS_DIR}/scripts/external/kmers_gwas/kmers_gwas.py"
CONDA_ENV="/work/schnablelab/waqarali/shared_kgwas_conda/7ac3439d6108de8abf6b55e209fc8710_"

# Output directory
OUTDIR="/work/schnablelab/jdavis132/fieldLeafImaging/out/gwas/kgwas/embeddings/${TRAIT_NAME}_1M"

# Activate conda environment (has Python 2 + dependencies)
source ~/.bashrc
conda activate ${CONDA_ENV}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "============================================"
echo "k-mer GWAS: ${TRAIT_NAME}"
echo "Phenotype: ${PHENOTYPE_FILE}"
echo "Output: ${OUTDIR}"
echo "Start: $(date)"
echo "============================================"

#mkdir -p ${OUTDIR}

python2 ${KGWAS_SCRIPT} \
    --min_data_points 30 \
    --pheno ${PHENOTYPE_FILE} \
    --kmers_table ${KMERS_TABLE} \
    --kmers_number 1000000 \
    --permutations 100 \
    --maf 0.05 --mac 5 \
    -l 31 -p 12 \
    --outdir ${OUTDIR}

echo "============================================"
echo "Done: $(date)"
echo "============================================"
echo ""
echo "Results are in: ${OUTDIR}"
echo "Key output files:"
echo "  - ${OUTDIR}/kmers/output/phenotype_value.assoc.txt.gz  (all 1M k-mer associations)"
echo "  - ${OUTDIR}/kmers/output/pass_threshold_5per  (significant k-mers at 5% threshold)"
echo ""
echo "To get the 95th permutation threshold:"
echo "  Look at: ${OUTDIR}/kmers/output/best_pvals"
echo "  Sort the 100 permutation -log10(p) values descending, pick rank 95"
