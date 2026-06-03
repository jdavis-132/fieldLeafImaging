#!/bin/bash
#SBATCH --job-name=kgwas_postproc
#SBATCH --partition=jclarke,schnablelab,guest,batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=kgwas_postproc_%j.out
#SBATCH --error=kgwas_postproc_%j.err

# ===================================================================
# Template: Post-process k-mer GWAS results
#   1. Extract all 1M k-mers + p-values
#   2. Align to Sorghum v5.0 reference with bowtie2
#   3. Filter unique + perfect matches
#   4. Calculate 95th permutation threshold
#   5. Generate Manhattan plot
# ===================================================================
#
# INSTRUCTIONS:
# 1. Copy this script to your own directory
# 2. Edit TRAIT_NAME below (must match what you used in the GWAS step)
# 3. Submit: sbatch postprocess_kgwas_template_labmember.sh
#
# PREREQUISITE: The GWAS step (run_kgwas_template_labmember.sh) must
# have finished successfully.
# ===================================================================

# ---- EDIT THIS LINE ----
TRAIT_NAME="YourTraitName"
# ---- END EDITS ----

module load bowtie/2.5
module load R/4.5

KGWAS_DIR="/work/schnablelab/waqarali/kgwas_sorghum_merged"
BASE="${KGWAS_DIR}/results/kmers_gwas/${TRAIT_NAME}_1M"
ALIGNED="${BASE}/aligned_1M"
BOWTIE2_INDEX="${KGWAS_DIR}/config/data/Sbicolor_730_v5.0"

# Check that GWAS output exists
if [ ! -f "${BASE}/kmers/output/phenotype_value.assoc.txt.gz" ]; then
    echo "ERROR: GWAS output not found at ${BASE}/kmers/output/phenotype_value.assoc.txt.gz"
    echo "Make sure the GWAS step completed successfully first."
    exit 1
fi

mkdir -p ${ALIGNED}

# ---- Step 1: Calculate 95th permutation threshold ----
echo "=== Post-processing: ${TRAIT_NAME} ==="
echo "Date: $(date)"
echo ""
echo "[Step 1] Calculating 95th permutation threshold..."

BEST_PVALS="${BASE}/kmers/output/best_pvals"
if [ -f "${BEST_PVALS}" ]; then
    # Extract the 100 permutation -log10(p) values, sort descending, pick rank 95
    THRESHOLD=$(awk 'NR > 1 {print -log($2)/log(10)}' ${BEST_PVALS} | sort -rn | awk 'NR==95{print}')
    echo "  95th permutation threshold: ${THRESHOLD}"
else
    echo "WARNING: best_pvals file not found. Cannot calculate threshold."
    echo "  Expected at: ${BEST_PVALS}"
    THRESHOLD="NA"
fi

# ---- Step 2: Extract all 1M k-mer sequences + p-values ----
echo "[Step 2] Extracting 1M k-mers from GWAS output..."
zcat ${BASE}/kmers/output/phenotype_value.assoc.txt.gz | \
    awk 'NR > 1 {split($2, a, "_"); kmer=""; for(i=1;i<length(a);i++) { if(i>1) kmer=kmer"_"; kmer=kmer""a[i] }; print kmer"\t"$9}' \
    > ${ALIGNED}/all_kmer_pvalues.txt
N_KMERS=$(wc -l < ${ALIGNED}/all_kmer_pvalues.txt)
echo "  Total k-mers: ${N_KMERS}"

# ---- Step 3: Create FASTA for alignment ----
echo "[Step 3] Creating FASTA..."
awk '{print ">"$1"\n"substr($1,1,31)}' ${ALIGNED}/all_kmer_pvalues.txt > ${ALIGNED}/all_kmers.fa

# ---- Step 4: Align with bowtie2 ----
echo "[Step 4] Aligning with bowtie2..."
bowtie2 --very-sensitive --end-to-end --no-unal -k 10 \
    -f -x ${BOWTIE2_INDEX} \
    -U ${ALIGNED}/all_kmers.fa \
    -S ${ALIGNED}/aligned_1M.sam \
    --threads 8 2>${ALIGNED}/bowtie2.log

echo "  Alignment stats:"
cat ${ALIGNED}/bowtie2.log

# ---- Step 5: Filter unique (MAPQ>0) + perfect (NM:i:0) matches ----
echo "[Step 5] Filtering unique (MAPQ>0) + perfect (NM:i:0) matches..."
awk '$5 > 0 && /NM:i:0/' ${ALIGNED}/aligned_1M.sam > ${ALIGNED}/unique_perfect.sam
N_UNIQUE=$(wc -l < ${ALIGNED}/unique_perfect.sam)
echo "  Unique perfect matches: ${N_UNIQUE}"

# ---- Step 6: Create Manhattan data CSV ----
echo "[Step 6] Creating Manhattan CSV..."
echo "kmer_id,chromosome,chr_num,position,p_lrt,log10p" > ${ALIGNED}/manhattan_data_1M.csv

awk '
BEGIN {OFS=","}
NR==FNR {pval[$1]=$2; next}
{
    kmer=$1; chr=$3; pos=$4;
    if (kmer in pval) {
        split(chr, c, "r");
        chr_num = sprintf("%02d", c[2]);
        p = pval[kmer];
        logp = -log(p)/log(10);
        print kmer, chr, chr_num, pos, p, logp
    }
}' ${ALIGNED}/all_kmer_pvalues.txt ${ALIGNED}/unique_perfect.sam >> ${ALIGNED}/manhattan_data_1M.csv

N_CSV=$(tail -n +2 ${ALIGNED}/manhattan_data_1M.csv | wc -l)
echo "  Manhattan CSV entries: ${N_CSV}"

# ---- Step 7: Count above threshold ----
echo "[Step 7] Counting k-mers above threshold..."
if [ "${THRESHOLD}" != "NA" ]; then
    N_ABOVE=$(awk -F',' -v t="${THRESHOLD}" 'NR>1 && $6+0 >= t+0' ${ALIGNED}/manhattan_data_1M.csv | wc -l)
    echo "  K-mers above threshold: ${N_ABOVE}"

    if [ "$N_ABOVE" -gt 0 ]; then
        echo "  Per-chromosome:"
        awk -F',' -v t="${THRESHOLD}" 'NR>1 && $6+0 >= t+0 {print $3}' ${ALIGNED}/manhattan_data_1M.csv | sort | uniq -c | sort -rn | awk '{printf "    Chr%d: %d k-mers\n", $2, $1}'
    fi
else
    echo "  Skipped (no threshold calculated)"
fi

# ---- Step 8: Generate Manhattan plot ----
echo "[Step 8] Generating Manhattan plot..."

# Save threshold to file for reference
echo "${THRESHOLD}" > ${ALIGNED}/threshold_95th.txt

# Create R script for this trait
cat > ${ALIGNED}/plot_manhattan.R << 'RSCRIPT'
library(ggplot2)
library(dplyr)
library(data.table)

args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[1]
threshold_95 <- as.numeric(args[2])
trait_name <- args[3]
out_dir <- args[4]

chr_colors <- rep(c("#B8973B", "#4A6FA5"), length.out = 10)

cat("Plotting kGWAS", trait_name, "\n")

data <- fread(csv_path)
data$CHR <- as.integer(data$chr_num)
data <- data[!is.na(data$CHR) & data$CHR >= 1 & data$CHR <= 10, ]
data <- data[order(data$CHR, data$position), ]

chr_info <- data %>%
  group_by(CHR) %>%
  summarise(chr_len = max(position)) %>%
  mutate(tot = cumsum(as.numeric(chr_len)) - chr_len)

data <- merge(data, chr_info[, c("CHR", "tot")], by = "CHR")
data$BPcum <- data$position + data$tot

axis_df <- data %>%
  group_by(CHR) %>%
  summarise(center = (max(BPcum) + min(BPcum)) / 2)

# Auto-scale y-axis: round up max to nearest even number
raw_max <- max(data$log10p, na.rm = TRUE)
ymax <- ceiling(raw_max / 2) * 2
if (ymax < threshold_95 + 1) ymax <- ceiling(threshold_95 / 2) * 2 + 2
cat("  Max -log10(p):", raw_max, "-> ymax:", ymax, "\n")

data$log10p_plot <- pmin(data$log10p, ymax)

p <- ggplot(data, aes(x = BPcum, y = log10p_plot, color = factor(CHR))) +
  geom_point(alpha = 0.7, size = 0.8) +
  scale_color_manual(values = chr_colors, guide = "none") +
  scale_x_continuous(labels = axis_df$CHR, breaks = axis_df$center, expand = c(0.01, 0)) +
  scale_y_continuous(limits = c(0, ymax), breaks = seq(0, ymax, by = 2), expand = c(0.02, 0)) +
  geom_hline(yintercept = threshold_95, color = "red", linetype = "dashed", linewidth = 0.5) +
  labs(x = "Chromosome", y = expression(-log[10](p))) +
  theme_classic() +
  theme(
    axis.text = element_text(size = 10, color = "black"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.line = element_line(color = "black", linewidth = 0.6),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white", color = NA)
  )

out_file <- file.path(out_dir, paste0("sorghum_kgwas_", trait_name, ".png"))
ggsave(out_file, p, width = 12, height = 5, dpi = 300, bg = "white")
cat("Saved:", out_file, "\n")

n_above <- sum(data$log10p >= threshold_95)
cat("K-mers above threshold:", n_above, "\n")
RSCRIPT

Rscript ${ALIGNED}/plot_manhattan.R \
    ${ALIGNED}/manhattan_data_1M.csv \
    ${THRESHOLD} \
    ${TRAIT_NAME} \
    ${ALIGNED}

echo ""
echo "=== DONE ==="
echo "Date: $(date)"
echo ""
echo "Output files:"
echo "  Manhattan CSV:  ${ALIGNED}/manhattan_data_1M.csv"
echo "  Manhattan plot: ${ALIGNED}/sorghum_kgwas_${TRAIT_NAME}.png"
echo "  Bowtie2 log:    ${ALIGNED}/bowtie2.log"
echo "  Threshold:      ${ALIGNED}/threshold_95th.txt"
