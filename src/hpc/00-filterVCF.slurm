#!/usr/bin/bash
#SBATCH --array=1
#SBATCH --job-name=filterVCF
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=02:30:00
#SBATCH --partition=schnablelab,jclarke,batch,guest
#SBATCH --output=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.out
#SBATCH --error=/work/schnablelab/jdavis132/fieldLeafImaging/log/%x_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

ml vcftools
cd $WORK/fieldLeafImaging

#prefix=$(sed -n "${SLURM_ARRAY_TASK_ID}p" data/prefix_vcf_filt.txt)
prefix=human_scores

vcftools --gzvcf /mnt/nrdstor/schnablelab/jschnable/filtered_vcfs/sorghum_888genotypes_filtered_v2.vcf.gz --out data/${prefix} --keep data/${prefix}_genotypes_keep.txt --recode
python src/HomoMAFfiltering.py data/${prefix}.recode.vcf data/${prefix}.vcf

#vcftools --vcf data/SAP_BQSR_imputed_allchr_sns_indels_renamed.vcf --out data/sam3_reseq --keep data/sam3_rs_genotypes_keep.txt --recode
#python src/HomoMAFfiltering.py data/sam3_reseq.recode.vcf data/sam3_reseq.vcf 0.05 0.1

#vcftools --vcf data/SAP_BQSR_imputed_allchr_sns_indels_renamed.vcf --out data/scores_813_reseq --keep data/scores_813_genotypes_keep.txt --recode
#python src/HomoMAFfiltering.py data/scores_813_reseq.recode.vcf data/scores_813_reseq.vcf 0.05 0.1

#vcftools --vcf data/SAP_BQSR_imputed_allchr_sns_indels_renamed.vcf --out data/scores_828_reseq --keep data/scores_828_genotypes_keep.txt --recode
#python src/HomoMAFfiltering.py data/scores_828_reseq.recode.vcf data/scores_828_reseq.vcf 0.05 0.1
