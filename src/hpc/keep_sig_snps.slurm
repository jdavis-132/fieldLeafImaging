#!/usr/bin/bash
#SBATCH --array=5
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

vcftools --vcf data/sam3.vcf --out data/all_sig_snps --recode --positions data/snps_keep.txt
