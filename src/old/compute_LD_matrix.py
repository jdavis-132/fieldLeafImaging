#!/usr/bin/env python3
"""
Script to compute LD matrix from VCF

This script:
1. Reads the VCF
2. Extracts genotypes from locus chr6
3. Computes correlation matrix (LD)
4. Saves the LD matrix

USE:
"""

import numpy as np
import pandas as pd
import gzip
import os
from pathlib import Path
import argparse

print("="*80)
print("COMPUTANDO MATRIZ LD DESDE VCF")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
parser = argparse.ArgumentParser(description="Estimate LD matrix for a genomic region")
parser.add_argument("vcf", type=Path, help="Path to the input vcf file")
parser.add_argument("--out-dir", type=Path, help="Path to output directory")
parser.add_argument("--chr", type=int, help="Chromosome region is located")
parser.add_argument("--start", type=int, help="Position, in bp, of beginning of region")
parser.add_argument("--end", type = int, help="Position, in bp, of end of region")

args = parser.parse_args()

VCF_FILE = args.vcf
OUTPUT_DIR = args.out_dir

# Locus parameters
CHROM = args.chr

START_POS = args.start
END_POS = args.end

# ============================================================================
# Make directory
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nOutput directory: {OUTPUT_DIR}")

# ============================================================================
# LD COMPUTER FUNCTION
# ============================================================================

def compute_ld_matrix(G):
    """
    Computes correlation matrix (LD) from genotypes.
    Parameters:
    -----------
    G : array (n_individuals, n_snps)
        Genotype matrix (0/1/2 or dosages)
    Returns:
    --------
    R : array (n_snps, n_snps)
        Pearson correlation matrix
    """
    print(f"  Computing LD for {G.shape[1]} SNPs...")

    # Center and scale
    means = np.mean(G, axis=0)
    stds = np.std(G, axis=0)
    stds[stds == 0] = 1  # Avoid division by zero

    G_scaled = (G - means) / stds

    # Correlation = X'X / n
    n = G.shape[0]
    R = np.dot(G_scaled.T, G_scaled) / n

    # Ensure diagonal = 1 and symmetry
    np.fill_diagonal(R, 1.0)
    R = (R + R.T) / 2

    print(f"  ✓ LD matrix computed: {R.shape}")

    return R

# ============================================================================
# READ VCF AND EXTRACT LOCUS GENOTYPES
# ============================================================================

print(f"\n{'='*80}")
print(f"STEP 1: Reading VCF for locus chr{CHROM}:{START_POS}-{END_POS}")
print(f"{'='*80}")

snp_ids = []
positions = []
genotypes_list = []

open_fn = gzip.open if str(VCF_FILE).endswith('.gz') else open
with open_fn(VCF_FILE, 'rt') as f:
    # Skip header
    for line in f:
        if line.startswith('#CHROM'):
            # Last header line with sample names
            header_parts = line.strip().split('\t')
            sample_names = header_parts[9:]  #Individuals names
            n_samples = len(sample_names)
            print(f"  Found {n_samples} samples in VCF")
            break
        elif line.startswith('#'):
            continue

    # Read data
    n_snps_total = 0
    n_snps_locus = 0

    for line in f:
        n_snps_total += 1

        if n_snps_total % 100000 == 0:
            print(f"  Processed {n_snps_total:,} SNPs, found {n_snps_locus} in locus...", end='\r')

        parts = line.strip().split('\t')
        chrom = parts[0]
        pos = int(parts[1])
        snp_id = parts[2]
        ref = parts[3]
        alt = parts[4]

        # Filter by locus
        if chrom != str(CHROM) and chrom != f'chr{CHROM}':
            continue

        if pos < START_POS or pos > END_POS:
            continue

        n_snps_locus += 1

        # Extract genotypes
        genotypes = parts[9:]

        # Convert to dosage (0/1/2)
        dosages = []
        for gt in genotypes:
            gt_call = gt.split(':')[0]  # Format: "0/0:..." o "0|0:..."

            if gt_call in ['./.', '.|.', '.']:
                dosages.append(np.nan)  # Missing
            else:
                alleles = gt_call.replace('|', '/').split('/')
                try:
                    dosage = sum(int(a) for a in alleles if a != '.')
                    dosages.append(dosage)
                except:
                    dosages.append(np.nan)

        snp_ids.append(f"{chrom}:{pos}:{ref}:{alt}")
        positions.append(pos)
        genotypes_list.append(dosages)

print(f"\n  ✓ Extracted {n_snps_locus:,} SNPs from locus")

# ============================================================================
# CONVERT TO MATRIX AND CLEAN
# ============================================================================

print(f"\n{'='*80}")
print(f"PASO 2: Processing genotypes")
print(f"{'='*80}")

# Convert to numpy array
G = np.array(genotypes_list, dtype=float).T  # (n_samples, n_snps)

print(f"  Genotype matrix: {G.shape} (samples × SNPs)")

# Calculate missingness by SNP
missing_rate = np.isnan(G).sum(axis=0) / G.shape[0]
print(f"  Missing rate range: {missing_rate.min():.2%} - {missing_rate.max():.2%}")

# Filter SNPs with high missing rates (>10%)
keep_snps = missing_rate < 0.10
G_filtered = G[:, keep_snps]
snp_ids_filtered = [s for i, s in enumerate(snp_ids) if keep_snps[i]]
positions_filtered = [p for i, p in enumerate(positions) if keep_snps[i]]

print(f"  Kept {G_filtered.shape[1]:,} SNPs with <10% missing")

# Impute missing values with the mean
col_means = np.nanmean(G_filtered, axis=0)
for j in range(G_filtered.shape[1]):
    mask = np.isnan(G_filtered[:, j])
    G_filtered[mask, j] = col_means[j]

print(f"  ✓ Missing values imputed")

# ============================================================================
# COMPUTED LD
# ============================================================================

print(f"\n{'='*80}")
print(f"STEP3: Computing LD matrix")
print(f"{'='*80}")

R = compute_ld_matrix(G_filtered)

# ============================================================================
# SAVED RESULTS
# ============================================================================

print(f"\n{'='*80}")
print(f"PASO 4: Saved results")
print(f"{'='*80}")

# Save LD matrix
ld_file = f'{OUTPUT_DIR}/ld_matrix_chr{CHROM}_{START_POS}_{END_POS}.npy'
np.save(ld_file, R)
print(f"  ✓ LD matrix saved: {ld_file}")

# Save SNP information
snp_info = pd.DataFrame({
    'SNP': snp_ids_filtered,
    'CHROM': CHROM,
    'POS': positions_filtered
})
snp_file = f'{OUTPUT_DIR}/snps_chr{CHROM}_{START_POS}_{END_POS}.csv'
snp_info.to_csv(snp_file, index=False)
print(f"  ✓ SNP info saved: {snp_file}")

# Save metadata
import json
metadata = {
    'chrom': CHROM,
    'start_pos': START_POS,
    'end_pos': END_POS,
    'n_snps': int(R.shape[0]),
    'n_samples': int(G_filtered.shape[0]),
    'vcf_file': str(VCF_FILE)
}

meta_file = f'{OUTPUT_DIR}/metadata_chr{CHROM}_{START_POS}_{END_POS}.json'
with open(meta_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓ Metadata saved: {meta_file}")

print(f"\n{'='*80}")
print(f"DONE!")
print(f"{'='*80}")
print(f"\nArchivos generados:")
print(f"  1. {ld_file}")
print(f"  2. {snp_file}")
print(f"  3. {meta_file}")
print()
