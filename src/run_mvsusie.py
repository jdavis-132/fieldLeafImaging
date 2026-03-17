#!/usr/bin/env python3
"""
This script does EVERYTHING:
1. Loads GWAS results for all traits (H, S, V, R, G, B)
2. Extracts chr6 locus
3. Aligns SNPs across traits
4. Creates multivariate z-score matrix
5. Loads LD matrix
6. Runs mvsusie_rss
7. Analyzes and saves results

USAGE:
    source /home/preethi/Susie/mvsusie-py/.venv/bin/activate
    export PYTHONPATH=/home/preethi/Susie/mvsusie-py/src
    python 02_run_mvsusie_RAW.py
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import pickle
import json
import argparse

# Add mvsusie-py path
sys.path.insert(0, '/work/schnablelab/jdavis132/mvsusie-py/src/')

print("="*80)
print("MVSUSIE-PY ANALYSIS")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
parser = argparse.ArgumentParser(description="Run mvSuSiE for a genomic region")
parser.add_argument("--gwas-results", type=Path, help="Directory with PANICLE GWAS results with SE values")
parser.add_argument("--ld", type=Path, help="Directory containing pre-computed LD matrix for the region")
parser.add_argument("--out-dir", type=Path, help="Path to output directory")
parser.add_argument("--chr", type=int, help="Chromosome region is located")
parser.add_argument("--start", type=int, help="Position, in bp, of beginning of region")
parser.add_argument("--end", type = int, help="Position, in bp, of end of region")
parser.add_argument("--traits", default=None,
                       help="Comma-separated list of trait names")

args = parser.parse_args()
# Directories
WORK_DIR = args.out_dir
GWAS_DIR = args.gwas_results
LD_DIR = args.ld

# Locus parameters
CHROM = args.chr

START_POS = args.start
END_POS = args.end

# Traits to analyze (ALL channels)
TRAITS = [t.strip() for t in args.traits.split(',')] if args.traits else None

# Sample size (from the first GWAS we read)
N_SAMPLES = None  # Will be inferred automatically

print(f"\nConfiguration:")
print(f"  Locus: chr{CHROM}:{START_POS:,}-{END_POS:,}")
# print(f"  Window: ±{WINDOW/1e6:.1f} Mb")
print(f"  Traits: {', '.join(TRAITS)}")

# ============================================================================
# CREATE WORKING DIRECTORY
# ============================================================================

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(f'{WORK_DIR}/plots', exist_ok=True)
print(f"  Output: {WORK_DIR}")

# ============================================================================
# STEP 1: LOAD AND FILTER GWAS
# ============================================================================

print(f"\n{'='*80}")
print(f"STEP 1: Loading GWAS results")
print(f"{'='*80}")

loci = {}
n_samples_list = []

for trait in TRAITS:
    gwas_file = f'{GWAS_DIR}/GWAS_{trait}_all_results.csv'

    if not os.path.exists(gwas_file):
        print(f"  ⚠️  WARNING: {gwas_file} not found, skipping {trait}")
        continue

    print(f"  Loading {trait}...", end=' ')

    try:
        gwas = pd.read_csv(gwas_file)

        # Check required columns
        required_cols = ['SNP', 'CHROM', 'POS', 'MLM_Effect', 'MLM_SE', 'MLM_P']
        missing_cols = [c for c in required_cols if c not in gwas.columns]

        if missing_cols:
            print(f"ERROR: Missing columns {missing_cols}")
            continue

        # Save sample size if we don't have it yet
        if N_SAMPLES is None and 'N' in gwas.columns:
            N_SAMPLES = int(gwas['N'].iloc[0])

        # Extract locus
        # Convert CHROM to string for comparison
        gwas['CHROM'] = gwas['CHROM'].astype(str)

        locus = gwas[
            (gwas['CHROM'].isin([str(CHROM), f'chr{CHROM}', f'Chr{CHROM}'])) &
            (gwas['POS'] >= START_POS) &
            (gwas['POS'] <= END_POS)
        ].copy()

        # Clean
        locus = locus[locus['MLM_P'].notna() & (locus['MLM_P'] > 0)]
        locus = locus[locus['MLM_SE'].notna() & (locus['MLM_SE'] > 0) & np.isfinite(locus['MLM_SE'])]
        locus = locus[locus['MLM_Effect'].notna() & np.isfinite(locus['MLM_Effect'])]

        # Sort by position
        locus = locus.sort_values('POS').reset_index(drop=True)

        loci[trait] = locus
        print(f"{len(locus):,} SNPs")

    except Exception as e:
        print(f"ERROR: {e}")

if len(loci) == 0:
    print("\n❌ ERROR: GWAS files could not be loaded")
    sys.exit(1)

# Update trait list with successfully loaded ones
TRAITS = list(loci.keys())
print(f"\n✓ Successfully loaded traits: {', '.join(TRAITS)}")

# If we could not infer N_SAMPLES, use default value
if N_SAMPLES is None:
    N_SAMPLES = 830  # Approximate value
    print(f"⚠️  WARNING: Sample size not found in GWAS, using N={N_SAMPLES}")
else:
    print(f"  Sample size: N={N_SAMPLES}")

# ============================================================================
# STEP 2: ALIGN SNPs ACROSS TRAITS
# ============================================================================

print(f"\n{'='*80}")
print(f"STEP 2: Aligning SNPs across traits")
print(f"{'='*80}")

# Find common SNPs
common_snps = set(loci[TRAITS[0]]['SNP'])
for trait in TRAITS[1:]:
    common_snps &= set(loci[trait]['SNP'])

print(f"  Common SNPs across all traits: {len(common_snps):,}")

if len(common_snps) == 0:
    print("\n❌ ERROR: No common SNPs across traits")
    sys.exit(1)

# Filter and sort
for trait in TRAITS:
    loci[trait] = loci[trait][loci[trait]['SNP'].isin(common_snps)].sort_values('POS').reset_index(drop=True)

# Check alignment
print(f"  Checking SNP order...", end=' ')
for i, trait in enumerate(TRAITS[1:], 1):
    if not (loci[TRAITS[0]]['SNP'] == loci[trait]['SNP']).all():
        print(f"\n❌ ERROR: SNP order mismatch for {trait}")
        sys.exit(1)
print("✓")

# Save SNP info
snp_info = loci[TRAITS[0]][['SNP', 'CHROM', 'POS']].copy()
snp_info.to_csv(f'{WORK_DIR}/snp_order.csv', index=False)
print(f"  ✓ SNP order saved: {WORK_DIR}/snp_order.csv")

# ============================================================================
# STEP 3: CREATE MULTIVARIATE Z-SCORE MATRIX
# ============================================================================

print(f"\n{'='*80}")
print(f"STEP 3: Creating multivariate z-score matrix")
print(f"{'='*80}")

z_list = []
for trait in TRAITS:
    z_trait = (loci[trait]['MLM_Effect'] / loci[trait]['MLM_SE']).values
    z_list.append(z_trait)

    # Check NaNs or Infs
    if np.any(~np.isfinite(z_trait)):
        n_bad = np.sum(~np.isfinite(z_trait))
        print(f"  ⚠️  WARNING: {trait} has {n_bad} non-finite z-scores")

z = np.column_stack(z_list)
print(f"  Z-score matrix: {z.shape} ({z.shape[0]:,} SNPs × {z.shape[1]} traits)")

# Check for NaNs/Infs
if np.any(~np.isfinite(z)):
    print(f"  ⚠️  WARNING: Removing {np.sum(~np.isfinite(z.any(axis=1)))} SNPs with invalid z-scores")
    valid_rows = np.isfinite(z).all(axis=1)
    z = z[valid_rows, :]
    snp_info = snp_info[valid_rows].reset_index(drop=True)
    for trait in TRAITS:
        loci[trait] = loci[trait][valid_rows].reset_index(drop=True)
    print(f"  Final Z-score matrix: {z.shape}")

# Save
np.save(f'{WORK_DIR}/z_scores.npy', z)
print(f"  ✓ Z-scores saved: {WORK_DIR}/z_scores.npy")

# ============================================================================
# STEP 4: LOAD LD MATRIX
# ============================================================================

print(f"\n{'='*80}")
print(f"STEP 4: Loading LD matrix")
print(f"{'='*80}")

ld_file = f'{LD_DIR}/ld_matrix_chr{CHROM}_{START_POS}_{END_POS}.npy'
ld_snps_file = f'{LD_DIR}/snps_chr{CHROM}_{START_POS}_{END_POS}.csv'

if not os.path.exists(ld_file):
    print(f"\n❌ ERROR: LD matrix not found at {ld_file}")
    print(f"\nYou must first run the script 01_compute_LD_matrix.py")
    sys.exit(1)

R = np.load(ld_file)
ld_snps = pd.read_csv(ld_snps_file)

print(f"  ✓ LD matrix loaded: {R.shape}")
print(f"  ✓ LD SNPs: {len(ld_snps):,}")

# Align LD matrix with our SNPs
print(f"\n  Aligning LD matrix with GWAS SNPs...")

# Ensure CHROM is int in both dataframes
snp_info['CHROM'] = snp_info['CHROM'].astype(int)
ld_snps['CHROM'] = ld_snps['CHROM'].astype(int)

ld_pos_to_idx = {(row['CHROM'], row['POS']): i for i, row in ld_snps.iterrows()}

ld_indices = []
gwas_indices_to_keep = []

for i, row in snp_info.iterrows():
    key = (row['CHROM'], row['POS'])
    if key in ld_pos_to_idx:
        ld_indices.append(ld_pos_to_idx[key])
        gwas_indices_to_keep.append(i)

if len(ld_indices) == 0:
    print(f"\n❌ ERROR: No common SNPs between GWAS and LD matrix")
    sys.exit(1)

print(f"  ✓ Common SNPs between GWAS and LD: {len(ld_indices):,}")

R_subset = R[np.ix_(ld_indices, ld_indices)]
z_subset = z[gwas_indices_to_keep, :]
snp_info_subset = snp_info.iloc[gwas_indices_to_keep].reset_index(drop=True)

print(f"  ✓ Final LD matrix: {R_subset.shape}")
print(f"  ✓ Final Z-scores: {z_subset.shape}")

R = R_subset
z = z_subset
snp_info = snp_info_subset

snp_info.to_csv(f'{WORK_DIR}/snp_order_final.csv', index=False)

# ============================================================================
# STEP 5: RUN mvsusie_rss
# ============================================================================

print(f"\n{'='*80}")
print(f"STEP 5: Running mvsusie_rss")
print(f"{'='*80}")

try:
    from mvsusie_py import mvsusie_rss
except ImportError as e:
    print(f"\n❌ ERROR: Could not import mvsusie_py")
    print(f"  {e}")
    sys.exit(1)

print(f"  Parameters:")
print(f"    z: {z.shape}")
print(f"    R: {R.shape}")
print(f"    n: {N_SAMPLES}")
print(f"    L: 10 (max effects)")
print(f"\n  Running (this may take several minutes)...")

fit = mvsusie_rss(
    z=z,
    R=R,
    n=N_SAMPLES,
    L=10
)

print(f"  ✓ Fine-mapping completed")


# ============================================================================
# STEP 6: ANALYZE RESULTS
# ============================================================================

print(f"\n{'='*80}")
print(f"STEP 6: Analyzing results")
print(f"{'='*80}")

# Extract PIPs per trait (each fit corresponds to one trait)
pip_dict = {}
for i, trait in enumerate(TRAITS):
    pip_dict[trait] = fit.fits[i].pip

# Compute aggregated PIP (maximum across traits)
pip_matrix = np.column_stack([pip_dict[trait] for trait in TRAITS])
pips_max = pip_matrix.max(axis=1)  # Max PIP across traits
pips_mean = pip_matrix.mean(axis=1)  # Mean PIP across traits

print(f"\nPIP statistics (max across traits):")
print(f"  Max PIP: {pips_max.max():.3f}")
print(f"  Median PIP: {np.median(pips_max):.5f}")
print(f"  SNPs with PIP > 0.1: {np.sum(pips_max > 0.1)}")
print(f"  SNPs with PIP > 0.5: {np.sum(pips_max > 0.5)}")
print(f"  SNPs with PIP > 0.9: {np.sum(pips_max > 0.9)}")

# Top SNPs
top_n = 20
top_indices = np.argsort(pips_max)[-top_n:][::-1]

print(f"\n{'='*80}")
print(f"Top {top_n} SNPs by PIP (max across traits):")
print(f"{'='*80}")
print(f"{'Rank':<6} {'SNP':<25} {'Chr':<6} {'Position':<12} {'PIP':<8}")
print(f"{'-'*80}")

for rank, idx in enumerate(top_indices, 1):
    snp = snp_info.loc[idx, 'SNP']
    chrom = snp_info.loc[idx, 'CHROM']
    pos = snp_info.loc[idx, 'POS']
    pip = pips_max[idx]
    print(f"{rank:<6} {snp:<25} {chrom:<6} {pos:<12,} {pip:<8.3f}")

# Strong candidates
strong = np.where(pips_max > 0.5)[0]
if len(strong) > 0:
    print(f"\n{'='*80}")
    print(f"{len(strong)} strong candidate SNPs (PIP > 0.5):")
    print(f"{'='*80}")
    for idx in strong:
        snp = snp_info.loc[idx, 'SNP']
        pos = snp_info.loc[idx, 'POS']
        print(f"  {snp} (pos: {pos:,}, PIP: {pips_max[idx]:.3f})")
else:
    print(f"\n  No SNPs with PIP > 0.5")

# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================

print(f"\n{'='*80}")
print(f"STEP 7: Saving results")
print(f"{'='*80}")

# Results table
results = snp_info.copy()
results['PIP_max'] = pips_max  # Max PIP across traits
results['PIP_mean'] = pips_mean  # Mean PIP across traits

# PIPs per trait
for trait in TRAITS:
    results[f'PIP_{trait}'] = pip_dict[trait]

# Coefficients per trait
from mvsusie_py import coef
try:
    coefs = coef(fit)  # Returns (p, r) array
    if coefs is not None:
        for i, trait in enumerate(TRAITS):
            results[f'coef_{trait}'] = coefs[:, i]
except:
    pass

results.to_csv(f'{WORK_DIR}/mvsusie_results.csv', index=False)
print(f"  ✓ Results: {WORK_DIR}/mvsusie_results.csv")

# Save fit object (pickle)
with open(f'{WORK_DIR}/mvsusie_fit.pkl', 'wb') as f:
    pickle.dump(fit, f)
print(f"  ✓ Fit object: {WORK_DIR}/mvsusie_fit.pkl")

# Save metadata
metadata = {
    'chrom': CHROM,
    'start_pos': START_POS,
    'end_pos': END_POS,
    'n_snps': int(z.shape[0]),
    'n_samples': int(N_SAMPLES),
    'traits': TRAITS,
    'max_pip': float(pips_max.max()),
    'n_snps_pip_gt_0.5': int(np.sum(pips_max > 0.5)),
    'converged': None  # MultiSusieResult may not have this
}

with open(f'{WORK_DIR}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓ Metadata: {WORK_DIR}/metadata.json")

# ============================================================================
# STEP 8: CREATE SIMPLE VISUALIZATIONS
# ============================================================================

print(f"\n{'='*80}")
print(f"STEP 8: Creating visualizations")
print(f"{'='*80}")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Plot 1: PIPs vs Position
    plt.figure(figsize=(12, 6))
    plt.scatter(snp_info['POS']/1e6, pips_max, alpha=0.6, s=20)
    plt.axhline(y=0.5, color='r', linestyle='--', label='PIP = 0.5')
    plt.xlabel('Position (Mb)')
    plt.ylabel('Posterior Inclusion Probability (PIP - max across traits)')
    plt.title(f'mvsusie-py Results - RAW data\nChromosome {CHROM}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{WORK_DIR}/plots/pip_vs_position.png', dpi=300)
    plt.close()
    print(f"  ✓ Plot: {WORK_DIR}/plots/pip_vs_position.png")

    # Plot 2: PIP distribution
    plt.figure(figsize=(10, 6))
    plt.hist(pips_max, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Posterior Inclusion Probability (PIP - max across traits)')
    plt.ylabel('Number of SNPs')
    plt.title('Distribution of PIPs')
    plt.axvline(x=0.5, color='r', linestyle='--', label='PIP = 0.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{WORK_DIR}/plots/pip_distribution.png', dpi=300)
    plt.close()
    print(f"  ✓ Plot: {WORK_DIR}/plots/pip_distribution.png")

except ImportError:
    print(f"  ⚠️  matplotlib not available, skipping visualizations")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print(f"✅ ANALYSIS COMPLETED SUCCESSFULLY")
print(f"{'='*80}")

print(f"\nFiles generated in {WORK_DIR}:")
print(f"  1. mvsusie_results.csv - Complete results table")
print(f"  2. mvsusie_fit.pkl - mvsusie fit object (pickle)")
print(f"  3. metadata.json - Analysis metadata")
print(f"  4. snp_order_final.csv - Final list of analyzed SNPs")
print(f"  5. plots/pip_vs_position.png - PIPs vs position plot")
print(f"  6. plots/pip_distribution.png - PIP distribution")

print(f"\nNext steps:")
print(f"  1. Review the results in {WORK_DIR}/mvsusie_results.csv")
print(f"  2. Compare with pySuSiE results (if available)")
print(f"  3. Run the analysis for color-corrected data with 03_run_mvsusie_CORRECTED.py")
print()
