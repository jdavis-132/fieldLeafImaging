import os
import pandas as pd
import numpy as np
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initial settings
signal_file = sys.argv[1]
#input_path = "test"
output_file = sys.argv[2]
#trait = sys.argv[3]
window_size = int(sys.argv[3])
min_snps_per_peak = int(sys.argv[4])

# Peak identification function
def identify_combined_peaks(df):
    """
    Identifies peaks across all traits, merging overlapping regions into single peaks.
    Ensures that peaks are only merged if they are on the same chromosome.
    """
    df = df.sort_values(by=['CHROM', 'POS'], inplace=False)
    peaks = []
    current_peak_start = None
    current_peak_end = None
    current_chrom = None  # Track the current chromosome
    peak_id = 1
    
    for _, row in df.iterrows():
        if current_peak_start is None:
            current_peak_start = row['POS']
            current_peak_end = row['POS']
            current_chrom = row['CHROM']  # Initialize chromosome tracking
            peaks.append(peak_id)
        else:
            # Ensure that SNPs are in the same chromosome before merging into the same peak
            if row['CHROM'] == current_chrom and row['POS'] - current_peak_end <= window_size:  # 1 Mb window
                current_peak_end = row['POS']
                peaks.append(peak_id)
            else:
                peak_id += 1
                current_peak_start = row['POS']
                current_peak_end = row['POS']
                current_chrom = row['CHROM']  # Update chromosome tracking
                peaks.append(peak_id)
    
    df['peak_id'] = peaks
    return df

significant_snps = pd.read_csv(signal_file)
# Filter to signals for this trait
#significant_snps = significant_snps[significant_snps['modelID']==trait]

significant_snps.reset_index(drop=True, inplace=True)

# Identifying peaks (with chromosome-based separation)
significant_snps = identify_combined_peaks(significant_snps)
            
# Select the most significant SNP per peak (smallest p-value)
top_snps_per_peak = significant_snps.loc[significant_snps.groupby('peak_id')['MLM_P'].idxmin()].rename(
    columns={'SNP': 'top_SNP', 'MLM_P': 'top_Pvalue'})[['peak_id', 'CHROM', 'POS', 'top_SNP', 'top_Pvalue']]
            
# Compute peak range, associated traits, and SNP count
peak_ranges = significant_snps.groupby('peak_id').agg(
    pStart=('POS', 'min'),
    pStop=('POS', 'max'),
    pLength=('POS', lambda x: x.max() - x.min()),
    num_SNPs=('POS', 'size')).reset_index()

# Merge top SNP info with peak ranges
peaks_summary = top_snps_per_peak.merge(peak_ranges, on='peak_id')
# Remove peaks with num_SNPs < min_snps_per_peak
peaks_summary = peaks_summary[peaks_summary['num_SNPs']>=min_snps_per_peak]
# Write to csv
peaks_summary.to_csv(output_file, index = False)