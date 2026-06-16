# Human-score GWAS and embedding overlap summary

Date: 2026-06-09

This note summarizes the current analysis of whether relaxed human disease severity GWAS peaks are recovered by SAM3 and DINOv2 image-embedding GWAS signals.

## Datasets and GWAS model

We analyzed two embedding sets:

| Embedding set | Traits tested | Traits with genome-wide hits | Significant marker-trait hits | Unique significant markers | Merged GWAS loci |
|---|---:|---:|---:|---:|---:|
| SAM3 | 2,048 | 648 | 13,809 | 5,555 | 319 |
| DINOv2 | 2,048 | 719 | 37,437 | 9,725 | 286 |

GWAS used PANICLE LOCO MLM with 5 genotype PCs, LOCO kinship, and LRT refinement. All analyses used the same filtered PLINK genotype cache derived from the sorghum VCF:

`output/conditional_panicle/condition_exg/condition_exg_plink`

The effective-marker Bonferroni threshold was:

`0.05 / 3,169,884 = 1.58e-08`

## Embedding redundancy

Many embedding traits tag the same markers or genomic intervals.

At Jaccard similarity >= 0.25 among GWAS-signal sets:

| Embedding set | Significant traits | GWAS-signal groups | Multi-trait clusters | Singleton traits | Significant traits in multi-trait clusters | Largest cluster |
|---|---:|---:|---:|---:|---:|---:|
| SAM3 | 648 | 212 | 87 | 125 | 523 | 140 traits |
| DINOv2 | 719 | 185 | 77 | 108 | 611 | 175 traits |

At Jaccard similarity >= 0.50:

| Embedding set | GWAS-signal groups | Multi-trait clusters | Singleton traits | Significant traits in multi-trait clusters | Largest cluster |
|---|---:|---:|---:|---:|---:|
| SAM3 | 265 | 87 | 178 | 470 | 129 traits |
| DINOv2 | 237 | 87 | 150 | 569 | 152 traits |

Interpretation: the embeddings are partly redundant, but not fully redundant. Many traits collapse into shared GWAS-signal groups, while a substantial number remain singleton signals.

## Direct human-score GWAS

We ran direct GWAS on genotype-level NE `human_score` values from `data/blues_all.csv`, aligned to the same genotype cache.

Human score had 560 aligned genotypes with nonmissing human-score values. It did not produce any globally significant GWAS hits:

| Trait | Samples | Best p | Global significant markers |
|---|---:|---:|---:|
| human_score | 560 | 1.59e-08 | 0 |

The best human-score p-value was just above the effective-marker threshold of 1.58e-08.

Because the goal here is not independent discovery but checking whether human-score signals overlap a predefined set of embedding GWAS hits, we also used a relaxed within-trait FDR threshold:

`human_score q < 0.2`

This produced:

- 62 relaxed human-score markers
- 18 relaxed human-score loci after merging markers within 200 kb

ExG percent-unhealthy was not included in the relaxed human-score overlap analysis summarized here.

## Anthracnose candidate overlap among relaxed human-score peaks

Two relaxed human-score loci were within 200 kb of proposed anthracnose candidate genes:

| Human-score locus | Human peak marker | Human p | Human q | Candidate genes |
|---|---:|---:|---:|---|
| Chr2: 7.47 Mb | 2:7468611:A:AAAAAG | 7.60e-07 | 0.104 | Sobic.002G072000 |
| Chr5: 70.94 Mb | 5:70937894:G:A | 1.59e-08 | 0.0237 | Sobic.005G177500; Sobic.005G177600 |

## Full embedding GWAS overlap with relaxed human-score peaks

Using the full embedding GWAS results, 7 of the 18 relaxed human-score loci overlapped at least one significant embedding GWAS locus within the human locus plus/minus 200 kb:

| Overlap class | Count |
|---|---:|
| Relaxed human-score loci total | 18 |
| Recovered by any embedding GWAS interval | 7 |
| Recovered by SAM3 | 5 |
| Recovered by DINOv2 | 4 |
| Recovered by both SAM3 and DINOv2 | 2 |

For the two anthracnose-adjacent human-score loci:

| Human-score locus | Candidate genes | Full embedding recovery | Number of recovering embeddings |
|---|---|---|---:|
| Chr2: 7.47 Mb | Sobic.002G072000 | DINOv2 `dinov2_std_731` | 1 |
| Chr5: 70.94 Mb | Sobic.005G177500; Sobic.005G177600 | SAM3 `embedding_std_161` | 1 |

Neither anthracnose-adjacent peak was recovered by multiple embedding traits under the current overlap definition.

## Matched-genotype local peak comparison

To test whether stronger embedding evidence was caused simply by larger embedding sample size, we reran the overlapping embedding traits using only the same 560 genotypes with human-score data.

For each relaxed human-score locus, we compared:

- the human-score local peak p-value;
- the best matched-genotype embedding local peak p-value inside the human locus plus/minus 200 kb;
- local marker R2 for the human peak and embedding peak.

R2 definition: marker dosage and phenotype were each residualized on an intercept plus 5 genotype PCs. R2 is the squared Pearson correlation between residualized marker dosage and residualized phenotype. Kinship was not included in this R2 summary.

We classified matched p-value differences by delta log10 p:

- embedding more significant: embedding local peak is more than 0.5 log10 p units stronger than human;
- roughly equivalent: within 0.5 log10 p units;
- embedding less significant: embedding local peak is more than 0.5 log10 p units weaker than human.

## Relaxed human-score peak recovery table

| Human locus | Human marker | Human p | Human R2 | Candidate | Best embedding status | Matched embedding p | Embedding R2 |
|---|---:|---:|---:|---|---|---:|---:|
| Chr10: 2.54 Mb | 10:2536853:C:A | 1.82e-06 | 0.050 |  | not recovered |  |  |
| Chr10: 37.20 Mb | 10:37204263:T:G | 9.99e-07 | 0.043 |  | not recovered |  |  |
| Chr10: 51.30 Mb | 10:51296457:A:G | 2.07e-07 | 0.049 |  | SAM3 `embedding_mean_716`, roughly equivalent | 2.16e-07 | 0.061 |
| Chr1: 20.90 Mb | 1:20899785:C:T | 1.28e-06 | 0.041 |  | SAM3 `embedding_std_215`, less significant | 2.40e-05 | 0.027 |
| Chr1: 67.68 Mb | 1:67678877:A:G | 2.40e-06 | 0.036 |  | DINOv2 `dinov2_mean_611`, more significant | 4.81e-07 | 0.060 |
| Chr1: 77.40-77.48 Mb | 1:77467301:G:A | 1.09e-07 | 0.051 |  | SAM3 `embedding_mean_108`, roughly equivalent | 8.53e-08 | 0.067 |
| Chr1: 77.84 Mb | 1:77836487:T:C | 1.93e-06 | 0.048 |  | not recovered |  |  |
| Chr1: 8.19 Mb | 1:8194464:TATTAAAAAA:T | 1.72e-06 | 0.039 |  | DINOv2 `dinov2_std_9`, more significant | 1.70e-07 | 0.056 |
| Chr1: 84.13 Mb | 1:84132672:TCGTTGGAGATA:T | 1.37e-06 | 0.053 |  | not recovered |  |  |
| Chr2: 7.47 Mb | 2:7468611:A:AAAAAG | 7.60e-07 | 0.032 | Sobic.002G072000 | DINOv2 `dinov2_std_731`, more significant; matched global sig | 4.99e-09 | 0.064 |
| Chr3: 12.24 Mb | 3:12240087:T:C | 2.00e-07 | 0.043 |  | not recovered |  |  |
| Chr3: 18.10 Mb | 3:18097659:G:A | 8.08e-07 | 0.053 |  | not recovered |  |  |
| Chr3: 18.83 Mb | 3:18829951:C:T | 2.18e-06 | 0.052 |  | not recovered |  |  |
| Chr5: 70.94 Mb | 5:70937894:G:A | 1.59e-08 | 0.057 | Sobic.005G177500; Sobic.005G177600 | SAM3 `embedding_std_161`, less significant | 2.90e-07 | 0.046 |
| Chr8: 2.35 Mb | 8:2354126:C:T | 5.99e-08 | 0.062 |  | not recovered |  |  |
| Chr8: 53.39 Mb | 8:53390594:T:A | 4.76e-07 | 0.042 |  | not recovered |  |  |
| Chr8: 54.05 Mb | 8:54053211:G:A | 5.89e-07 | 0.034 |  | not recovered |  |  |
| Chr8: 55.09 Mb | 8:55099770:A:C | 1.82e-07 | 0.050 |  | not recovered |  |  |

## Key anthracnose-adjacent cases

### Chr2 near Sobic.002G072000

This is the strongest example that an embedding can recover and sharpen a relaxed human-score signal.

Human-score signal:

- peak marker: `2:7468611:A:AAAAAG`
- human p: 7.60e-07
- human q: 0.104
- human local marker R2: 0.032

DINOv2 recovery:

- embedding: `dinov2_std_731`
- full-data embedding p at overlapping locus: 4.77e-09
- matched-genotype local peak marker: `2:7535841:G:C`
- matched-genotype local peak p: 4.99e-09
- matched local marker R2: 0.064
- matched local peak remains globally significant

This embedding is moderately correlated with human score:

- Spearman rho with human score: 0.341
- n = 598
- p = 9.12e-18

### Chr5 near Sobic.005G177500 and Sobic.005G177600

This is a weaker embedding recovery case.

Human-score signal:

- peak marker: `5:70937894:G:A`
- human p: 1.59e-08
- human q: 0.0237
- human local marker R2: 0.057

SAM3 full-data embedding recovery:

- embedding: `embedding_std_161`
- full-data SAM3 peak marker: `5:71000192:T:G`
- full-data SAM3 p: 3.65e-09

Matched-genotype local comparison:

- matched local peak marker: `5:70994272:A:G`
- matched local peak p: 2.90e-07
- matched local marker R2: 0.046
- not globally significant in the 560-genotype matched rerun
- weaker than the human-score local peak in the matched local comparison

This embedding is modestly correlated with human score:

- Spearman rho with human score: 0.261
- n = 598
- p = 9.14e-11

## Relationship between the two anthracnose-adjacent embeddings

`dinov2_std_731` and `embedding_std_161` are moderately correlated with each other on the same n = 598 human-score genotype set:

- Spearman rho: 0.325
- p = 3.48e-16
- Pearson r: 0.368
- p = 1.31e-20

They are not redundant. Each recovers a different anthracnose-adjacent human-score locus.

## Current interpretation

Human disease scores alone do not identify genome-wide significant loci, although the top signal is very close to the global threshold. Relaxed human-score peaks provide useful anchors for asking whether image embeddings capture genetically controlled disease-associated image features.

The embedding GWAS results show three patterns:

1. Many relaxed human-score peaks are not recovered by embedding GWAS.
2. Some peaks are recovered by embeddings but are not stronger after matching sample size.
3. At least one anthracnose-adjacent peak, Chr2 near `Sobic.002G072000`, is recovered more strongly by a DINOv2 embedding even when restricted to the same human-scored genotypes.

The strongest lead example is therefore Chr2 near `Sobic.002G072000`, where the human score is suggestive, while DINOv2 `dinov2_std_731` gives a stronger and globally significant matched-genotype local signal with higher local marker R2.

The Chr5 anthracnose-adjacent peak remains relevant biologically, but the matched-genotype comparison does not support the claim that the SAM3 embedding is more statistically powerful at that interval than the human score.

## Key output files

- Direct severity GWAS:
  `output/reframing_results/ne_severity_trait_mlm_lrt/ne_severity_trait_gwas_summary.csv`
- Relaxed human-score loci:
  `output/reframing_results/ne_severity_trait_mlm_lrt/human_score_relaxed_q020_loci.csv`
- Relaxed human-score candidate gene overlaps:
  `output/reframing_results/ne_severity_trait_mlm_lrt/human_score_relaxed_q020_anthracnose_candidate_gene_overlaps.csv`
- Full embedding overlap with relaxed human peaks:
  `output/reframing_results/ne_severity_trait_mlm_lrt/human_score_relaxed_q020_overlaps_with_embedding_gwas.csv`
- Matched-genotype local peak and R2 summary:
  `output/reframing_results/ne_severity_trait_mlm_lrt/human_matched_embedding_gwas/human_relaxed_peak_recovery_summary_local_peak_r2.csv`
- Matched-genotype local embedding peaks:
  `output/reframing_results/ne_severity_trait_mlm_lrt/human_matched_embedding_gwas/matched_embedding_local_peaks_in_human_relaxed_windows.csv`
