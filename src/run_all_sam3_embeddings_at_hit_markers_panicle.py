#!/usr/bin/env python3
"""Run LOCO MLM for all SAM3 embeddings at known hit-marker positions.

This is a targeted all-embedding expansion, not a genome-wide scan. It tests
all 1,024 SAM3 mean embeddings and all 1,024 SAM3 std embeddings at the
previously RMIP-supported marker positions using genome-wide PCs and LOCO
kinship from the filtered 895-sample PLINK matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from panicle.association.mlm_loco import PANICLE_MLM_LOCO_MULTI
from panicle.data.loaders import load_genotype_file
from panicle.matrix.kinship_loco import PANICLE_K_VanRaden_LOCO
from panicle.matrix.pca import PANICLE_PCA


ROOT = Path(__file__).resolve().parents[1]
CONDITIONAL_DIR = ROOT / "output" / "conditional_panicle"
DATASET_DIR = CONDITIONAL_DIR / "condition_exg"
OUT_DIR = ROOT / "output" / "reframing_results" / "all_sam3_hit_marker_loco"


def bh_qvalues(pvalues: pd.Series) -> pd.Series:
    q = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.astype(float).dropna().sort_values()
    m = valid.shape[0]
    if m == 0:
        return q
    ranked = valid * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked.iloc[::-1]).iloc[::-1].clip(upper=1)
    q.loc[adjusted.index] = adjusted
    return q


def marker_ids(geno_map) -> np.ndarray:
    return geno_map.to_dataframe()["MARKER"].astype(str).to_numpy()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    genome_plink = DATASET_DIR / "condition_exg_plink"
    hit_vcf = DATASET_DIR / "condition_exg_hit_positions.vcf.gz"

    print(f"Loading genome-wide genotype matrix: {genome_plink}", flush=True)
    genome_geno, genome_ids, genome_map = load_genotype_file(
        genome_plink,
        file_format="plink",
        precompute_alleles=False,
    )
    print(
        f"Genome matrix: {genome_geno.n_individuals} samples x "
        f"{genome_geno.n_markers} markers",
        flush=True,
    )

    print("Computing 5 PCs", flush=True)
    pcs = PANICLE_PCA(M=genome_geno, pcs_keep=5, verbose=False)

    print("Computing LOCO kinship", flush=True)
    loco = PANICLE_K_VanRaden_LOCO(genome_geno, genome_map, verbose=False)

    print(f"Loading hit-marker genotype matrix: {hit_vcf}", flush=True)
    hit_geno, hit_ids, hit_map = load_genotype_file(
        hit_vcf,
        file_format="vcf",
        precompute_alleles=False,
    )
    if list(hit_ids) != list(genome_ids):
        raise ValueError("Hit-marker VCF sample order does not match genome PLINK order")

    blues = pd.read_csv(ROOT / "data" / "blues_all.csv")
    embedding_cols = [
        c for c in blues.columns
        if c.startswith("embedding_mean_") or c.startswith("embedding_std_")
    ]
    ne_blues = blues.loc[blues["location"].eq("NE"), ["genotype", *embedding_cols]]
    pheno = ne_blues.set_index("genotype").reindex(genome_ids)
    missing_rows = pheno[embedding_cols].isna().any(axis=1).sum()
    if missing_rows:
        raise ValueError(f"{missing_rows} aligned samples have missing SAM3 embedding values")

    y_matrix = pheno[embedding_cols].to_numpy(dtype=float)
    print(
        f"Running LOCO MLM for {len(embedding_cols)} embeddings x "
        f"{hit_geno.n_markers} hit markers",
        flush=True,
    )
    results = PANICLE_MLM_LOCO_MULTI(
        phe=y_matrix,
        geno=hit_geno,
        map_data=hit_map,
        trait_names=embedding_cols,
        loco_kinship=loco,
        CV=pcs,
        cpu=1,
        verbose=False,
        lrt_refinement=True,
    )

    hit_markers = marker_ids(hit_map)
    rows = []
    for trait, res in results.items():
        df = res.to_dataframe()
        if len(df) != len(hit_markers):
            raise ValueError(f"Unexpected result length for {trait}: {len(df)}")
        stat = "mean" if trait.startswith("embedding_mean_") else "std"
        embedding_index = int(trait.rsplit("_", 1)[1])
        for i, row in df.iterrows():
            marker = hit_markers[i]
            chrom, pos, ref, alt = marker.split(":", 3)
            rows.append(
                {
                    "trait": trait,
                    "embedding_stat": stat,
                    "embedding_index": embedding_index,
                    "marker": marker,
                    "chromosome": chrom,
                    "position": int(pos),
                    "ref": ref,
                    "alt": alt,
                    "effect": float(row["Effect"]),
                    "se": float(row["SE"]),
                    "p_value": float(row["P-value"]),
                    "n_samples": len(genome_ids),
                }
            )

    out = pd.DataFrame(rows)
    out["q_value_all_tests"] = bh_qvalues(out["p_value"])
    out["q_value_within_trait"] = np.nan
    for _trait, idx in out.groupby("trait").groups.items():
        out.loc[idx, "q_value_within_trait"] = bh_qvalues(out.loc[idx, "p_value"])

    out_path = OUT_DIR / "all_sam3_embeddings_hit_marker_loco_results.csv"
    out.to_csv(out_path, index=False)

    summary_rows = []
    for label, frame in [
        ("all", out),
        ("mean", out.loc[out["embedding_stat"].eq("mean")]),
        ("std", out.loc[out["embedding_stat"].eq("std")]),
    ]:
        summary_rows.append(
            {
                "embedding_set": label,
                "n_tests": len(frame),
                "n_traits": frame["trait"].nunique(),
                "n_markers": frame["marker"].nunique(),
                "n_p_lt_0_05": int((frame["p_value"] < 0.05).sum()),
                "n_p_lt_1e_4": int((frame["p_value"] < 1e-4).sum()),
                "n_q_all_lt_0_05": int((frame["q_value_all_tests"] < 0.05).sum()),
                "n_within_trait_q_lt_0_05": int((frame["q_value_within_trait"] < 0.05).sum()),
                "n_traits_with_within_trait_q_lt_0_05": int(
                    frame.loc[frame["q_value_within_trait"] < 0.05, "trait"].nunique()
                ),
                "n_markers_with_within_trait_q_lt_0_05": int(
                    frame.loc[frame["q_value_within_trait"] < 0.05, "marker"].nunique()
                ),
                "min_p": float(frame["p_value"].min()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "all_sam3_embeddings_hit_marker_loco_summary.csv"
    summary.to_csv(summary_path, index=False)

    top_path = OUT_DIR / "all_sam3_embeddings_hit_marker_loco_top100.csv"
    out.sort_values("p_value").head(100).to_csv(top_path, index=False)

    print(summary.to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
