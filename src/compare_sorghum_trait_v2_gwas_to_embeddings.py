#!/usr/bin/env python3
"""Compare sorghum_trait_data_v2 GWAS loci with SAM3 and DINOv2 embedding GWAS loci."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output" / "reframing_results"
TRAIT_DIR = BASE / "sorghum_trait_data_v2_gwas"
OUT_DIR = BASE / "sorghum_trait_data_v2_gwas_overlap"
WINDOW_BP = 200_000


def load_traits_metadata() -> pd.DataFrame:
    path = BASE / "sorghum_trait_data_v2_input" / "sorghum_trait_data_v2" / "traits.tsv"
    return pd.read_csv(path, sep="\t")[["canonical_name", "category", "description"]]


def interval_overlaps(left: pd.DataFrame, right: pd.DataFrame, window_bp: int = WINDOW_BP) -> pd.DataFrame:
    rows = []
    right_by_chrom = {chrom: group.copy() for chrom, group in right.groupby("CHROM")}
    for left_row in left.itertuples(index=False):
        candidates = right_by_chrom.get(left_row.CHROM)
        if candidates is None:
            continue
        hit = candidates[
            (candidates["end"] >= left_row.start - window_bp)
            & (candidates["start"] <= left_row.end + window_bp)
        ].copy()
        if hit.empty:
            continue
        trait_mid = (left_row.start + left_row.end) / 2
        emb_mid = (hit["start"] + hit["end"]) / 2
        hit["center_distance_bp"] = (emb_mid - trait_mid).abs()
        for right_row in hit.itertuples(index=False):
            rows.append(
                {
                    "trait_locus_id": left_row.locus_id,
                    "trait_CHROM": left_row.CHROM,
                    "trait_start": left_row.start,
                    "trait_end": left_row.end,
                    "trait_n_markers": left_row.n_markers,
                    "embedding_set": right_row.embedding_set,
                    "embedding_locus_id": right_row.locus_id,
                    "embedding_CHROM": right_row.CHROM,
                    "embedding_start": right_row.start,
                    "embedding_end": right_row.end,
                    "embedding_n_markers": right_row.n_markers,
                    "center_distance_bp": right_row.center_distance_bp,
                }
            )
    return pd.DataFrame(rows)


def summarize_embedding_support(label: str, prefix: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    loci = pd.read_csv(prefix / "gwas_signal_loci.csv")
    reps = pd.read_csv(prefix / "trait_locus_representative_signals.csv")
    clusters = pd.read_csv(prefix / "embedding_gwas_signal_clusters.csv")
    support = (
        reps.groupby("locus_id")
        .agg(
            embedding_traits=("trait", "nunique"),
            embedding_min_p=("p_value", "min"),
            embedding_best_trait=("trait", lambda x: x.iloc[0]),
        )
        .reset_index()
    )
    support["embedding_neg_log10_min_p"] = -np.log10(support["embedding_min_p"])
    support["embedding_set"] = label
    support = support.merge(
        clusters[["trait", "cluster_jaccard_similarity_ge_0_25"]].rename(
            columns={"trait": "embedding_best_trait", "cluster_jaccard_similarity_ge_0_25": "embedding_best_cluster025"}
        ),
        on="embedding_best_trait",
        how="left",
    )
    loci = loci.merge(support, on="locus_id", how="left")
    loci["embedding_set"] = label
    return loci, reps


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trait_loci = pd.read_csv(TRAIT_DIR / "trait_env_gwas_loci.csv")
    trait_reps = pd.read_csv(TRAIT_DIR / "trait_env_locus_representative_signals.csv")
    trait_summary = pd.read_csv(TRAIT_DIR / "all_trait_env_summary.csv")
    trait_meta = load_traits_metadata()

    trait_support = (
        trait_reps.groupby("locus_id")
        .agg(
            trait_envs_at_locus=("trait", "nunique"),
            canonical_traits_at_locus=("canonical_name", "nunique"),
            categories_at_locus=("canonical_name", "count"),
            trait_min_p=("p_value", "min"),
            best_trait_env=("trait", lambda x: x.iloc[0]),
        )
        .reset_index()
    )
    trait_support["trait_neg_log10_min_p"] = -np.log10(trait_support["trait_min_p"])
    trait_loci = trait_loci.merge(trait_support, on="locus_id", how="left")

    sam_loci, _ = summarize_embedding_support("SAM3", BASE / "all_sam3_full_mlm_lrt")
    dino_loci, _ = summarize_embedding_support("DINOv2", BASE / "all_dinov2_20260522_full_mlm_lrt")
    embedding_loci = pd.concat([sam_loci, dino_loci], ignore_index=True)

    overlaps = interval_overlaps(trait_loci, embedding_loci)
    overlaps = overlaps.merge(
        embedding_loci[
            [
                "embedding_set",
                "locus_id",
                "embedding_traits",
                "embedding_min_p",
                "embedding_neg_log10_min_p",
                "embedding_best_trait",
                "embedding_best_cluster025",
            ]
        ].rename(columns={"locus_id": "embedding_locus_id"}),
        on=["embedding_set", "embedding_locus_id"],
        how="left",
    )
    overlaps = overlaps.merge(
        trait_loci[
            [
                "locus_id",
                "trait_envs_at_locus",
                "canonical_traits_at_locus",
                "trait_min_p",
                "trait_neg_log10_min_p",
                "best_trait_env",
            ]
        ].rename(columns={"locus_id": "trait_locus_id"}),
        on="trait_locus_id",
        how="left",
    )
    overlaps.to_csv(OUT_DIR / "trait_embedding_locus_overlaps.csv", index=False)

    sam_flags = overlaps.query("embedding_set == 'SAM3'").groupby("trait_locus_id").size().rename("n_sam3_overlapping_loci")
    dino_flags = overlaps.query("embedding_set == 'DINOv2'").groupby("trait_locus_id").size().rename("n_dinov2_overlapping_loci")
    trait_locus_summary = trait_loci.merge(sam_flags, left_on="locus_id", right_index=True, how="left")
    trait_locus_summary = trait_locus_summary.merge(dino_flags, left_on="locus_id", right_index=True, how="left")
    trait_locus_summary[["n_sam3_overlapping_loci", "n_dinov2_overlapping_loci"]] = trait_locus_summary[
        ["n_sam3_overlapping_loci", "n_dinov2_overlapping_loci"]
    ].fillna(0).astype(int)
    trait_locus_summary["overlaps_sam3"] = trait_locus_summary["n_sam3_overlapping_loci"] > 0
    trait_locus_summary["overlaps_dinov2"] = trait_locus_summary["n_dinov2_overlapping_loci"] > 0
    trait_locus_summary["overlap_class"] = np.select(
        [
            trait_locus_summary["overlaps_sam3"] & trait_locus_summary["overlaps_dinov2"],
            trait_locus_summary["overlaps_sam3"],
            trait_locus_summary["overlaps_dinov2"],
        ],
        ["both", "SAM3_only", "DINOv2_only"],
        default="neither",
    )
    trait_locus_summary.to_csv(OUT_DIR / "trait_locus_overlap_summary.csv", index=False)

    trait_env_locus = trait_reps[["trait", "env_id", "canonical_name", "locus_id", "p_value"]].merge(
        trait_locus_summary[["locus_id", "overlaps_sam3", "overlaps_dinov2", "overlap_class"]],
        on="locus_id",
        how="left",
    )
    trait_env_summary = (
        trait_env_locus.groupby(["trait", "env_id", "canonical_name"])
        .agg(
            n_significant_loci=("locus_id", "nunique"),
            n_loci_overlapping_sam3=("overlaps_sam3", "sum"),
            n_loci_overlapping_dinov2=("overlaps_dinov2", "sum"),
            min_p_across_loci=("p_value", "min"),
        )
        .reset_index()
    )
    trait_env_summary["any_sam3_overlap"] = trait_env_summary["n_loci_overlapping_sam3"] > 0
    trait_env_summary["any_dinov2_overlap"] = trait_env_summary["n_loci_overlapping_dinov2"] > 0
    trait_env_summary = trait_env_summary.merge(trait_meta, on="canonical_name", how="left")
    trait_env_summary.to_csv(OUT_DIR / "trait_env_overlap_summary.csv", index=False)

    top_examples = overlaps.sort_values(
        ["embedding_traits", "trait_neg_log10_min_p", "embedding_neg_log10_min_p", "center_distance_bp"],
        ascending=[False, False, False, True],
    ).head(100)
    top_examples.to_csv(OUT_DIR / "top_trait_embedding_overlap_examples.csv", index=False)

    all_trait_envs = trait_summary.shape[0]
    sig_trait_envs = int((trait_summary["n_significant_effective_bonferroni"] > 0).sum())
    both = int(((trait_locus_summary["overlaps_sam3"]) & (trait_locus_summary["overlaps_dinov2"])).sum())
    sam_only = int(((trait_locus_summary["overlaps_sam3"]) & ~(trait_locus_summary["overlaps_dinov2"])).sum())
    dino_only = int((~(trait_locus_summary["overlaps_sam3"]) & trait_locus_summary["overlaps_dinov2"]).sum())
    neither = int((~trait_locus_summary["overlaps_sam3"] & ~trait_locus_summary["overlaps_dinov2"]).sum())
    discordant_total = sam_only + dino_only
    discordant_p = binomtest(dino_only, discordant_total, p=0.5).pvalue if discordant_total else np.nan
    discordant_dino_to_sam_ratio = dino_only / sam_only if sam_only else np.inf

    embedding_loci_with_trait_overlap = (
        overlaps.groupby(["embedding_set", "embedding_locus_id"]).size().reset_index(name="n_trait_loci")
    )
    embedding_overlap_counts = embedding_loci_with_trait_overlap.groupby("embedding_set").size().to_dict()
    sam_embedding_overlap = int(embedding_overlap_counts.get("SAM3", 0))
    dino_embedding_overlap = int(embedding_overlap_counts.get("DINOv2", 0))
    embedding_odds, embedding_fisher_p = fisher_exact(
        [
            [dino_embedding_overlap, int(dino_loci.shape[0]) - dino_embedding_overlap],
            [sam_embedding_overlap, int(sam_loci.shape[0]) - sam_embedding_overlap],
        ]
    )

    category_summary = (
        trait_env_summary.groupby("category")
        .agg(
            significant_trait_envs=("trait", "nunique"),
            trait_envs_with_sam3_overlap=("any_sam3_overlap", "sum"),
            trait_envs_with_dinov2_overlap=("any_dinov2_overlap", "sum"),
        )
        .reset_index()
    )
    category_summary.to_csv(OUT_DIR / "category_trait_env_overlap_summary.csv", index=False)

    report = {
        "all_trait_envs_tested": int(all_trait_envs),
        "significant_trait_envs": sig_trait_envs,
        "significant_trait_loci": int(trait_locus_summary.shape[0]),
        "trait_loci_overlap_both": both,
        "trait_loci_overlap_sam3_only": sam_only,
        "trait_loci_overlap_dinov2_only": dino_only,
        "trait_loci_overlap_neither": neither,
        "trait_loci_overlapping_sam3_any": int(trait_locus_summary["overlaps_sam3"].sum()),
        "trait_loci_overlapping_dinov2_any": int(trait_locus_summary["overlaps_dinov2"].sum()),
        "discordant_trait_loci_sam3_only": sam_only,
        "discordant_trait_loci_dinov2_only": dino_only,
        "discordant_dinov2_to_sam3_ratio": float(discordant_dino_to_sam_ratio),
        "discordant_exact_binomial_p_value": float(discordant_p),
        "sam3_embedding_loci_overlapping_trait_loci": sam_embedding_overlap,
        "dinov2_embedding_loci_overlapping_trait_loci": dino_embedding_overlap,
        "embedding_locus_overlap_fisher_or_dinov2_vs_sam3": float(embedding_odds),
        "embedding_locus_overlap_fisher_p_value": float(embedding_fisher_p),
        "sam3_loci_total": int(sam_loci.shape[0]),
        "dinov2_loci_total": int(dino_loci.shape[0]),
        "overlap_window_bp": WINDOW_BP,
    }
    (OUT_DIR / "overlap_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(category_summary.to_string(index=False))


if __name__ == "__main__":
    main()
