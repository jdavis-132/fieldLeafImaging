#!/usr/bin/env python3
"""Classify DINOv2 GWAS-signal clusters by severity correlation and anthracnose candidate support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr


ROOT = Path(__file__).resolve().parents[1]
GWAS_DIR = ROOT / "output" / "reframing_results" / "all_dinov2_20260522_full_mlm_lrt"
PHENO_FILE = ROOT / "output" / "dinov2_20260522_blues" / "dinov2_20260522_blues_all_compatible_gwas_ids.csv"
OUT_DIR = ROOT / "output" / "reframing_results" / "dinov2_cluster_disease_classification"
CANDIDATE_FILE = (
    ROOT
    / "output"
    / "reframing_results"
    / "cluster_disease_classification"
    / "full_anthracnose_candidate_genes_mapped_to_annotation.csv"
)


def odds_ratio(a: int, b: int, c: int, d: int) -> float:
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def severity_correlations() -> pd.DataFrame:
    dino = pd.read_csv(PHENO_FILE)
    dino = dino.loc[dino["location"].eq("NE")].copy()
    severity = pd.read_csv(ROOT / "data" / "blues_all.csv", usecols=["location", "genotype", "human_score", "percentUnhealthy"])
    severity = severity.loc[severity["location"].eq("NE")].copy()
    merged = dino.merge(severity, on=["location", "genotype"], how="inner")
    traits = [c for c in merged.columns if c.startswith("dinov2_mean_") or c.startswith("dinov2_std_")]

    rows = []
    for trait in traits:
        for severity_col in ["human_score", "percentUnhealthy"]:
            sub = merged[[trait, severity_col]].dropna()
            rho = spearmanr(sub[trait], sub[severity_col]).statistic if len(sub) >= 3 else np.nan
            rows.append(
                {
                    "trait": trait,
                    "severity": severity_col,
                    "n": int(len(sub)),
                    "spearman": float(rho),
                    "abs_spearman": float(abs(rho)),
                }
            )
    return pd.DataFrame(rows)


def candidate_loci(loci: pd.DataFrame) -> pd.DataFrame:
    candidates = pd.read_csv(CANDIDATE_FILE).dropna(subset=["CHROM", "start", "end"]).copy()
    candidates["CHROM"] = candidates["CHROM"].astype(int).astype(str)
    loci = loci.copy()
    loci["CHROM"] = loci["CHROM"].astype(str)

    rows = []
    for locus in loci.itertuples(index=False):
        hits = candidates[
            candidates["CHROM"].eq(str(locus.CHROM))
            & (candidates["end"].astype(float) >= int(locus.start) - 200_000)
            & (candidates["start"].astype(float) <= int(locus.end) + 200_000)
        ]
        for hit in hits.itertuples(index=False):
            rows.append(
                {
                    "locus_id": locus.locus_id,
                    "gene_id": hit.gene_id,
                    "category": hit.category,
                    "source_sheet": hit.source_sheet,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    severity = severity_correlations()
    severity.to_csv(OUT_DIR / "all_dinov2_severity_correlations.csv", index=False)
    severity_wide = (
        severity.pivot_table(index="trait", columns="severity", values="abs_spearman")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    severity_wide["max_abs_severity_r"] = severity_wide[["human_score", "percentUnhealthy"]].max(axis=1)
    severity_wide["severity_r_ge_0_4"] = severity_wide["max_abs_severity_r"] >= 0.4
    severity_wide["severity_r_ge_0_5"] = severity_wide["max_abs_severity_r"] >= 0.5

    clusters = pd.read_csv(GWAS_DIR / "embedding_gwas_signal_clusters.csv")
    clusters = clusters.loc[clusters["n_significant_loci"] > 0].copy()
    clusters["cluster025"] = clusters["cluster_jaccard_similarity_ge_0_25"].astype(int)
    clusters = clusters.merge(severity_wide, on="trait", how="left")

    reps = pd.read_csv(GWAS_DIR / "trait_locus_representative_signals.csv")
    loci = pd.read_csv(GWAS_DIR / "gwas_signal_loci.csv")
    candidate_hits = candidate_loci(loci)
    candidate_hits.to_csv(OUT_DIR / "anthracnose_candidate_genes_hit_by_dinov2_full_mlm_loci.csv", index=False)

    trait_loci = reps[["trait", "locus_id"]].drop_duplicates()
    trait_loci = trait_loci.merge(candidate_hits, on="locus_id", how="left")
    trait_loci["has_validated_locus"] = trait_loci["category"].eq("validated_anthracnose")
    trait_loci["has_proposed_locus"] = trait_loci["category"].eq("proposed_anthracnose")
    trait_candidate = (
        trait_loci.groupby("trait")
        .agg(
            n_loci=("locus_id", "nunique"),
            has_validated_locus=("has_validated_locus", "max"),
            has_proposed_locus=("has_proposed_locus", "max"),
            candidate_loci=("locus_id", lambda x: ";".join(sorted(set(x.dropna())))),
            candidate_genes=("gene_id", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
        )
        .reset_index()
    )
    for col in ["has_validated_locus", "has_proposed_locus"]:
        trait_candidate[col] = trait_candidate[col].fillna(False).astype(bool)
    trait_candidate["has_validated_or_proposed_locus"] = (
        trait_candidate["has_validated_locus"] | trait_candidate["has_proposed_locus"]
    )

    trait_table = clusters.merge(trait_candidate, on="trait", how="left")
    for col in ["has_validated_locus", "has_proposed_locus", "has_validated_or_proposed_locus"]:
        trait_table[col] = trait_table[col].fillna(False).astype(bool)
    trait_table.to_csv(OUT_DIR / "dinov2_embedding_cluster_severity_candidate_table.csv", index=False)

    cluster_summary = (
        trait_table.groupby("cluster025")
        .agg(
            n_embeddings=("trait", "size"),
            median_max_abs_severity_r=("max_abs_severity_r", "median"),
            mean_max_abs_severity_r=("max_abs_severity_r", "mean"),
            max_abs_severity_r=("max_abs_severity_r", "max"),
            fraction_embeddings_r_ge_0_4=("severity_r_ge_0_4", "mean"),
            fraction_embeddings_r_ge_0_5=("severity_r_ge_0_5", "mean"),
            n_embeddings_r_ge_0_4=("severity_r_ge_0_4", "sum"),
            n_embeddings_r_ge_0_5=("severity_r_ge_0_5", "sum"),
            n_loci=("n_loci", "sum"),
            has_validated_locus=("has_validated_locus", "max"),
            has_validated_or_proposed_locus=("has_validated_or_proposed_locus", "max"),
            candidate_genes=(
                "candidate_genes",
                lambda x: ";".join(sorted({g for v in x.dropna().astype(str) for g in v.split(";") if g})),
            ),
            example_embeddings=("trait", lambda x: ";".join(list(x)[:12])),
        )
        .reset_index()
    )
    cluster_summary["cluster_type"] = np.where(cluster_summary["n_embeddings"] > 1, "multi_trait", "singleton")
    cluster_summary["severity_related"] = (
        (cluster_summary["median_max_abs_severity_r"] >= 0.4)
        | (cluster_summary["fraction_embeddings_r_ge_0_4"] >= 0.5)
        | (cluster_summary["max_abs_severity_r"] >= 0.5)
    )
    cluster_summary["severity_class"] = np.where(
        cluster_summary["severity_related"], "severity_correlated", "not_severity_correlated"
    )
    cluster_summary.to_csv(OUT_DIR / "dinov2_cluster025_severity_candidate_summary.csv", index=False)

    rows = []
    for unit_set, frame in [
        ("all_cluster_labels_including_singletons", cluster_summary),
        ("multi_trait_clusters_only", cluster_summary.loc[cluster_summary["cluster_type"].eq("multi_trait")]),
        ("singletons_only", cluster_summary.loc[cluster_summary["cluster_type"].eq("singleton")]),
    ]:
        sev = frame["severity_related"]
        for candidate_col in ["has_validated_locus", "has_validated_or_proposed_locus"]:
            cand = frame[candidate_col]
            a = int((sev & cand).sum())
            b = int((sev & ~cand).sum())
            c = int((~sev & cand).sum())
            d = int((~sev & ~cand).sum())
            fisher_or, fisher_p = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append(
                {
                    "unit_set": unit_set,
                    "candidate_definition": candidate_col,
                    "severity_related_with_candidate": a,
                    "severity_related_without_candidate": b,
                    "not_severity_related_with_candidate": c,
                    "not_severity_related_without_candidate": d,
                    "severity_related_candidate_fraction": a / (a + b) if (a + b) else np.nan,
                    "not_severity_related_candidate_fraction": c / (c + d) if (c + d) else np.nan,
                    "odds_ratio": odds_ratio(a, b, c, d),
                    "fisher_exact_p_greater": fisher_p,
                    "fisher_exact_or": fisher_or,
                }
            )
    enrichment = pd.DataFrame(rows)
    enrichment.to_csv(OUT_DIR / "dinov2_severity_related_cluster_candidate_enrichment.csv", index=False)

    report = [
        "# DINOv2 Cluster Disease Classification",
        "",
        "Severity-related classification uses only cluster member correlations with human score and ExG percent-unhealthy area.",
        "A cluster/singleton is called severity-correlated if median max(|r_human|, |r_ExG|) >= 0.4, or at least half of members have max |r| >= 0.4, or any member has max |r| >= 0.5.",
        "",
        "## Counts",
        cluster_summary.groupby(["cluster_type", "severity_class"]).size().reset_index(name="n").to_markdown(index=False),
        "",
        "## Candidate Enrichment",
        enrichment.to_markdown(index=False),
        "",
        "## Top Severity-Correlated Labels",
        cluster_summary.sort_values(
            ["severity_related", "median_max_abs_severity_r", "n_embeddings"],
            ascending=[False, False, False],
        )
        .head(30)[
            [
                "cluster025",
                "cluster_type",
                "n_embeddings",
                "median_max_abs_severity_r",
                "max_abs_severity_r",
                "fraction_embeddings_r_ge_0_4",
                "has_validated_locus",
                "has_validated_or_proposed_locus",
                "candidate_genes",
                "example_embeddings",
            ]
        ]
        .to_markdown(index=False),
        "",
    ]
    (OUT_DIR / "dinov2_cluster_disease_classification_report.md").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
