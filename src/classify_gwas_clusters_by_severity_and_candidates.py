#!/usr/bin/env python3
"""Classify SAM3 GWAS-signal clusters by severity correlation and candidate loci."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parents[1]
GWAS_DIR = ROOT / "output" / "reframing_results" / "all_sam3_full_mlm_lrt"
STRUCT_DIR = ROOT / "output" / "reframing_results" / "embedding_structure"
OUT_DIR = ROOT / "output" / "reframing_results" / "cluster_disease_classification"


def marker_to_locus(marker: str, loci: pd.DataFrame) -> str | None:
    chrom, pos, *_ = str(marker).split(":")
    pos = int(pos)
    hits = loci[
        (loci["CHROM"].astype(str) == str(chrom))
        & (loci["start"].astype(int) <= pos)
        & (loci["end"].astype(int) >= pos)
    ]
    if hits.empty:
        return None
    return str(hits.iloc[0]["locus_id"])


def odds_ratio(a: int, b: int, c: int, d: int) -> float:
    # Haldane-Anscombe correction for small or zero cells.
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    severity = pd.read_csv(STRUCT_DIR / "all_sam3_severity_correlations.csv")
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

    known = pd.read_csv(ROOT / "figures" / "supplemental" / "hits_near_known_Rgenes.csv")
    proposed = pd.read_csv(ROOT / "figures" / "supplemental" / "hits_near_proposed_Rgenes.csv")
    new = pd.read_csv(ROOT / "figures" / "supplemental" / "new_candidate_genes.csv")
    marker_categories = []
    for label, df in [
        ("validated", known),
        ("proposed", proposed),
        ("new_annotation", new),
    ]:
        for row in df.itertuples(index=False):
            locus_id = marker_to_locus(row.Marker, loci)
            marker_categories.append(
                {
                    "marker": row.Marker,
                    "gene": row.Gene,
                    "category": label,
                    "locus_id": locus_id,
                }
            )
    marker_cat = pd.DataFrame(marker_categories).dropna(subset=["locus_id"])
    marker_cat.to_csv(OUT_DIR / "reported_candidate_markers_mapped_to_full_mlm_loci.csv", index=False)

    trait_loci = reps[["trait", "locus_id"]].drop_duplicates()
    trait_loci = trait_loci.merge(marker_cat, on="locus_id", how="left")
    trait_loci["has_validated_locus"] = trait_loci["category"].eq("validated")
    trait_loci["has_proposed_locus"] = trait_loci["category"].eq("proposed")
    trait_loci["has_new_annotation_locus"] = trait_loci["category"].eq("new_annotation")
    trait_candidate = (
        trait_loci.groupby("trait")
        .agg(
            n_loci=("locus_id", "nunique"),
            has_validated_locus=("has_validated_locus", "max"),
            has_proposed_locus=("has_proposed_locus", "max"),
            has_new_annotation_locus=("has_new_annotation_locus", "max"),
            candidate_loci=("locus_id", lambda x: ";".join(sorted(set(x.dropna())))),
            reported_candidate_genes=("gene", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
        )
        .reset_index()
    )
    for col in ["has_validated_locus", "has_proposed_locus", "has_new_annotation_locus"]:
        trait_candidate[col] = trait_candidate[col].fillna(False).astype(bool)
    trait_candidate["has_validated_or_proposed_locus"] = (
        trait_candidate["has_validated_locus"] | trait_candidate["has_proposed_locus"]
    )
    trait_candidate["has_any_reported_candidate_locus"] = (
        trait_candidate["has_validated_or_proposed_locus"] | trait_candidate["has_new_annotation_locus"]
    )

    trait_table = clusters.merge(trait_candidate, on="trait", how="left")
    for col in [
        "has_validated_locus",
        "has_proposed_locus",
        "has_new_annotation_locus",
        "has_validated_or_proposed_locus",
        "has_any_reported_candidate_locus",
    ]:
        trait_table[col] = trait_table[col].fillna(False).astype(bool)
    trait_table.to_csv(OUT_DIR / "embedding_cluster_severity_candidate_table.csv", index=False)

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
            has_any_reported_candidate_locus=("has_any_reported_candidate_locus", "max"),
            reported_candidate_genes=(
                "reported_candidate_genes",
                lambda x: ";".join(sorted({g for v in x.dropna().astype(str) for g in v.split(";") if g})),
            ),
            example_embeddings=("trait", lambda x: ";".join(list(x)[:12])),
        )
        .reset_index()
    )
    cluster_summary["cluster_type"] = np.where(cluster_summary["n_embeddings"] > 1, "multi_trait", "singleton")

    # Binary severity classification based only on severity associations.
    # This is intentionally transparent and can be tuned.
    cluster_summary["severity_related"] = (
        (cluster_summary["median_max_abs_severity_r"] >= 0.4)
        | (cluster_summary["fraction_embeddings_r_ge_0_4"] >= 0.5)
        | (cluster_summary["max_abs_severity_r"] >= 0.5)
    )
    cluster_summary["severity_class"] = np.where(
        cluster_summary["severity_related"], "severity_correlated", "not_severity_correlated"
    )

    cluster_summary.to_csv(OUT_DIR / "cluster025_severity_candidate_summary.csv", index=False)

    rows = []
    for unit_filter_name, frame in [
        ("all_cluster_labels_including_singletons", cluster_summary),
        ("multi_trait_clusters_only", cluster_summary.loc[cluster_summary["cluster_type"].eq("multi_trait")]),
        ("singletons_only", cluster_summary.loc[cluster_summary["cluster_type"].eq("singleton")]),
    ]:
        for candidate_col in [
            "has_validated_locus",
            "has_validated_or_proposed_locus",
            "has_any_reported_candidate_locus",
        ]:
            sev = frame["severity_related"]
            cand = frame[candidate_col]
            a = int((sev & cand).sum())
            b = int((sev & ~cand).sum())
            c = int((~sev & cand).sum())
            d = int((~sev & ~cand).sum())
            fisher_or, fisher_p = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append(
                {
                    "unit_set": unit_filter_name,
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
    enrichment.to_csv(OUT_DIR / "severity_related_cluster_candidate_enrichment.csv", index=False)

    top = cluster_summary.sort_values(
        ["severity_related", "median_max_abs_severity_r", "n_embeddings"],
        ascending=[False, False, False],
    ).head(30)

    report = [
        "# Cluster Disease Classification From Severity Correlations",
        "",
        "Severity-related classification uses only cluster member correlations with human score and ExG percent-unhealthy area.",
        "A cluster/singleton is called severity-correlated if median max(|r_human|, |r_ExG|) >= 0.4, or at least half of members have max |r| >= 0.4, or any member has max |r| >= 0.5.",
        "",
        "Candidate loci are mapped from the manuscript's reported known/proposed/new candidate marker tables to merged full-MLM GWAS loci.",
        "",
        "## Counts",
        cluster_summary.groupby(["cluster_type", "severity_class"]).size().reset_index(name="n").to_markdown(index=False),
        "",
        "## Candidate Enrichment",
        enrichment.to_markdown(index=False),
        "",
        "## Top Severity-Correlated Cluster Labels",
        top[
            [
                "cluster025",
                "cluster_type",
                "n_embeddings",
                "median_max_abs_severity_r",
                "max_abs_severity_r",
                "fraction_embeddings_r_ge_0_4",
                "has_validated_locus",
                "has_validated_or_proposed_locus",
                "has_any_reported_candidate_locus",
                "reported_candidate_genes",
                "example_embeddings",
            ]
        ].to_markdown(index=False),
        "",
    ]
    (OUT_DIR / "cluster_disease_classification_report.md").write_text("\n".join(report))

    print("\n".join(report))
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
