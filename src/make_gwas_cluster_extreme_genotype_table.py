#!/usr/bin/env python3
"""Select genotype examples from both tails of each SAM3 GWAS-hit group."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
CLASS_DIR = ROOT / "output" / "reframing_results" / "cluster_disease_classification"
OUT_DIR = ROOT / "output" / "reframing_results" / "gwas_cluster_extreme_genotypes"


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    sd = np.nanstd(values, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.full_like(values, np.nan, dtype=float)
    return (values - mean) / sd


def first_pc_or_singleton(matrix: pd.DataFrame) -> np.ndarray:
    z = matrix.apply(lambda col: zscore(col.to_numpy()), axis=0).to_numpy()
    if z.shape[1] == 1:
        return z[:, 0]

    complete = np.isfinite(z).all(axis=1)
    axis = np.full(z.shape[0], np.nan, dtype=float)
    axis[complete] = PCA(n_components=1).fit_transform(z[complete, :]).ravel()
    return axis


def spearman_with_anchor(axis: np.ndarray, anchor: np.ndarray) -> float:
    ok = np.isfinite(axis) & np.isfinite(anchor)
    if ok.sum() < 3:
        return np.nan
    return float(spearmanr(axis[ok], anchor[ok]).statistic)


def summarize_image_metadata(image_metadata: pd.DataFrame) -> pd.DataFrame:
    ne = image_metadata.loc[
        image_metadata["location"].eq("NE") & ~image_metadata["excluded"].fillna(False)
    ].copy()
    rows = []
    for genotype, group in ne.groupby("genotype", dropna=True):
        paths = group["image_path"].dropna().astype(str).head(5).tolist()
        rows.append(
            {
                "genotype": genotype,
                "n_ne_unexcluded_images": int(len(group)),
                "ne_plot_numbers": ";".join(map(str, sorted(group["plotNumber"].dropna().astype(int).unique()))),
                "example_ne_image_paths": ";".join(paths),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cluster_summary = pd.read_csv(CLASS_DIR / "cluster025_severity_candidate_summary.csv")
    trait_table = pd.read_csv(CLASS_DIR / "embedding_cluster_severity_candidate_table.csv")
    blues = pd.read_csv(ROOT / "data" / "blues_all.csv")
    ne = blues.loc[blues["location"].eq("NE")].reset_index(drop=True)

    image_summary = summarize_image_metadata(pd.read_csv(ROOT / "data" / "image_metadata.csv"))

    human_z = zscore(ne["human_score"].to_numpy())
    exg_z = zscore(ne["percentUnhealthy"].to_numpy())
    severity_anchor = human_z + exg_z

    representatives = []
    axis_metadata = []
    n_per_tail = 5

    for row in cluster_summary.sort_values("cluster025").itertuples(index=False):
        cluster_id = int(row.cluster025)
        traits = trait_table.loc[trait_table["cluster025"].eq(cluster_id), "trait"].tolist()
        missing = [trait for trait in traits if trait not in ne.columns]
        if missing:
            raise ValueError(f"Cluster {cluster_id} has traits not present in data/blues_all.csv: {missing}")

        axis = first_pc_or_singleton(ne[traits])
        rho = spearman_with_anchor(axis, severity_anchor)
        orientation = "positive_axis_aligned_to_higher_human_exg_severity"
        if np.isfinite(rho) and rho < 0:
            axis = -axis
            rho = -rho

        axis_name = f"cluster025_{cluster_id}"
        group_meta = {
            "cluster025": cluster_id,
            "axis_id": axis_name,
            "severity_related": bool(row.severity_related),
            "severity_class": row.severity_class,
            "cluster_type": row.cluster_type,
            "n_embeddings": int(row.n_embeddings),
            "n_loci": int(row.n_loci),
            "has_validated_locus": bool(row.has_validated_locus),
            "has_validated_or_proposed_locus": bool(row.has_validated_or_proposed_locus),
            "median_max_abs_severity_r": row.median_max_abs_severity_r,
            "max_abs_severity_r": row.max_abs_severity_r,
            "axis_spearman_abs_r_with_human_exg_anchor": rho,
            "axis_orientation": orientation,
            "example_embeddings": row.example_embeddings,
            "reported_candidate_genes": row.reported_candidate_genes,
        }
        axis_metadata.append(group_meta)

        scored = pd.DataFrame(
            {
                "genotype": ne["genotype"],
                "axis_value": axis,
                "human_score_blue": ne["human_score"],
                "percent_unhealthy_blue": ne["percentUnhealthy"],
            }
        ).dropna(subset=["axis_value"])

        for tail, ascending in [("low_axis_tail", True), ("high_axis_tail", False)]:
            tail_df = scored.sort_values("axis_value", ascending=ascending).head(n_per_tail).copy()
            for rank, sample in enumerate(tail_df.itertuples(index=False), start=1):
                representatives.append(
                    {
                        **group_meta,
                        "axis_tail": tail,
                        "tail_rank": rank,
                        "genotype": sample.genotype,
                        "axis_value": sample.axis_value,
                        "human_score_blue": sample.human_score_blue,
                        "percent_unhealthy_blue": sample.percent_unhealthy_blue,
                    }
                )

    representative_table = pd.DataFrame(representatives).merge(image_summary, on="genotype", how="left")
    metadata_table = pd.DataFrame(axis_metadata)

    representative_table.to_csv(OUT_DIR / "gwas_cluster_extreme_genotypes_top5_each_tail.csv", index=False)
    metadata_table.to_csv(OUT_DIR / "gwas_cluster_axis_metadata.csv", index=False)

    top1 = representative_table.loc[representative_table["tail_rank"].eq(1)].copy()
    top1.to_csv(OUT_DIR / "gwas_cluster_extreme_genotypes_top1_each_tail.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "n_gwas_hit_groups": int(cluster_summary.shape[0]),
                "n_severity_related_groups": int(cluster_summary["severity_related"].sum()),
                "n_not_severity_related_groups": int((~cluster_summary["severity_related"]).sum()),
                "n_representative_rows_top5": int(representative_table.shape[0]),
                "n_representative_rows_top1": int(top1.shape[0]),
            }
        ]
    )
    summary.to_csv(OUT_DIR / "gwas_cluster_extreme_genotype_summary.csv", index=False)
    with pd.ExcelWriter(OUT_DIR / "gwas_cluster_extreme_genotypes.xlsx") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        top1.to_excel(writer, sheet_name="top1_each_tail", index=False)
        representative_table.to_excel(writer, sheet_name="top5_each_tail", index=False)
        metadata_table.to_excel(writer, sheet_name="axis_metadata", index=False)

    print(summary.to_string(index=False))
    print(f"Wrote {representative_table.shape[0]} rows to {OUT_DIR / 'gwas_cluster_extreme_genotypes_top5_each_tail.csv'}")
    print(f"Wrote top-1 table to {OUT_DIR / 'gwas_cluster_extreme_genotypes_top1_each_tail.csv'}")
    print(f"Wrote workbook to {OUT_DIR / 'gwas_cluster_extreme_genotypes.xlsx'}")


if __name__ == "__main__":
    main()
