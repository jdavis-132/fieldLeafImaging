#!/usr/bin/env python3
"""Summarize agreement between individual human disease score raters."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[1]
MANUAL_DIR = ROOT / "data" / "manual"
OUT_DIR = ROOT / "output" / "reframing_results" / "rater_agreement"


def exg_leaf_name_to_jpg(image_name: str) -> str:
    image_name = re.sub(r"^device\d+_", "", str(image_name))
    return re.sub(r"_leaf\.png$", ".jpg", image_name)


def correlation_metrics(x: pd.Series, y: pd.Series) -> dict[str, float]:
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def summarize_pairs(pairs: pd.DataFrame, label: str) -> dict[str, float | str | int]:
    diff = pairs["Libia"] - pairs["Ruben"]
    abs_diff = diff.abs()
    libia_ordinal = (pairs["Libia"] * 2).round().astype(int)
    ruben_ordinal = (pairs["Ruben"] * 2).round().astype(int)
    metrics = {
        "comparison": label,
        "n_images": len(pairs),
        "libia_mean": pairs["Libia"].mean(),
        "ruben_mean": pairs["Ruben"].mean(),
        "mean_score_mean": pairs["mean_human_score"].mean(),
        "mean_signed_difference_libia_minus_ruben": diff.mean(),
        "mean_absolute_difference": abs_diff.mean(),
        "median_absolute_difference": abs_diff.median(),
        "exact_agreement_fraction": (abs_diff == 0).mean(),
        "within_0_5_fraction": (abs_diff <= 0.5).mean(),
        "within_1_0_fraction": (abs_diff <= 1.0).mean(),
        "greater_than_1_0_fraction": (abs_diff > 1.0).mean(),
        "linear_weighted_kappa": cohen_kappa_score(
            libia_ordinal, ruben_ordinal, weights="linear"
        ),
        "quadratic_weighted_kappa": cohen_kappa_score(
            libia_ordinal, ruben_ordinal, weights="quadratic"
        ),
    }
    metrics.update(correlation_metrics(pairs["Libia"], pairs["Ruben"]))
    return metrics


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(MANUAL_DIR / "all_image_scores.csv")
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    scores = scores.dropna(subset=["score", "image", "username"])

    pairs = (
        scores.pivot_table(
            index=["project", "image"],
            columns="username",
            values="score",
            aggfunc="first",
        )
        .reset_index()
        .dropna(subset=["Libia", "Ruben"])
    )
    metadata = (
        scores.sort_values(["project", "image"])
        .drop_duplicates(["project", "image"])[
            ["project", "image", "plotNumber", "genotype", "block"]
        ]
    )
    pairs = pairs.merge(metadata, on=["project", "image"], how="left")
    pairs["mean_human_score"] = pairs[["Libia", "Ruben"]].mean(axis=1)
    pairs["score_difference_libia_minus_ruben"] = pairs["Libia"] - pairs["Ruben"]
    pairs["absolute_difference"] = pairs["score_difference_libia_minus_ruben"].abs()

    summary_rows = [summarize_pairs(pairs, "all")]
    for project, project_pairs in pairs.groupby("project"):
        summary_rows.append(summarize_pairs(project_pairs, project))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "rater_agreement_summary.csv", index=False)

    diff_dist = (
        pairs.groupby("absolute_difference")
        .size()
        .reset_index(name="n_images")
        .assign(fraction=lambda x: x["n_images"] / len(pairs))
    )
    diff_dist.to_csv(OUT_DIR / "absolute_difference_distribution.csv", index=False)

    signed_diff_dist = (
        pairs.groupby("score_difference_libia_minus_ruben")
        .size()
        .reset_index(name="n_images")
        .assign(fraction=lambda x: x["n_images"] / len(pairs))
    )
    signed_diff_dist.to_csv(OUT_DIR / "signed_difference_distribution.csv", index=False)

    cross_tab = pd.crosstab(
        pairs["Libia"],
        pairs["Ruben"],
        rownames=["Libia"],
        colnames=["Ruben"],
        dropna=False,
    )
    cross_tab.to_csv(OUT_DIR / "rater_score_crosstab.csv")

    score_bin_dist = (
        pairs.groupby("mean_human_score")
        .size()
        .reset_index(name="n_images")
        .assign(fraction=lambda x: x["n_images"] / len(pairs))
    )
    score_bin_dist.to_csv(OUT_DIR / "mean_human_score_bin_distribution.csv", index=False)

    exg = pd.read_csv(MANUAL_DIR / "ne2025_exg_percent_disease_area_updated.csv")
    exg["image"] = exg["image_name"].map(exg_leaf_name_to_jpg)
    exg = exg.rename(columns={"score_average": "exg_percent_unhealthy"})
    exg["exg_percent_unhealthy"] = pd.to_numeric(
        exg["exg_percent_unhealthy"], errors="coerce"
    )
    exg = exg.dropna(subset=["image", "exg_percent_unhealthy"])

    pairs_exg = pairs.merge(
        exg[["image", "leaf_pixels", "exg_percent_unhealthy"]],
        on="image",
        how="inner",
    )
    pairs_exg.to_csv(OUT_DIR / "paired_rater_scores_with_exg.csv", index=False)

    exg_corr = {
        "comparison": "mean_human_score_vs_exg_percent_unhealthy",
        "n_images": len(pairs_exg),
    }
    exg_corr.update(
        correlation_metrics(
            pairs_exg["mean_human_score"], pairs_exg["exg_percent_unhealthy"]
        )
    )
    pd.DataFrame([exg_corr]).to_csv(OUT_DIR / "human_exg_correlation.csv", index=False)

    exg_by_human_bin = (
        pairs_exg.groupby("mean_human_score")
        .agg(
            n_images=("exg_percent_unhealthy", "size"),
            exg_mean=("exg_percent_unhealthy", "mean"),
            exg_sd=("exg_percent_unhealthy", "std"),
            exg_min=("exg_percent_unhealthy", "min"),
            exg_q10=("exg_percent_unhealthy", lambda x: x.quantile(0.10)),
            exg_q25=("exg_percent_unhealthy", lambda x: x.quantile(0.25)),
            exg_median=("exg_percent_unhealthy", "median"),
            exg_q75=("exg_percent_unhealthy", lambda x: x.quantile(0.75)),
            exg_q90=("exg_percent_unhealthy", lambda x: x.quantile(0.90)),
            exg_max=("exg_percent_unhealthy", "max"),
        )
        .reset_index()
    )
    exg_by_human_bin["exg_iqr_width"] = (
        exg_by_human_bin["exg_q75"] - exg_by_human_bin["exg_q25"]
    )
    exg_by_human_bin["exg_10_90_width"] = (
        exg_by_human_bin["exg_q90"] - exg_by_human_bin["exg_q10"]
    )
    exg_by_human_bin.to_csv(OUT_DIR / "exg_spread_by_mean_human_score.csv", index=False)

    large_bins = exg_by_human_bin[exg_by_human_bin["n_images"] >= 20].copy()
    collapse_summary = {
        "n_paired_human_images": len(pairs),
        "n_exg_matched_images": len(pairs_exg),
        "n_possible_individual_score_values_used": scores["score"].nunique(),
        "n_mean_human_score_bins_used": pairs["mean_human_score"].nunique(),
        "most_common_mean_score": score_bin_dist.sort_values(
            "n_images", ascending=False
        ).iloc[0]["mean_human_score"],
        "most_common_mean_score_fraction": score_bin_dist["fraction"].max(),
        "n_mean_score_bins_with_at_least_20_images": len(large_bins),
        "median_iqr_width_exg_within_large_mean_score_bins": large_bins[
            "exg_iqr_width"
        ].median(),
        "median_10_90_width_exg_within_large_mean_score_bins": large_bins[
            "exg_10_90_width"
        ].median(),
        "human_exg_spearman_r": exg_corr["spearman_r"],
        "human_exg_spearman_r2": exg_corr["spearman_r"] ** 2,
    }
    pd.DataFrame([collapse_summary]).to_csv(
        OUT_DIR / "ordinal_collapse_summary.csv", index=False
    )

    lines = [
        "# Human Rater Agreement Summary",
        "",
        f"Paired human scores were available for {len(pairs):,} images "
        f"({', '.join(f'{k}: {v}' for k, v in pairs.groupby('project').size().items())}).",
        "",
        "## Agreement",
    ]
    all_summary = summary.loc[summary["comparison"] == "all"].iloc[0]
    lines.extend(
        [
            f"- Spearman r between raters: {all_summary['spearman_r']:.3f}; "
            f"Pearson r: {all_summary['pearson_r']:.3f}.",
            f"- Quadratic weighted kappa: {all_summary['quadratic_weighted_kappa']:.3f}; "
            f"linear weighted kappa: {all_summary['linear_weighted_kappa']:.3f}.",
            f"- Exact agreement: {all_summary['exact_agreement_fraction']:.1%}; "
            f"within 0.5 score units: {all_summary['within_0_5_fraction']:.1%}; "
            f"within 1.0 score units: {all_summary['within_1_0_fraction']:.1%}; "
            f">1.0 score units apart: {all_summary['greater_than_1_0_fraction']:.1%}.",
            f"- Mean signed difference, Libia minus Ruben: "
            f"{all_summary['mean_signed_difference_libia_minus_ruben']:.3f} score units.",
            "",
            "## Ordinal Collapse",
            f"- The two-rater mean used {collapse_summary['n_mean_human_score_bins_used']} "
            "discrete score bins across all images.",
            f"- {collapse_summary['n_exg_matched_images']:,} scored images matched ExG disease-area "
            "estimates.",
            f"- Mean human score vs ExG Spearman r: {exg_corr['spearman_r']:.3f} "
            f"(r2 = {exg_corr['spearman_r'] ** 2:.3f}).",
            "- For human-score bins with at least 20 images, the median within-bin ExG IQR "
            f"was {collapse_summary['median_iqr_width_exg_within_large_mean_score_bins']:.2f}, "
            "and the median 10th-to-90th percentile width was "
            f"{collapse_summary['median_10_90_width_exg_within_large_mean_score_bins']:.2f}.",
            "",
            "Interpretation: the human scores are consistent enough to be useful, but not "
            "precise image phenotypes. They are ordinal, moderately noisy, and compress "
            "continuous image-level variation into a small number of bins.",
        ]
    )
    (OUT_DIR / "rater_agreement_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
