#!/usr/bin/env python3
"""Analyses supporting reframing around symptom-specific disease genetics.

This script uses existing genotype-level BLUEs, RF feature importances, GWAS
hits, and significant-marker genotype calls. It produces:

1. Correlations between high-feature-importance SAM3 embeddings and
   conventional single-axis severity metrics.
2. Marker effects on associated embeddings before and after conditioning on
   human disease score and/or ExG percent-unhealthy area.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_ROOT = ROOT.parent / "LaTeXManuscript"
OUT_DIR = ROOT / "output" / "reframing_results"


def embedding_label_to_column(label: str) -> str:
    match = re.fullmatch(r"\s*(\d+)\s+\((Mean|SD)\)\s*", label)
    if not match:
        raise ValueError(f"Could not parse embedding label: {label!r}")
    emb_num, stat = match.groups()
    return f"embedding_{'mean' if stat == 'Mean' else 'std'}_{emb_num}"


def feature_to_embedding_column(feature: int) -> str:
    if feature < 1024:
        return f"embedding_mean_{feature}"
    return f"embedding_std_{feature - 1024}"


def normalize_genotype(value: object) -> str:
    return str(value).replace(" ", "").strip()


def read_sig_marker_genotypes(vcf_path: Path) -> pd.DataFrame:
    samples: list[str] | None = None
    marker_rows: list[dict[str, object]] = []
    with vcf_path.open() as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                continue
            if line.startswith("#"):
                continue
            if samples is None:
                raise ValueError("VCF header with samples was not found.")
            parts = line.rstrip("\n").split("\t")
            chrom, pos, _id, ref, alt = parts[:5]
            marker = f"{chrom}:{pos}:{ref}:{alt}"
            calls = parts[9:]
            row: dict[str, object] = {"marker": marker}
            for sample, call in zip(samples, calls):
                gt = call.split(":", 1)[0]
                if gt in {"0/0", "0|0"}:
                    row[normalize_genotype(sample)] = 0
                elif gt in {"1/1", "1|1"}:
                    row[normalize_genotype(sample)] = 1
                else:
                    row[normalize_genotype(sample)] = np.nan
            marker_rows.append(row)
    return pd.DataFrame(marker_rows).set_index("marker")


def fit_marker_model(df: pd.DataFrame, y: str, covariates: list[str]) -> dict[str, object]:
    cols = ["marker_dosage", y] + covariates
    dat = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    counts = dat["marker_dosage"].value_counts()
    if dat.shape[0] < 20 or counts.get(0, 0) < 5 or counts.get(1, 0) < 5:
        return {
            "n": dat.shape[0],
            "n_ref": int(counts.get(0, 0)),
            "n_alt": int(counts.get(1, 0)),
            "marker_effect": np.nan,
            "marker_p": np.nan,
            "r2": np.nan,
        }
    x = sm.add_constant(dat[["marker_dosage"] + covariates], has_constant="add")
    model = sm.OLS(dat[y], x).fit()
    return {
        "n": dat.shape[0],
        "n_ref": int(counts.get(0, 0)),
        "n_alt": int(counts.get(1, 0)),
        "marker_effect": model.params["marker_dosage"],
        "marker_p": model.pvalues["marker_dosage"],
        "r2": model.rsquared,
    }


def bh_qvalues(pvalues: pd.Series) -> pd.Series:
    p = pvalues.astype(float)
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    m = valid.shape[0]
    if m == 0:
        return q
    ranked = valid * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked.iloc[::-1]).iloc[::-1].clip(upper=1)
    q.loc[adjusted.index] = adjusted
    return q


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    blues = pd.read_csv(ROOT / "data" / "blues_all.csv")
    blues["genotype_norm"] = blues["genotype"].map(normalize_genotype)
    ne = blues.loc[blues["location"] == "NE"].copy()

    # Analysis 2: high-FI embedding correlations with conventional metrics.
    fi = pd.read_csv(ROOT / "data" / "rf" / "sam3_human_scores_embedding_feature_importances_rf.csv")
    fi_values = fi.drop(columns=["Unnamed: 0"], errors="ignore")
    mean_fi = fi_values.mean(axis=0)
    high_fi_features = mean_fi[mean_fi > 0.003].sort_values(ascending=False)
    high_fi_cols = [feature_to_embedding_column(int(feature)) for feature in high_fi_features.index]

    corr_rows = []
    for col in high_fi_cols:
        row = {
            "embedding": col,
            "mean_feature_importance": high_fi_features[str(int(col.rsplit("_", 1)[1]) + (1024 if "_std_" in col else 0))]
            if False
            else np.nan,
        }
        for metric in ["human_score", "percentUnhealthy"]:
            dat = ne[[col, metric]].dropna()
            row[f"spearman_r_{metric}"] = dat[col].corr(dat[metric], method="spearman")
            row[f"spearman_r2_{metric}"] = row[f"spearman_r_{metric}"] ** 2
            row[f"n_{metric}"] = dat.shape[0]
        corr_rows.append(row)
    corr = pd.DataFrame(corr_rows)
    corr["mean_feature_importance"] = [
        high_fi_features.iloc[i] for i in range(len(high_fi_cols))
    ]
    corr.to_csv(OUT_DIR / "high_fi_embedding_severity_correlations.csv", index=False)

    emb_corr = ne[high_fi_cols].corr(method="spearman")
    abs_vals = emb_corr.where(np.triu(np.ones(emb_corr.shape), k=1).astype(bool)).stack().abs()
    embedding_corr_summary = pd.DataFrame(
        {
            "n_embeddings": [len(high_fi_cols)],
            "min_abs_pairwise_spearman_r": [abs_vals.min()],
            "median_abs_pairwise_spearman_r": [abs_vals.median()],
            "max_abs_pairwise_spearman_r": [abs_vals.max()],
            "n_pairwise_abs_r_lt_0_3": [(abs_vals < 0.3).sum()],
            "n_pairwise_abs_r_lt_0_5": [(abs_vals < 0.5).sum()],
            "n_pairs": [abs_vals.shape[0]],
        }
    )
    embedding_corr_summary.to_csv(OUT_DIR / "high_fi_embedding_pairwise_correlation_summary.csv", index=False)

    severity_summary = pd.DataFrame(
        {
            "metric": [
                "abs Spearman r with human_score",
                "abs Spearman r with percentUnhealthy",
                "Spearman r2 with human_score",
                "Spearman r2 with percentUnhealthy",
            ],
            "min": [
                corr["spearman_r_human_score"].abs().min(),
                corr["spearman_r_percentUnhealthy"].abs().min(),
                corr["spearman_r2_human_score"].min(),
                corr["spearman_r2_percentUnhealthy"].min(),
            ],
            "median": [
                corr["spearman_r_human_score"].abs().median(),
                corr["spearman_r_percentUnhealthy"].abs().median(),
                corr["spearman_r2_human_score"].median(),
                corr["spearman_r2_percentUnhealthy"].median(),
            ],
            "max": [
                corr["spearman_r_human_score"].abs().max(),
                corr["spearman_r_percentUnhealthy"].abs().max(),
                corr["spearman_r2_human_score"].max(),
                corr["spearman_r2_percentUnhealthy"].max(),
            ],
        }
    )
    severity_summary.to_csv(OUT_DIR / "high_fi_embedding_severity_correlation_summary.csv", index=False)

    # Analysis 1: marker effects after conditioning on conventional severity.
    hits = pd.read_csv(ROOT / "figures" / "supplemental" / "all_sig_hits_embeddings.csv")
    hits["embedding_col"] = hits["Embedding"].map(embedding_label_to_column)
    vcf = read_sig_marker_genotypes(MANUSCRIPT_ROOT / "fixfig4" / "output" / "all_sig_snps.recode.vcf")

    ne_indexed = ne.set_index("genotype_norm")
    marker_rows = []
    missing_markers = sorted(set(hits["Marker"]) - set(vcf.index))
    for hit in hits.itertuples(index=False):
        if hit.Marker not in vcf.index:
            continue
        gt = vcf.loc[hit.Marker].rename("marker_dosage").to_frame()
        dat = ne_indexed.join(gt, how="inner")
        for model_name, covars in [
            ("marker_only", []),
            ("condition_human_score", ["human_score"]),
            ("condition_percentUnhealthy", ["percentUnhealthy"]),
            ("condition_both", ["human_score", "percentUnhealthy"]),
        ]:
            result = fit_marker_model(dat, hit.embedding_col, covars)
            marker_rows.append(
                {
                    "marker": hit.Marker,
                    "embedding": hit.Embedding,
                    "embedding_col": hit.embedding_col,
                    "rmip": hit.RMIP,
                    "model": model_name,
                    "covariates": "+".join(covars) if covars else "none",
                    **result,
                }
            )

    marker_effects = pd.DataFrame(marker_rows)
    marker_effects.to_csv(OUT_DIR / "marker_embedding_effects_conditioned_on_severity.csv", index=False)
    pd.DataFrame({"missing_marker": missing_markers}).to_csv(
        OUT_DIR / "marker_effect_conditioning_missing_genotypes.csv", index=False
    )

    pivot_p = marker_effects.pivot_table(
        index=["marker", "embedding", "embedding_col", "rmip"],
        columns="model",
        values="marker_p",
        aggfunc="first",
    ).reset_index()
    pivot_p["available_conditioned_both"] = pivot_p["condition_both"].notna()
    pivot_p["marker_only_sig_0_05"] = pivot_p["marker_only"] < 0.05
    pivot_p["condition_human_sig_0_05"] = pivot_p["condition_human_score"] < 0.05
    pivot_p["condition_pct_sig_0_05"] = pivot_p["condition_percentUnhealthy"] < 0.05
    pivot_p["condition_both_sig_0_05"] = pivot_p["condition_both"] < 0.05
    for model_name in [
        "marker_only",
        "condition_human_score",
        "condition_percentUnhealthy",
        "condition_both",
    ]:
        pivot_p[f"{model_name}_q"] = bh_qvalues(pivot_p[model_name])
        pivot_p[f"{model_name}_q_lt_0_05"] = pivot_p[f"{model_name}_q"] < 0.05
    pivot_p.to_csv(OUT_DIR / "marker_embedding_conditioned_pvalue_summary.csv", index=False)

    count_summary = pd.DataFrame(
        {
            "quantity": [
                "rmip_supported_marker_embedding_associations",
                "associations_with_marker_genotypes_available",
                "unique_markers_in_hits",
                "unique_markers_with_genotypes_available",
                "unique_markers_missing_genotypes",
                "associations_marker_only_p_lt_0_05",
                "associations_condition_human_p_lt_0_05",
                "associations_condition_pct_p_lt_0_05",
                "associations_condition_both_p_lt_0_05",
                "associations_marker_only_q_lt_0_05",
                "associations_condition_human_q_lt_0_05",
                "associations_condition_pct_q_lt_0_05",
                "associations_condition_both_q_lt_0_05",
            ],
            "value": [
                hits.shape[0],
                pivot_p.shape[0],
                hits["Marker"].nunique(),
                hits.loc[hits["Marker"].isin(vcf.index), "Marker"].nunique(),
                len(missing_markers),
                int(pivot_p["marker_only_sig_0_05"].sum()),
                int(pivot_p["condition_human_sig_0_05"].sum()),
                int(pivot_p["condition_pct_sig_0_05"].sum()),
                int(pivot_p["condition_both_sig_0_05"].sum()),
                int(pivot_p["marker_only_q_lt_0_05"].sum()),
                int(pivot_p["condition_human_score_q_lt_0_05"].sum()),
                int(pivot_p["condition_percentUnhealthy_q_lt_0_05"].sum()),
                int(pivot_p["condition_both_q_lt_0_05"].sum()),
            ],
        }
    )
    count_summary.to_csv(OUT_DIR / "marker_embedding_conditioned_count_summary.csv", index=False)

    print("Wrote outputs to", OUT_DIR)
    print(count_summary.to_string(index=False))
    print()
    print(severity_summary.to_string(index=False))
    print()
    print(embedding_corr_summary.to_string(index=False))


if __name__ == "__main__":
    main()
