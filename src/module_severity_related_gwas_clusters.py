#!/usr/bin/env python3
"""Group severity-related GWAS cluster labels by embedding phenotype correlation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "reframing_results" / "severity_cluster_modules"
CLASS_DIR = ROOT / "output" / "reframing_results" / "cluster_disease_classification"


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / np.nanstd(x, ddof=1)


def cluster_axis(matrix: pd.DataFrame, severity_anchor: pd.Series) -> np.ndarray:
    z = matrix.apply(lambda col: zscore(col.to_numpy()), axis=0).to_numpy()
    if z.shape[1] == 1:
        axis = z[:, 0]
    else:
        axis = PCA(n_components=1).fit_transform(z).ravel()
    rho = spearmanr(axis, severity_anchor, nan_policy="omit").statistic
    if np.isfinite(rho) and rho < 0:
        axis = -axis
    return axis


def residualize(y: np.ndarray, covars: pd.DataFrame) -> np.ndarray:
    ok = np.isfinite(y) & np.isfinite(covars.to_numpy()).all(axis=1)
    out = np.full_like(y, np.nan, dtype=float)
    model = LinearRegression().fit(covars.loc[ok].to_numpy(dtype=float), y[ok])
    out[ok] = y[ok] - model.predict(covars.loc[ok].to_numpy(dtype=float))
    return out


def spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    out = np.eye(len(cols), dtype=float)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = spearmanr(df.iloc[:, i], df.iloc[:, j], nan_policy="omit").statistic
            out[i, j] = out[j, i] = rho
    return pd.DataFrame(out, index=cols, columns=cols)


def module_labels(corr: pd.DataFrame, similarity_threshold: float) -> pd.Series:
    # Treat strong positive or inverse relationships as the same module.
    dist = 1 - corr.abs().clip(upper=1)
    np.fill_diagonal(dist.values, 0)
    z = linkage(squareform(dist.values, checks=False), method="average")
    labels = fcluster(z, t=1 - similarity_threshold, criterion="distance")
    return pd.Series(labels, index=corr.index, name=f"module_abs_spearman_ge_{similarity_threshold:g}")


def summarize_modules(labels: pd.Series, corr: pd.DataFrame, metadata: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for module, members in labels.groupby(labels).groups.items():
        members = list(members)
        sub = corr.loc[members, members]
        vals = sub.where(~np.eye(len(members), dtype=bool)).stack().abs()
        md = metadata.set_index("axis_id").loc[members]
        rows.append(
            {
                "analysis": name,
                "module": int(module),
                "n_cluster_labels": len(members),
                "n_multitrait_labels": int(md["cluster_type"].eq("multi_trait").sum()),
                "n_singleton_labels": int(md["cluster_type"].eq("singleton").sum()),
                "median_abs_within_module_r": float(vals.median()) if not vals.empty else np.nan,
                "max_abs_within_module_r": float(vals.max()) if not vals.empty else np.nan,
                "candidate_supported_labels": int(md["has_validated_or_proposed_locus"].sum()),
                "example_axis_ids": ";".join(members[:12]),
                "example_embeddings": ";".join(md["example_embeddings"].astype(str).head(12)),
            }
        )
    return pd.DataFrame(rows).sort_values("n_cluster_labels", ascending=False)


def between_module_summary(labels: pd.Series, corr: pd.DataFrame) -> dict:
    pairs = []
    for i, a in enumerate(corr.index):
        for b in corr.index[i + 1 :]:
            if labels.loc[a] != labels.loc[b]:
                pairs.append(abs(corr.loc[a, b]))
    arr = np.asarray(pairs, dtype=float)
    return {
        "n_between_module_pairs": int(arr.size),
        "median_abs_between_module_r": float(np.nanmedian(arr)),
        "p90_abs_between_module_r": float(np.nanpercentile(arr, 90)),
        "fraction_abs_between_module_r_lt_0_3": float(np.nanmean(arr < 0.3)),
        "fraction_abs_between_module_r_lt_0_2": float(np.nanmean(arr < 0.2)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cluster_summary = pd.read_csv(CLASS_DIR / "cluster025_severity_candidate_summary.csv")
    trait_table = pd.read_csv(CLASS_DIR / "embedding_cluster_severity_candidate_table.csv")
    blues = pd.read_csv(ROOT / "data" / "blues_all.csv")
    ne = blues.loc[blues["location"].eq("NE")].copy()
    ne = ne.dropna(subset=["human_score", "percentUnhealthy"]).reset_index(drop=True)
    severity_anchor = pd.Series(
        zscore(ne["human_score"].to_numpy()) + zscore(ne["percentUnhealthy"].to_numpy()),
        index=ne.index,
    )

    severity_clusters = cluster_summary.loc[cluster_summary["severity_related"]].copy()
    axes = {}
    meta_rows = []
    for row in severity_clusters.itertuples(index=False):
        cluster_id = int(row.cluster025)
        traits = trait_table.loc[trait_table["cluster025"].eq(cluster_id), "trait"].tolist()
        axis_id = f"cluster025_{cluster_id}"
        axes[axis_id] = cluster_axis(ne[traits], severity_anchor)
        meta_rows.append(
            {
                "axis_id": axis_id,
                "cluster025": cluster_id,
                "n_embeddings": int(row.n_embeddings),
                "cluster_type": row.cluster_type,
                "median_max_abs_severity_r": row.median_max_abs_severity_r,
                "max_abs_severity_r": row.max_abs_severity_r,
                "has_validated_or_proposed_locus": bool(row.has_validated_or_proposed_locus),
                "has_validated_locus": bool(row.has_validated_locus),
                "example_embeddings": row.example_embeddings,
            }
        )

    axis_df = pd.DataFrame(axes)
    covars = ne[["human_score", "percentUnhealthy"]].apply(lambda col: zscore(col.to_numpy()), axis=0)
    residual_df = pd.DataFrame(
        {col: residualize(axis_df[col].to_numpy(), covars) for col in axis_df.columns},
        index=axis_df.index,
    )
    metadata = pd.DataFrame(meta_rows)
    metadata.to_csv(OUT_DIR / "severity_related_cluster_axis_metadata.csv", index=False)
    axis_df.to_csv(OUT_DIR / "severity_related_cluster_axes_raw.csv", index=False)
    residual_df.to_csv(OUT_DIR / "severity_related_cluster_axes_residualized_human_exg.csv", index=False)

    report_rows = []
    for name, data in [("raw", axis_df), ("residualized_human_exg", residual_df)]:
        corr = spearman_matrix(data)
        corr.to_csv(OUT_DIR / f"{name}_severity_cluster_axis_spearman.csv")
        for threshold in [0.5, 0.4, 0.3]:
            labels = module_labels(corr, threshold)
            label_df = labels.rename("module").reset_index().rename(columns={"index": "axis_id"})
            label_df["analysis"] = name
            label_df["similarity_threshold"] = threshold
            label_df.to_csv(OUT_DIR / f"{name}_modules_abs_spearman_ge_{threshold:g}.csv", index=False)
            module_summary = summarize_modules(labels, corr, metadata, name)
            module_summary["similarity_threshold"] = threshold
            module_summary.to_csv(OUT_DIR / f"{name}_module_summary_abs_spearman_ge_{threshold:g}.csv", index=False)
            between = between_module_summary(labels, corr)
            report_rows.append(
                {
                    "analysis": name,
                    "similarity_threshold": threshold,
                    "n_axes": len(labels),
                    "n_modules": int(labels.nunique()),
                    "n_multi_axis_modules": int((labels.value_counts() > 1).sum()),
                    "largest_module_size": int(labels.value_counts().max()),
                    **between,
                }
            )

    report = pd.DataFrame(report_rows)
    report.to_csv(OUT_DIR / "severity_cluster_module_summary.csv", index=False)
    print(report.to_string(index=False))
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
