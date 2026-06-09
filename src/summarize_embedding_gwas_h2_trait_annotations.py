#!/usr/bin/env python3
"""Summarize embedding GWAS groups, heritability, and trait annotations."""

from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/reframing_results/embedding_gwas_h2_trait_annotations"
OUT.mkdir(parents=True, exist_ok=True)

SAM_H2_URL = (
    "https://raw.githubusercontent.com/jdavis-132/fieldLeafImaging/main/"
    "output/embedding_vp_ne.csv"
)


def zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean(axis=0, skipna=True)) / df.std(axis=0, ddof=1, skipna=True)


def pc1_axis(df: pd.DataFrame) -> pd.Series:
    z = zscore_frame(df).dropna(axis=0, how="any")
    if z.shape[1] == 1:
        return z.iloc[:, 0]
    u, s, _ = np.linalg.svd(z.to_numpy(), full_matrices=False)
    return pd.Series(u[:, 0] * s[0], index=z.index)


def clean_numeric_series(x: pd.Series) -> pd.Series:
    out = pd.to_numeric(x, errors="coerce")
    if out.notna().sum() < 20:
        return out
    lo, hi = out.quantile([0.01, 0.99])
    return out.clip(lo, hi)


def spearman_pair(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 20:
        return np.nan, np.nan, len(tmp)
    if tmp.iloc[:, 0].nunique() < 3 or tmp.iloc[:, 1].nunique() < 3:
        return np.nan, np.nan, len(tmp)
    rho, p = stats.spearmanr(tmp.iloc[:, 0], tmp.iloc[:, 1])
    return float(rho), float(p), int(len(tmp))


def load_sam_h2() -> pd.DataFrame:
    local = ROOT / "output/embedding_vp_ne.csv"
    if not local.exists():
        urllib.request.urlretrieve(SAM_H2_URL, local)
    raw = pd.read_csv(local)
    geno = raw.loc[raw["grp"].eq("genotype"), ["label", "pctVar"]].copy()
    geno = geno.rename(columns={"label": "trait"})
    geno["broad_sense_h2"] = geno["pctVar"] / 100.0
    return geno[["trait", "broad_sense_h2"]]


def h2_summary(prefix: str, h2: pd.DataFrame, gwas: pd.DataFrame) -> dict:
    merged = h2.merge(gwas, on="trait", how="inner")
    merged["has_hit"] = merged["n_significant_effective_bonferroni"] > 0
    merged.to_csv(OUT / f"{prefix}_trait_h2_gwas_status.csv", index=False)

    hit = merged.loc[merged["has_hit"], "broad_sense_h2"].dropna()
    no = merged.loc[~merged["has_hit"], "broad_sense_h2"].dropna()
    u = stats.mannwhitneyu(hit, no, alternative="greater")
    rho_hit, p_hit = stats.spearmanr(
        merged["broad_sense_h2"],
        merged["has_hit"].astype(int),
        nan_policy="omit",
    )
    rho_nsig, p_nsig = stats.spearmanr(
        merged["broad_sense_h2"],
        merged["n_significant_effective_bonferroni"],
        nan_policy="omit",
    )
    bins = pd.qcut(merged["broad_sense_h2"], 4, labels=False, duplicates="drop")
    by_bin = (
        merged.assign(h2_quartile=bins + 1)
        .groupby("h2_quartile", dropna=False)
        .agg(
            n_traits=("trait", "size"),
            median_h2=("broad_sense_h2", "median"),
            hit_traits=("has_hit", "sum"),
            hit_rate=("has_hit", "mean"),
        )
        .reset_index()
    )
    by_bin.to_csv(OUT / f"{prefix}_h2_quartile_hit_rates.csv", index=False)
    return {
        "prefix": prefix,
        "n_traits": int(len(merged)),
        "n_hit": int(merged["has_hit"].sum()),
        "median_h2_hit": float(hit.median()),
        "median_h2_no_hit": float(no.median()),
        "mean_h2_hit": float(hit.mean()),
        "mean_h2_no_hit": float(no.mean()),
        "mannwhitney_u": float(u.statistic),
        "mannwhitney_p_greater": float(u.pvalue),
        "spearman_h2_hit_indicator_rho": float(rho_hit),
        "spearman_h2_hit_indicator_p": float(p_hit),
        "spearman_h2_n_significant_rho": float(rho_nsig),
        "spearman_h2_n_significant_p": float(p_nsig),
        "quartiles": by_bin.to_dict(orient="records"),
    }


def load_trait_matrix() -> tuple[pd.DataFrame, dict[str, str]]:
    trait_frames = []
    categories = {}

    v2 = pd.read_csv(
        ROOT / "output/reframing_results/sorghum_trait_data_v2_gwas/sorghum_trait_data_v2_trait_env_blues.csv"
    ).set_index("genotype")
    meta = pd.read_csv(
        ROOT / "output/reframing_results/sorghum_trait_data_v2_gwas/sorghum_trait_data_v2_trait_env_blue_metadata.csv"
    )
    trait_defs = pd.read_csv(
        ROOT / "output/reframing_results/sorghum_trait_data_v2_input/sorghum_trait_data_v2/traits.tsv",
        sep="\t",
    )
    cat_by_name = trait_defs.set_index("canonical_name")["category"].to_dict()
    meta_cat = meta.set_index("trait")["canonical_name"].map(cat_by_name).fillna("other")
    categories.update(meta_cat.to_dict())
    trait_frames.append(v2)

    nd = pd.read_csv(
        ROOT / "output/reframing_results/ne_non_disease_trait_mlm_lrt/ne_non_disease_trait_blues.csv"
    ).set_index("genotype")
    nd = nd.rename(columns={c: f"NE2025_image__{c}" for c in nd.columns})
    categories.update(
        {
            "NE2025_image__days_to_flower": "phenology",
            "NE2025_image__leaf_pixels": "image_leaf_size",
            "NE2025_image__log_leaf_pixels": "image_leaf_size",
        }
    )
    trait_frames.append(nd)

    severity = pd.read_csv(ROOT / "data/blues_all.csv")
    severity = severity.loc[severity["location"].eq("NE"), ["genotype", "human_score", "percentUnhealthy"]]
    severity = severity.set_index("genotype").rename(
        columns={
            "human_score": "NE2025_image__human_score",
            "percentUnhealthy": "NE2025_image__percentUnhealthy",
        }
    )
    categories.update(
        {
            "NE2025_image__human_score": "disease_severity",
            "NE2025_image__percentUnhealthy": "disease_severity",
        }
    )
    trait_frames.append(severity)

    traits = pd.concat(trait_frames, axis=1)
    traits = traits.loc[:, ~traits.columns.duplicated()]
    for col in traits.columns:
        traits[col] = clean_numeric_series(traits[col])
    return traits, categories


def group_summary(prefix: str, clusters: pd.DataFrame, blues: pd.DataFrame, categories: dict[str, str]) -> pd.DataFrame:
    cluster_col = "cluster_jaccard_similarity_ge_0_25"
    members = clusters.loc[clusters[cluster_col].ge(0), ["trait", cluster_col]].copy()
    trait_cols = [c for c in blues.columns if c != "location"]
    emb = blues.loc[blues["location"].eq("NE"), trait_cols].set_index("genotype")
    traits, categories = load_trait_matrix()

    records = []
    corr_records = []
    for gid, sub in members.groupby(cluster_col):
        emb_names = [t for t in sub["trait"] if t in emb.columns]
        if not emb_names:
            continue
        mat = emb[emb_names]
        axis = pc1_axis(mat)

        within = np.nan
        if len(emb_names) > 1:
            corr = mat.corr(method="spearman").abs()
            vals = corr.to_numpy()[np.triu_indices_from(corr.to_numpy(), k=1)]
            within = float(np.nanmedian(vals))

        top_corrs = []
        for trait in traits.columns:
            rho, p, n = spearman_pair(axis, traits[trait])
            if math.isfinite(rho):
                top_corrs.append((trait, rho, p, n, categories.get(trait, "other")))
        top_corrs = sorted(top_corrs, key=lambda x: abs(x[1]), reverse=True)
        for rank, (trait, rho, p, n, category) in enumerate(top_corrs[:5], start=1):
            corr_records.append(
                {
                    "embedding_set": prefix,
                    "group_id": int(gid),
                    "rank": rank,
                    "trait": trait,
                    "category": category,
                    "rho": rho,
                    "p": p,
                    "n": n,
                    "n_embeddings": len(emb_names),
                    "median_within_group_abs_spearman": within,
                    "example_embeddings": ";".join(emb_names[:8]),
                }
            )
        field_corrs = [x for x in top_corrs if not x[0].startswith("NE2025_image__")]
        if top_corrs:
            top_trait, top_rho, top_p, top_n, top_cat = top_corrs[0]
        else:
            top_trait, top_rho, top_p, top_n, top_cat = None, np.nan, np.nan, 0, None
        if field_corrs:
            top_field_trait, top_field_rho, top_field_p, top_field_n, top_field_cat = field_corrs[0]
        else:
            top_field_trait, top_field_rho, top_field_p, top_field_n, top_field_cat = None, np.nan, np.nan, 0, None
        records.append(
            {
                "embedding_set": prefix,
                "group_id": int(gid),
                "n_embeddings": len(emb_names),
                "median_within_group_abs_spearman": within,
                "top_trait": top_trait,
                "top_trait_category": top_cat,
                "top_trait_rho": top_rho,
                "top_trait_p": top_p,
                "top_trait_n": top_n,
                "top_field_trait": top_field_trait,
                "top_field_trait_category": top_field_cat,
                "top_field_trait_rho": top_field_rho,
                "top_field_trait_p": top_field_p,
                "top_field_trait_n": top_field_n,
                "example_embeddings": ";".join(emb_names[:8]),
            }
        )

    group_df = pd.DataFrame(records).sort_values(["n_embeddings", "group_id"], ascending=[False, True])
    corr_df = pd.DataFrame(corr_records)
    group_df.to_csv(OUT / f"{prefix}_gwas_group_trait_annotations.csv", index=False)
    corr_df.to_csv(OUT / f"{prefix}_gwas_group_top_trait_correlations.csv", index=False)
    return group_df


def summarize_groups(df: pd.DataFrame) -> dict:
    multi = df[df["n_embeddings"] > 1]
    singleton = df[df["n_embeddings"] == 1]
    category_counts = (
        df.assign(abs_top_rho=df["top_trait_rho"].abs())
        .query("abs_top_rho >= 0.3")
        .groupby("top_trait_category")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )
    field_category_counts = (
        df.assign(abs_top_field_rho=df["top_field_trait_rho"].abs())
        .query("abs_top_field_rho >= 0.3")
        .groupby("top_field_trait_category")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )
    top_examples = (
        df.assign(abs_top_rho=df["top_trait_rho"].abs())
        .sort_values(["n_embeddings", "abs_top_rho"], ascending=[False, False])
        .head(8)
        .to_dict(orient="records")
    )
    top_field_examples = (
        df.assign(abs_top_field_rho=df["top_field_trait_rho"].abs())
        .sort_values(["abs_top_field_rho", "n_embeddings"], ascending=[False, False])
        .head(8)
        .to_dict(orient="records")
    )
    return {
        "n_groups": int(len(df)),
        "n_multi_trait_groups": int(len(multi)),
        "n_singletons": int(len(singleton)),
        "median_within_group_abs_spearman_multi": float(multi["median_within_group_abs_spearman"].median()),
        "groups_with_top_trait_abs_rho_ge_0_3": int((df["top_trait_rho"].abs() >= 0.3).sum()),
        "groups_with_top_trait_abs_rho_ge_0_4": int((df["top_trait_rho"].abs() >= 0.4).sum()),
        "groups_with_top_field_trait_abs_rho_ge_0_3": int((df["top_field_trait_rho"].abs() >= 0.3).sum()),
        "groups_with_top_field_trait_abs_rho_ge_0_4": int((df["top_field_trait_rho"].abs() >= 0.4).sum()),
        "top_trait_category_counts_abs_rho_ge_0_3": category_counts,
        "top_field_trait_category_counts_abs_rho_ge_0_3": field_category_counts,
        "top_examples": top_examples,
        "top_field_examples": top_field_examples,
    }


def main() -> None:
    sam_h2 = load_sam_h2()
    dino_h2 = pd.read_csv(ROOT / "output/dinov2_20260522_heritability/dinov2_20260522_heritability.csv")
    dino_h2 = dino_h2.loc[dino_h2["status"].eq("ok"), ["trait_blues_all_name", "broad_sense_h2"]]
    dino_h2 = dino_h2.rename(columns={"trait_blues_all_name": "trait"})

    sam_gwas = pd.read_csv(ROOT / "output/reframing_results/all_sam3_full_mlm_lrt/all_traits_summary.csv")
    dino_gwas = pd.read_csv(ROOT / "output/reframing_results/all_dinov2_20260522_full_mlm_lrt/all_traits_summary.csv")
    sam_h2_result = h2_summary("sam3", sam_h2, sam_gwas)
    dino_h2_result = h2_summary("dinov2", dino_h2, dino_gwas)

    _, categories = load_trait_matrix()
    sam_groups = group_summary(
        "sam3",
        pd.read_csv(ROOT / "output/reframing_results/all_sam3_full_mlm_lrt/embedding_gwas_signal_clusters.csv"),
        pd.read_csv(ROOT / "data/blues_all.csv"),
        categories,
    )
    dino_groups = group_summary(
        "dinov2",
        pd.read_csv(ROOT / "output/reframing_results/all_dinov2_20260522_full_mlm_lrt/embedding_gwas_signal_clusters.csv"),
        pd.read_csv(ROOT / "output/dinov2_20260522_blues/dinov2_20260522_blues_all_compatible.csv"),
        categories,
    )

    report = {
        "heritability": {"sam3": sam_h2_result, "dinov2": dino_h2_result},
        "groups": {"sam3": summarize_groups(sam_groups), "dinov2": summarize_groups(dino_groups)},
        "outputs": {
            "sam3_trait_h2_gwas_status": str(OUT / "sam3_trait_h2_gwas_status.csv"),
            "dinov2_trait_h2_gwas_status": str(OUT / "dinov2_trait_h2_gwas_status.csv"),
            "sam3_group_annotations": str(OUT / "sam3_gwas_group_trait_annotations.csv"),
            "dinov2_group_annotations": str(OUT / "dinov2_gwas_group_trait_annotations.csv"),
        },
    }
    (OUT / "embedding_gwas_h2_trait_annotation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    lines = ["# Embedding GWAS heritability and trait-annotation summary", ""]
    for key, res in report["heritability"].items():
        lines.append(f"## {key}")
        lines.append(
            f"- Traits with h2 and GWAS summaries: {res['n_traits']}; traits with significant GWAS hits: {res['n_hit']}."
        )
        lines.append(
            f"- Median h2 for hit traits: {res['median_h2_hit']:.3f}; median h2 for no-hit traits: {res['median_h2_no_hit']:.3f}; Mann-Whitney one-sided p = {res['mannwhitney_p_greater']:.3g}."
        )
        lines.append(
            f"- Spearman h2 vs hit indicator rho = {res['spearman_h2_hit_indicator_rho']:.3f}, p = {res['spearman_h2_hit_indicator_p']:.3g}; h2 vs number of significant markers rho = {res['spearman_h2_n_significant_rho']:.3f}, p = {res['spearman_h2_n_significant_p']:.3g}."
        )
        lines.append("")
    for key, res in report["groups"].items():
        lines.append(f"## {key} GWAS-signal groups")
        lines.append(
            f"- Groups at Jaccard >= 0.25: {res['n_groups']}; multi-trait groups: {res['n_multi_trait_groups']}; singletons: {res['n_singletons']}."
        )
        lines.append(
            f"- Median within-group abs Spearman correlation among multi-trait group embedding BLUEs: {res['median_within_group_abs_spearman_multi']:.3f}."
        )
        lines.append(
            f"- Groups with top non-image/scored-trait correlation |rho| >= 0.3: {res['groups_with_top_trait_abs_rho_ge_0_3']}; |rho| >= 0.4: {res['groups_with_top_trait_abs_rho_ge_0_4']}."
        )
        lines.append(f"- Top trait categories at |rho| >= 0.3: {res['top_trait_category_counts_abs_rho_ge_0_3']}.")
        lines.append(
            f"- Excluding NE image-derived traits, groups with top field-trait correlation |rho| >= 0.3: {res['groups_with_top_field_trait_abs_rho_ge_0_3']}; |rho| >= 0.4: {res['groups_with_top_field_trait_abs_rho_ge_0_4']}."
        )
        lines.append(f"- Top field-trait categories at |rho| >= 0.3: {res['top_field_trait_category_counts_abs_rho_ge_0_3']}.")
        lines.append("- Examples:")
        for ex in res["top_examples"][:5]:
            lines.append(
                f"  - group {ex['group_id']}, n={ex['n_embeddings']}, within |rho|={ex['median_within_group_abs_spearman']:.3f}, top trait {ex['top_trait']} ({ex['top_trait_category']}), rho={ex['top_trait_rho']:.3f}; examples {ex['example_embeddings']}"
            )
        lines.append("- Top field-trait examples:")
        for ex in res["top_field_examples"][:5]:
            lines.append(
                f"  - group {ex['group_id']}, n={ex['n_embeddings']}, top field trait {ex['top_field_trait']} ({ex['top_field_trait_category']}), rho={ex['top_field_trait_rho']:.3f}; examples {ex['example_embeddings']}"
            )
        lines.append("")
    (OUT / "embedding_gwas_h2_trait_annotation_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
