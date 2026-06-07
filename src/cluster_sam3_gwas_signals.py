#!/usr/bin/env python3
"""Cluster SAM3 embeddings based on shared full-MLM GWAS signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "output" / "reframing_results" / "all_sam3_full_mlm_lrt"


def build_loci(sig: pd.DataFrame, window_bp: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sig.empty:
        empty_loci = pd.DataFrame(columns=["locus_id", "CHROM", "start", "end", "n_markers"])
        return sig.assign(locus_id=pd.Series(dtype=str)), empty_loci

    sig = sig.copy()
    sig["CHROM"] = sig["CHROM"].astype(str)
    sig["POS"] = sig["POS"].astype(int)
    sig = sig.sort_values(["CHROM", "POS", "p_value"]).reset_index(drop=True)

    loci = []
    assigned = []
    for chrom, chrom_df in sig.groupby("CHROM", sort=False):
        locus_idx = 0
        current_start = None
        current_end = None
        current_rows = []
        for idx, row in chrom_df.iterrows():
            pos = int(row["POS"])
            if current_start is None or pos - current_end > window_bp:
                if current_start is not None:
                    locus_id = f"{chrom}:{current_start}-{current_end}"
                    loci.append(
                        {
                            "locus_id": locus_id,
                            "CHROM": chrom,
                            "start": current_start,
                            "end": current_end,
                            "n_markers": len(current_rows),
                        }
                    )
                    assigned.extend((i, locus_id) for i in current_rows)
                locus_idx += 1
                current_start = pos
                current_end = pos
                current_rows = [idx]
            else:
                current_end = max(current_end, pos)
                current_rows.append(idx)

        if current_start is not None:
            locus_id = f"{chrom}:{current_start}-{current_end}"
            loci.append(
                {
                    "locus_id": locus_id,
                    "CHROM": chrom,
                    "start": current_start,
                    "end": current_end,
                    "n_markers": len(current_rows),
                }
            )
            assigned.extend((i, locus_id) for i in current_rows)

    assigned_df = pd.DataFrame(assigned, columns=["row_index", "locus_id"]).set_index("row_index")
    sig["locus_id"] = assigned_df.loc[sig.index, "locus_id"].to_numpy()
    loci_df = pd.DataFrame(loci)
    return sig, loci_df


def representative_trait_locus(sig_loci: pd.DataFrame) -> pd.DataFrame:
    if sig_loci.empty:
        return pd.DataFrame()
    ordered = sig_loci.sort_values("p_value")
    reps = ordered.groupby(["trait", "locus_id"], as_index=False).first()
    reps["neg_log10_p"] = -np.log10(reps["p_value"])
    reps["signed_neg_log10_p"] = np.sign(reps["effect"]) * reps["neg_log10_p"]
    return reps


def cluster_embeddings(reps: pd.DataFrame, all_traits: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if reps.empty:
        return (
            pd.DataFrame({"trait": all_traits, "cluster_jaccard_0_25": np.arange(1, len(all_traits) + 1)}),
            pd.DataFrame(),
        )

    binary = (
        reps.assign(present=1)
        .pivot_table(index="trait", columns="locus_id", values="present", fill_value=0, aggfunc="max")
        .reindex(all_traits, fill_value=0)
    )
    signal = (
        reps.pivot_table(
            index="trait",
            columns="locus_id",
            values="signed_neg_log10_p",
            fill_value=0,
            aggfunc="max",
        )
        .reindex(index=all_traits, columns=binary.columns, fill_value=0)
    )

    has_signal = binary.sum(axis=1).to_numpy() > 0
    cluster_df = pd.DataFrame(
        {
            "trait": all_traits,
            "n_significant_loci": binary.sum(axis=1).to_numpy(dtype=int),
            "cluster_jaccard_similarity_ge_0_25": -1,
            "cluster_jaccard_similarity_ge_0_50": -1,
        }
    )
    if has_signal.sum() >= 2:
        dist = pdist(binary.loc[has_signal].to_numpy(dtype=bool), metric="jaccard")
        z = linkage(dist, method="average")
        labels_025 = fcluster(z, t=0.75, criterion="distance")
        labels_050 = fcluster(z, t=0.50, criterion="distance")
        cluster_df.loc[has_signal, "cluster_jaccard_similarity_ge_0_25"] = labels_025
        cluster_df.loc[has_signal, "cluster_jaccard_similarity_ge_0_50"] = labels_050
    elif has_signal.sum() == 1:
        cluster_df.loc[has_signal, "cluster_jaccard_similarity_ge_0_25"] = 1
        cluster_df.loc[has_signal, "cluster_jaccard_similarity_ge_0_50"] = 1

    return cluster_df, signal


def summarize_clusters(cluster_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    clustered = cluster_df.loc[cluster_df[label_col] > 0]
    if clustered.empty:
        return pd.DataFrame(columns=[label_col, "n_traits", "median_significant_loci", "example_traits"])
    return (
        clustered.groupby(label_col)
        .agg(
            n_traits=("trait", "size"),
            median_significant_loci=("n_significant_loci", "median"),
            example_traits=("trait", lambda x: ";".join(list(x)[:8])),
        )
        .reset_index()
        .sort_values("n_traits", ascending=False)
    )


def write_report(
    out_dir: Path,
    metadata: dict,
    summary: pd.DataFrame,
    sig_loci: pd.DataFrame,
    loci: pd.DataFrame,
    cluster_df: pd.DataFrame,
    cluster_summary_025: pd.DataFrame,
    cluster_summary_050: pd.DataFrame,
) -> None:
    n_traits = int(summary.shape[0])
    n_traits_sig = int((summary["n_significant_effective_bonferroni"] > 0).sum())
    n_sig = int(summary["n_significant_effective_bonferroni"].sum())
    n_markers = int(sig_loci["MARKER"].nunique()) if not sig_loci.empty else 0
    n_loci = int(loci["locus_id"].nunique()) if not loci.empty else 0
    top_traits = summary.sort_values("n_significant_effective_bonferroni", ascending=False).head(15)

    lines = [
        "# All SAM3 Full MLM LRT Report",
        "",
        "## Model",
        f"- Samples: {metadata.get('n_samples')}",
        f"- Markers: {metadata.get('n_markers')}",
        f"- Traits: {n_traits}",
        f"- PCs: {metadata.get('n_pcs')}",
        "- Kinship: LOCO VanRaden",
        "- Association: PANICLE multi-trait LOCO MLM with LRT refinement",
        f"- Effective marker number: {metadata.get('effective_markers')}",
        f"- Effective Bonferroni threshold: {metadata.get('effective_bonferroni_threshold'):.3e}",
        "",
        "## Significant Signals",
        f"- Traits with at least one effective-threshold marker: {n_traits_sig}/{n_traits}",
        f"- Total significant marker-trait associations: {n_sig}",
        f"- Unique significant markers: {n_markers}",
        f"- Significant loci after window merging: {n_loci}",
        "",
        "## Top Traits By Significant Marker Count",
        top_traits[
            [
                "trait",
                "embedding_stat",
                "embedding_index",
                "n_significant_effective_bonferroni",
                "n_q_lt_0_05_within_trait",
                "min_p",
            ]
        ].to_markdown(index=False),
        "",
        "## GWAS Signal Clustering",
        (
            "- Clustering uses binary sharing of significant loci, with markers merged into loci "
            "by chromosome and position window before computing Jaccard distances."
        ),
        f"- Traits with any significant locus: {int((cluster_df['n_significant_loci'] > 0).sum())}/{n_traits}",
        (
            "- At Jaccard similarity >= 0.25: "
            f"{int((cluster_summary_025['n_traits'] > 1).sum())} multi-trait clusters; "
            f"largest size {int(cluster_summary_025['n_traits'].max()) if not cluster_summary_025.empty else 0}."
        ),
        (
            "- At Jaccard similarity >= 0.50: "
            f"{int((cluster_summary_050['n_traits'] > 1).sum())} multi-trait clusters; "
            f"largest size {int(cluster_summary_050['n_traits'].max()) if not cluster_summary_050.empty else 0}."
        ),
        "",
        "## Largest Clusters, Similarity >= 0.25",
        cluster_summary_025.head(20).to_markdown(index=False),
        "",
        "## Largest Clusters, Similarity >= 0.50",
        cluster_summary_050.head(20).to_markdown(index=False),
        "",
        "## Interpretation Placeholder",
        (
            "Use this section to interpret whether shared loci form coherent embedding modules. "
            "Large clusters support shared genetic control across embedding axes; smaller clusters "
            "or singleton traits support genetically distinct symptom/image axes."
        ),
        "",
    ]
    (out_dir / "all_sam3_full_mlm_lrt_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window-bp", type=int, default=200_000)
    args = parser.parse_args()

    out_dir = args.out_dir
    metadata = json.loads((out_dir / "run_metadata.json").read_text())
    summary = pd.read_csv(out_dir / "all_traits_summary.csv")
    sig = pd.read_csv(out_dir / "all_traits_significant_hits.csv")

    all_traits = summary.sort_values(["embedding_stat", "embedding_index"])["trait"].tolist()
    sig_loci, loci = build_loci(sig, args.window_bp)
    reps = representative_trait_locus(sig_loci)
    cluster_df, signal_matrix = cluster_embeddings(reps, all_traits)
    cluster_summary_025 = summarize_clusters(cluster_df, "cluster_jaccard_similarity_ge_0_25")
    cluster_summary_050 = summarize_clusters(cluster_df, "cluster_jaccard_similarity_ge_0_50")

    sig_loci.to_csv(out_dir / "all_traits_significant_hits_with_loci.csv", index=False)
    loci.to_csv(out_dir / "gwas_signal_loci.csv", index=False)
    reps.to_csv(out_dir / "trait_locus_representative_signals.csv", index=False)
    cluster_df.to_csv(out_dir / "embedding_gwas_signal_clusters.csv", index=False)
    cluster_summary_025.to_csv(out_dir / "embedding_gwas_signal_cluster_summary_jaccard025.csv", index=False)
    cluster_summary_050.to_csv(out_dir / "embedding_gwas_signal_cluster_summary_jaccard050.csv", index=False)
    if not signal_matrix.empty:
        signal_matrix.to_csv(out_dir / "embedding_locus_signed_neglog10p_matrix.csv")

    write_report(
        out_dir,
        metadata,
        summary,
        sig_loci,
        loci,
        cluster_df,
        cluster_summary_025,
        cluster_summary_050,
    )
    print(f"Wrote clustering outputs and report to {out_dir}")


if __name__ == "__main__":
    main()
