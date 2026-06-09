#!/usr/bin/env python3
"""Cluster DINOv2 embeddings based on shared full-MLM GWAS signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cluster_sam3_gwas_signals import (
    build_loci,
    cluster_embeddings,
    representative_trait_locus,
    summarize_clusters,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "output" / "reframing_results" / "all_dinov2_20260522_full_mlm_lrt"


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
        "# All DINOv2 20260522 Full MLM LRT Report",
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
    ]
    (out_dir / "all_dinov2_20260522_full_mlm_lrt_report.md").write_text("\n".join(lines))


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
    print(f"Wrote DINOv2 clustering outputs and report to {out_dir}")


if __name__ == "__main__":
    main()
