#!/usr/bin/env python3
"""Rerun overlapping embedding GWAS traits on the human-score genotype subset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from panicle.association.mlm_loco import PANICLE_MLM_LOCO_MULTI
from panicle.data.loaders import load_genotype_file
from panicle.matrix.kinship_loco import PANICLE_K_VanRaden_LOCO
from panicle.matrix.pca import PANICLE_PCA

from cluster_sam3_gwas_signals import build_loci
from run_all_sam3_full_mlm_lrt import GENOTYPE_PREFIX, bh_qvalues, load_effective_tests, now, timed


ROOT = Path(__file__).resolve().parents[1]
SEV_DIR = ROOT / "output" / "reframing_results" / "ne_severity_trait_mlm_lrt"
OUT_DIR = SEV_DIR / "human_matched_embedding_gwas"
OVERLAP_TRAITS = SEV_DIR / "human_score_relaxed_q020_overlapping_embedding_traits_with_human_correlations.csv"
HUMAN_MARKERS = SEV_DIR / "human_score_relaxed_q020_hits_with_loci.csv"
HUMAN_LOCI = SEV_DIR / "human_score_relaxed_q020_loci.csv"


def load_human_scores(genome_ids: list[str]) -> pd.Series:
    severity = pd.read_csv(ROOT / "data" / "blues_all.csv", usecols=["location", "genotype", "human_score"])
    ne = severity.loc[severity["location"].eq("NE"), ["genotype", "human_score"]]
    return ne.set_index("genotype")["human_score"].reindex(genome_ids)


def load_trait_values(source: str, trait: str, genome_ids: list[str]) -> pd.Series:
    if source == "SAM3":
        pheno = pd.read_csv(ROOT / "data" / "blues_all.csv", usecols=["location", "genotype", trait])
        ne = pheno.loc[pheno["location"].eq("NE"), ["genotype", trait]]
    elif source == "DINOv2":
        pheno = pd.read_csv(
            ROOT / "output" / "dinov2_20260522_blues" / "dinov2_20260522_blues_all_compatible_gwas_ids.csv",
            usecols=["location", "genotype", trait],
        )
        ne = pheno.loc[pheno["location"].eq("NE"), ["genotype", trait]]
    else:
        raise ValueError(f"Unknown source: {source}")
    return ne.set_index("genotype")[trait].reindex(genome_ids)


def result_frames(
    source: str,
    trait: str,
    result,
    marker_df: pd.DataFrame,
    threshold: float,
    human_marker_ids: set[str],
    top_k: int = 500,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = np.asarray(result.pvalues, dtype=float)
    effects = np.asarray(result.effects, dtype=float)
    ses = np.asarray(result.se, dtype=float)
    q = bh_qvalues(p)
    sig_idx = np.flatnonzero(np.isfinite(p) & (p < threshold))

    summary = {
        "embedding_set": source,
        "trait": trait,
        "n_markers_tested": int(np.isfinite(p).sum()),
        "min_p": float(np.nanmin(p)),
        "n_significant_effective_bonferroni": int(len(sig_idx)),
        "n_q_lt_0_05_within_trait": int(np.nansum(q < 0.05)),
    }

    base_cols = ["MARKER", "CHROM", "POS", "REF", "ALT"]
    sig = marker_df.iloc[sig_idx][base_cols].copy()
    if not sig.empty:
        sig.insert(0, "embedding_set", source)
        sig.insert(1, "trait", trait)
        sig["effect"] = effects[sig_idx]
        sig["se"] = ses[sig_idx]
        sig["p_value"] = p[sig_idx]
        sig["q_value_within_trait"] = q[sig_idx]

    k = min(top_k, p.size)
    top_idx = np.argpartition(p, k - 1)[:k]
    top_idx = top_idx[np.argsort(p[top_idx])]
    top = marker_df.iloc[top_idx][base_cols].copy()
    top.insert(0, "embedding_set", source)
    top.insert(1, "trait", trait)
    top["effect"] = effects[top_idx]
    top["se"] = ses[top_idx]
    top["p_value"] = p[top_idx]
    top["q_value_within_trait"] = q[top_idx]
    top["passes_effective_bonferroni"] = p[top_idx] < threshold

    human_idx = np.flatnonzero(marker_df["MARKER"].isin(human_marker_ids).to_numpy())
    human = marker_df.iloc[human_idx][base_cols].copy()
    human.insert(0, "embedding_set", source)
    human.insert(1, "trait", trait)
    human["effect"] = effects[human_idx]
    human["se"] = ses[human_idx]
    human["p_value"] = p[human_idx]
    human["q_value_within_trait"] = q[human_idx]
    human["passes_effective_bonferroni"] = p[human_idx] < threshold
    return summary, sig, top, human


def interval_overlaps(left: pd.DataFrame, right: pd.DataFrame, window_bp: int = 200_000) -> pd.DataFrame:
    rows = []
    for left_row in left.itertuples(index=False):
        hits = right[
            right["CHROM"].astype(str).eq(str(left_row.CHROM))
            & (right["end"].astype(int) >= int(left_row.start) - window_bp)
            & (right["start"].astype(int) <= int(left_row.end) + window_bp)
        ]
        for hit in hits.itertuples(index=False):
            rows.append(
                {
                    "human_locus_id": left_row.locus_id,
                    "human_CHROM": left_row.CHROM,
                    "human_start": left_row.start,
                    "human_end": left_row.end,
                    "embedding_locus_id": hit.locus_id,
                    "embedding_CHROM": hit.CHROM,
                    "embedding_start": hit.start,
                    "embedding_end": hit.end,
                    "embedding_n_markers": hit.n_markers,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = (
        pd.read_csv(OVERLAP_TRAITS)[["embedding_set", "embedding_trait"]]
        .drop_duplicates()
        .rename(columns={"embedding_trait": "trait"})
        .sort_values(["embedding_set", "trait"])
    )
    human_hits = pd.read_csv(HUMAN_MARKERS)
    human_marker_ids = set(human_hits["MARKER"].astype(str))

    (loaded, load_elapsed) = timed(
        "genotype cache load",
        lambda: load_genotype_file(GENOTYPE_PREFIX, file_format="plink", precompute_alleles=False),
    )
    geno, genome_ids, geno_map = loaded
    genome_ids = list(genome_ids)
    marker_df = geno_map.to_dataframe()
    human_score = load_human_scores(genome_ids)
    sample_idx = np.flatnonzero(human_score.notna().to_numpy()).tolist()
    sub_geno = geno.subset_individuals(sample_idx)

    effective_info, effective_elapsed = load_effective_tests(geno, geno_map, OUT_DIR, recompute=False)
    threshold = 0.05 / int(effective_info["Me"])
    pcs, pc_elapsed = timed("human subset compute 5 PCs", lambda: PANICLE_PCA(M=sub_geno, pcs_keep=5, verbose=False))
    loco, loco_elapsed = timed(
        "human subset compute LOCO kinship",
        lambda: PANICLE_K_VanRaden_LOCO(sub_geno, geno_map, verbose=False),
    )

    metadata = {
        "run_started": now(),
        "n_human_score_samples": len(sample_idx),
        "n_selected_embedding_traits": int(selected.shape[0]),
        "n_human_relaxed_markers": len(human_marker_ids),
        "effective_markers": int(effective_info["Me"]),
        "effective_bonferroni_threshold": threshold,
        "model": "PANICLE LOCO MLM/LRT with 5 PCs, restricted to genotypes with nonmissing human_score",
        "genotype_load_seconds": load_elapsed,
        "effective_tests_seconds": effective_elapsed,
        "pc_seconds": pc_elapsed,
        "loco_kinship_seconds": loco_elapsed,
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    summaries = []
    sig_frames = []
    top_frames = []
    human_marker_frames = []
    for row in selected.itertuples(index=False):
        done_path = OUT_DIR / f"{row.embedding_set}_{row.trait}.done"
        summary_path = OUT_DIR / f"{row.embedding_set}_{row.trait}_summary.csv"
        sig_path = OUT_DIR / f"{row.embedding_set}_{row.trait}_significant_hits.csv"
        top_path = OUT_DIR / f"{row.embedding_set}_{row.trait}_top_hits.csv"
        human_path = OUT_DIR / f"{row.embedding_set}_{row.trait}_human_marker_pvalues.csv"
        if done_path.exists() and summary_path.exists() and sig_path.exists() and top_path.exists() and human_path.exists():
            summaries.append(pd.read_csv(summary_path).iloc[0].to_dict())
            sig_frames.append(pd.read_csv(sig_path))
            top_frames.append(pd.read_csv(top_path))
            human_marker_frames.append(pd.read_csv(human_path))
            continue

        values = load_trait_values(row.embedding_set, row.trait, genome_ids).iloc[sample_idx].to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError(f"{row.embedding_set} {row.trait} has missing values in human-score sample subset")
        y = values.reshape(-1, 1)
        print(f"[{now()}] {row.embedding_set} {row.trait}: {len(sample_idx)} samples", flush=True)
        results, assoc_elapsed = timed(
            f"{row.embedding_set} {row.trait} LOCO MLM + LRT",
            lambda: PANICLE_MLM_LOCO_MULTI(
                phe=y,
                geno=sub_geno,
                map_data=geno_map,
                trait_names=[row.trait],
                loco_kinship=loco,
                CV=pcs,
                maxLine=1000,
                cpu=1,
                lrt_refinement=True,
                lrt_solver="GEMMA",
                lrt_batch_size=2048,
                verbose=False,
            ),
        )
        summary, sig, top, human = result_frames(
            row.embedding_set,
            row.trait,
            results[row.trait],
            marker_df,
            threshold,
            human_marker_ids,
        )
        summary["n_samples"] = len(sample_idx)
        summary["association_seconds"] = assoc_elapsed
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        sig.to_csv(sig_path, index=False)
        top.to_csv(top_path, index=False)
        human.to_csv(human_path, index=False)
        done_path.write_text(f"{now()}\n")
        summaries.append(summary)
        sig_frames.append(sig)
        top_frames.append(top)
        human_marker_frames.append(human)

        pd.DataFrame(summaries).to_csv(OUT_DIR / "matched_embedding_gwas_summary.csv", index=False)
        pd.concat(sig_frames, ignore_index=True).to_csv(OUT_DIR / "matched_embedding_gwas_significant_hits.csv", index=False)
        pd.concat(top_frames, ignore_index=True).to_csv(OUT_DIR / "matched_embedding_gwas_top_hits.csv", index=False)
        pd.concat(human_marker_frames, ignore_index=True).to_csv(OUT_DIR / "matched_embedding_human_marker_pvalues.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    sig_df = pd.concat(sig_frames, ignore_index=True) if sig_frames else pd.DataFrame()
    top_df = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame()
    human_df = pd.concat(human_marker_frames, ignore_index=True) if human_marker_frames else pd.DataFrame()
    summary_df.to_csv(OUT_DIR / "matched_embedding_gwas_summary.csv", index=False)
    sig_df.to_csv(OUT_DIR / "matched_embedding_gwas_significant_hits.csv", index=False)
    top_df.to_csv(OUT_DIR / "matched_embedding_gwas_top_hits.csv", index=False)
    human_df.to_csv(OUT_DIR / "matched_embedding_human_marker_pvalues.csv", index=False)

    if not sig_df.empty:
        sig_loci, loci = build_loci(sig_df, window_bp=200_000)
        sig_loci.to_csv(OUT_DIR / "matched_embedding_gwas_significant_hits_with_loci.csv", index=False)
        loci.to_csv(OUT_DIR / "matched_embedding_gwas_loci.csv", index=False)
        reps = sig_loci.sort_values("p_value").groupby(["embedding_set", "trait", "locus_id"], as_index=False).first()
        reps["neg_log10_p"] = -np.log10(reps["p_value"])
        reps.to_csv(OUT_DIR / "matched_embedding_locus_representative_signals.csv", index=False)

        human_loci = pd.read_csv(HUMAN_LOCI)
        overlaps = interval_overlaps(human_loci, loci)
        if not overlaps.empty:
            reps_support = (
                reps.groupby("locus_id")
                .agg(
                    embedding_sets=("embedding_set", lambda x: ";".join(sorted(set(x)))),
                    embedding_traits=("trait", lambda x: ";".join(sorted(set(x)))),
                    n_embedding_traits=("trait", "nunique"),
                    min_embedding_p=("p_value", "min"),
                )
                .reset_index()
                .rename(columns={"locus_id": "embedding_locus_id"})
            )
            overlaps = overlaps.merge(reps_support, on="embedding_locus_id", how="left")
        overlaps.to_csv(OUT_DIR / "matched_embedding_loci_overlapping_human_relaxed_loci.csv", index=False)

    metadata["run_finished"] = now()
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary_df.to_string(index=False))
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
