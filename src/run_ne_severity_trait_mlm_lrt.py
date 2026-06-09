#!/usr/bin/env python3
"""Run direct PANICLE LOCO MLM/LRT GWAS for NE disease severity traits."""

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
OUT_DIR = ROOT / "output" / "reframing_results" / "ne_severity_trait_mlm_lrt"
TRAITS = ["human_score", "percentUnhealthy"]


def load_phenotypes(genome_ids: list[str]) -> pd.DataFrame:
    blues = pd.read_csv(ROOT / "data" / "blues_all.csv", usecols=["location", "genotype", *TRAITS])
    ne = blues.loc[blues["location"].eq("NE"), ["genotype", *TRAITS]].copy()
    pheno = ne.set_index("genotype").reindex(genome_ids)
    pheno.index.name = "genotype"
    return pheno


def result_rows(trait: str, result, marker_df: pd.DataFrame, threshold: float, top_k: int = 500) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    p = np.asarray(result.pvalues, dtype=float)
    effects = np.asarray(result.effects, dtype=float)
    ses = np.asarray(result.se, dtype=float)
    q = bh_qvalues(p)
    sig_idx = np.flatnonzero(np.isfinite(p) & (p < threshold))
    summary = {
        "trait": trait,
        "n_markers_tested": int(np.isfinite(p).sum()),
        "min_p": float(np.nanmin(p)),
        "n_significant_effective_bonferroni": int(len(sig_idx)),
        "n_q_lt_0_05_within_trait": int(np.nansum(q < 0.05)),
    }
    base_cols = ["MARKER", "CHROM", "POS", "REF", "ALT"]
    sig = marker_df.iloc[sig_idx][base_cols].copy()
    if not sig.empty:
        sig.insert(0, "trait", trait)
        sig["effect"] = effects[sig_idx]
        sig["se"] = ses[sig_idx]
        sig["p_value"] = p[sig_idx]
        sig["q_value_within_trait"] = q[sig_idx]

    k = min(top_k, p.size)
    top_idx = np.argpartition(p, k - 1)[:k]
    top_idx = top_idx[np.argsort(p[top_idx])]
    top = marker_df.iloc[top_idx][base_cols].copy()
    top.insert(0, "trait", trait)
    top["effect"] = effects[top_idx]
    top["se"] = ses[top_idx]
    top["p_value"] = p[top_idx]
    top["q_value_within_trait"] = q[top_idx]
    top["passes_effective_bonferroni"] = p[top_idx] < threshold
    return summary, sig, top


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (loaded, load_elapsed) = timed(
        "genotype cache load",
        lambda: load_genotype_file(GENOTYPE_PREFIX, file_format="plink", precompute_alleles=False),
    )
    geno, genome_ids, geno_map = loaded
    genome_ids = list(genome_ids)
    marker_df = geno_map.to_dataframe()
    phenotypes = load_phenotypes(genome_ids)
    phenotypes.reset_index().to_csv(OUT_DIR / "ne_severity_trait_blues_aligned.csv", index=False)

    effective_info, effective_elapsed = load_effective_tests(geno, geno_map, OUT_DIR, recompute=False)
    threshold = 0.05 / int(effective_info["Me"])
    metadata = {
        "run_started": now(),
        "n_genotype_cache_samples": geno.n_individuals,
        "n_markers": geno.n_markers,
        "effective_markers": int(effective_info["Me"]),
        "effective_bonferroni_threshold": threshold,
        "traits": TRAITS,
        "model": "PANICLE LOCO MLM/LRT with 5 PCs; NE genotype-level BLUEs from data/blues_all.csv",
        "genotype_load_seconds": load_elapsed,
        "effective_tests_seconds": effective_elapsed,
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    summaries = []
    sig_frames = []
    top_frames = []
    for trait in TRAITS:
        pheno = phenotypes[trait]
        sample_idx = np.flatnonzero(pheno.notna().to_numpy()).tolist()
        y = pheno.iloc[sample_idx].to_numpy(dtype=float).reshape(-1, 1)
        sub_geno = geno.subset_individuals(sample_idx)
        print(f"[{now()}] {trait}: {len(sample_idx)} samples", flush=True)
        pcs, pc_elapsed = timed(f"{trait} compute 5 PCs", lambda: PANICLE_PCA(M=sub_geno, pcs_keep=5, verbose=False))
        loco, loco_elapsed = timed(
            f"{trait} compute LOCO kinship",
            lambda: PANICLE_K_VanRaden_LOCO(sub_geno, geno_map, verbose=False),
        )
        results, assoc_elapsed = timed(
            f"{trait} LOCO MLM + LRT",
            lambda: PANICLE_MLM_LOCO_MULTI(
                phe=y,
                geno=sub_geno,
                map_data=geno_map,
                trait_names=[trait],
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
        summary, sig, top = result_rows(trait, results[trait], marker_df, threshold)
        summary["n_samples"] = int(len(sample_idx))
        summary["pc_seconds"] = pc_elapsed
        summary["loco_kinship_seconds"] = loco_elapsed
        summary["association_seconds"] = assoc_elapsed
        summaries.append(summary)
        if not sig.empty:
            sig_frames.append(sig)
        top_frames.append(top)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "ne_severity_trait_gwas_summary.csv", index=False)
    sig_df = pd.concat(sig_frames, ignore_index=True) if sig_frames else pd.DataFrame()
    sig_df.to_csv(OUT_DIR / "ne_severity_trait_significant_hits.csv", index=False)
    pd.concat(top_frames, ignore_index=True).to_csv(OUT_DIR / "ne_severity_trait_top_hits.csv", index=False)
    if not sig_df.empty:
        sig_loci, loci = build_loci(sig_df, window_bp=200_000)
        sig_loci.to_csv(OUT_DIR / "ne_severity_trait_significant_hits_with_loci.csv", index=False)
        loci.to_csv(OUT_DIR / "ne_severity_trait_gwas_loci.csv", index=False)
        reps = sig_loci.sort_values("p_value").groupby(["trait", "locus_id"], as_index=False).first()
        reps["neg_log10_p"] = -np.log10(reps["p_value"])
        reps.to_csv(OUT_DIR / "ne_severity_trait_locus_representative_signals.csv", index=False)
    metadata["run_finished"] = now()
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary_df.to_string(index=False))
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
