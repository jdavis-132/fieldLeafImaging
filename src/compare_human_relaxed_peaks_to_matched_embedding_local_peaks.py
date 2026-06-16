#!/usr/bin/env python3
"""Compare relaxed human-score GWAS peaks to matched-genotype embedding local peaks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from panicle.association.mlm_loco import PANICLE_MLM_LOCO_MULTI
from panicle.data.loaders import load_genotype_file
from panicle.matrix.kinship_loco import PANICLE_K_VanRaden_LOCO
from panicle.matrix.pca import PANICLE_PCA

from run_all_sam3_full_mlm_lrt import GENOTYPE_PREFIX, load_effective_tests, now, timed


ROOT = Path(__file__).resolve().parents[1]
SEV_DIR = ROOT / "output" / "reframing_results" / "ne_severity_trait_mlm_lrt"
OUT_DIR = SEV_DIR / "human_matched_embedding_gwas"
WINDOW_BP = 200_000
ROUGH_DELTA_LOG10 = 0.5


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


def residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(covariates.shape[0]), covariates])
    beta, *_ = np.linalg.lstsq(x, values, rcond=None)
    return values - x @ beta


def pc_adjusted_marker_r2(y: np.ndarray, marker: np.ndarray, pcs: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    marker = np.asarray(marker, dtype=float)
    pcs = np.asarray(pcs, dtype=float)
    keep = np.isfinite(y) & np.isfinite(marker) & np.isfinite(pcs).all(axis=1)
    if keep.sum() < pcs.shape[1] + 3:
        return np.nan
    ry = residualize(y[keep], pcs[keep])
    rx = residualize(marker[keep], pcs[keep])
    if np.nanstd(ry) == 0 or np.nanstd(rx) == 0:
        return np.nan
    return float(np.corrcoef(ry, rx)[0, 1] ** 2)


def load_recovered_traits() -> pd.DataFrame:
    overlap = pd.read_csv(SEV_DIR / "human_score_relaxed_q020_overlapping_embedding_traits_with_human_correlations.csv")
    return (
        overlap[["human_locus_id", "embedding_set", "embedding_locus_id", "embedding_trait"]]
        .rename(columns={"embedding_trait": "trait"})
        .drop_duplicates()
        .sort_values(["embedding_set", "trait", "human_locus_id"])
    )


def candidate_summary() -> pd.DataFrame:
    anth = pd.read_csv(SEV_DIR / "human_score_relaxed_q020_anthracnose_candidate_gene_overlaps.csv")
    all_dis = pd.read_csv(SEV_DIR / "human_score_relaxed_q020_all_disease_candidate_gene_overlaps.csv")
    frames = []
    for prefix, df in [("anthracnose", anth), ("all_disease", all_dis)]:
        if df.empty:
            continue
        frames.append(
            df.groupby("human_locus_id")
            .agg(
                **{
                    f"{prefix}_candidate_genes": ("gene_id", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
                    f"{prefix}_candidate_categories": ("category", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
                }
            )
            .reset_index()
        )
    out = None
    for frame in frames:
        out = frame if out is None else out.merge(frame, on="human_locus_id", how="outer")
    return out if out is not None else pd.DataFrame(columns=["human_locus_id"])


def classify_delta(delta: float) -> str:
    if not np.isfinite(delta):
        return "not_recovered_by_embeddings"
    if delta > ROUGH_DELTA_LOG10:
        return "embedding_more_significant_matched"
    if delta < -ROUGH_DELTA_LOG10:
        return "embedding_less_significant_matched"
    return "roughly_equivalent_matched"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    human_loci = pd.read_csv(SEV_DIR / "human_score_relaxed_q020_loci.csv")
    human_reps = pd.read_csv(SEV_DIR / "human_score_relaxed_q020_locus_representative_signals.csv")
    human_reps = human_reps.rename(
        columns={"locus_id": "human_locus_id", "MARKER": "human_peak_marker", "p_value": "human_peak_p", "q_value_within_trait": "human_peak_q"}
    )
    recovered = load_recovered_traits()

    (loaded, load_elapsed) = timed(
        "genotype cache load",
        lambda: load_genotype_file(GENOTYPE_PREFIX, file_format="plink", precompute_alleles=False),
    )
    geno, genome_ids, geno_map = loaded
    genome_ids = list(genome_ids)
    marker_df = geno_map.to_dataframe().reset_index(names="marker_index")
    marker_df["CHROM_str"] = marker_df["CHROM"].astype(str)
    marker_index_by_id = pd.Series(marker_df["marker_index"].to_numpy(), index=marker_df["MARKER"].astype(str)).to_dict()

    human_score = load_human_scores(genome_ids)
    sample_idx = np.flatnonzero(human_score.notna().to_numpy()).tolist()
    sub_geno = geno.subset_individuals(sample_idx)
    y_human = human_score.iloc[sample_idx].to_numpy(dtype=float)

    effective_info, effective_elapsed = load_effective_tests(geno, geno_map, OUT_DIR, recompute=False)
    threshold = 0.05 / int(effective_info["Me"])
    pcs, pc_elapsed = timed("human subset compute 5 PCs", lambda: PANICLE_PCA(M=sub_geno, pcs_keep=5, verbose=False))
    loco, loco_elapsed = timed(
        "human subset compute LOCO kinship",
        lambda: PANICLE_K_VanRaden_LOCO(sub_geno, geno_map, verbose=False),
    )
    pcs = np.asarray(pcs, dtype=float)

    locus_markers = {}
    human_peak_r2 = {}
    for locus in human_loci.itertuples(index=False):
        idx = marker_df[
            marker_df["CHROM_str"].eq(str(locus.CHROM))
            & (marker_df["POS"].astype(int) >= int(locus.start) - WINDOW_BP)
            & (marker_df["POS"].astype(int) <= int(locus.end) + WINDOW_BP)
        ]["marker_index"].to_numpy(dtype=int)
        locus_markers[locus.locus_id] = idx
    for rep in human_reps.itertuples(index=False):
        marker_idx = marker_index_by_id.get(str(rep.human_peak_marker))
        if marker_idx is None:
            human_peak_r2[rep.human_locus_id] = np.nan
            continue
        marker = np.asarray(geno.get_marker_imputed(marker_idx), dtype=float)[sample_idx]
        human_peak_r2[rep.human_locus_id] = pc_adjusted_marker_r2(y_human, marker, pcs)

    local_rows = []
    trait_groups = recovered[["embedding_set", "trait"]].drop_duplicates()
    for trait_row in trait_groups.itertuples(index=False):
        source = trait_row.embedding_set
        trait = trait_row.trait
        values = load_trait_values(source, trait, genome_ids).iloc[sample_idx].to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError(f"{source} {trait} has missing values on human-score subset")
        print(f"[{now()}] {source} {trait}: local peak extraction", flush=True)
        results, assoc_elapsed = timed(
            f"{source} {trait} LOCO MLM + LRT",
            lambda: PANICLE_MLM_LOCO_MULTI(
                phe=values.reshape(-1, 1),
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
        result = results[trait]
        p = np.asarray(result.pvalues, dtype=float)
        effects = np.asarray(result.effects, dtype=float)
        ses = np.asarray(result.se, dtype=float)

        for rec in recovered[(recovered["embedding_set"].eq(source)) & (recovered["trait"].eq(trait))].itertuples(index=False):
            idx = locus_markers[rec.human_locus_id]
            if idx.size == 0:
                continue
            local_p = p[idx]
            valid = np.isfinite(local_p)
            if not valid.any():
                continue
            valid_idx = idx[valid]
            best_idx = int(valid_idx[np.argmin(local_p[valid])])
            best_marker = marker_df.iloc[best_idx]
            marker = np.asarray(geno.get_marker_imputed(best_idx), dtype=float)[sample_idx]
            local_rows.append(
                {
                    "human_locus_id": rec.human_locus_id,
                    "embedding_set": source,
                    "embedding_trait": trait,
                    "full_embedding_locus_id": rec.embedding_locus_id,
                    "matched_local_peak_marker": best_marker["MARKER"],
                    "matched_local_peak_CHROM": best_marker["CHROM"],
                    "matched_local_peak_POS": int(best_marker["POS"]),
                    "matched_local_peak_effect": float(effects[best_idx]),
                    "matched_local_peak_se": float(ses[best_idx]),
                    "matched_local_peak_p": float(p[best_idx]),
                    "matched_local_peak_neglog10_p": float(-np.log10(p[best_idx])),
                    "matched_local_peak_pc_adjusted_marker_r2": pc_adjusted_marker_r2(values, marker, pcs),
                    "matched_local_peak_passes_effective_bonferroni": bool(p[best_idx] < threshold),
                    "matched_association_seconds": assoc_elapsed,
                    "n_window_markers": int(idx.size),
                }
            )
    local = pd.DataFrame(local_rows)
    local.to_csv(OUT_DIR / "matched_embedding_local_peaks_in_human_relaxed_windows.csv", index=False)

    best = pd.DataFrame()
    if not local.empty:
        best = local.sort_values("matched_local_peak_p").groupby("human_locus_id", as_index=False).first()
    human = human_reps[
        [
            "human_locus_id",
            "human_peak_marker",
            "CHROM",
            "POS",
            "human_peak_p",
            "human_peak_q",
            "neg_log10_p",
        ]
    ].rename(columns={"CHROM": "human_peak_CHROM", "POS": "human_peak_POS", "neg_log10_p": "human_peak_neglog10_p"})
    human["human_peak_pc_adjusted_marker_r2"] = human["human_locus_id"].map(human_peak_r2)
    summary = human.merge(best, on="human_locus_id", how="left")
    for col in ["embedding_set", "embedding_trait", "full_embedding_locus_id"]:
        if col not in summary:
            summary[col] = np.nan
    summary["recovered_by_full_embedding_gwas"] = summary["embedding_trait"].notna()
    summary["delta_neglog10_matched_embedding_minus_human"] = (
        summary["matched_local_peak_neglog10_p"] - summary["human_peak_neglog10_p"]
    )
    summary["matched_significance_comparison"] = summary["delta_neglog10_matched_embedding_minus_human"].map(classify_delta)
    candidates = candidate_summary()
    summary = summary.merge(candidates, on="human_locus_id", how="left")
    summary.to_csv(OUT_DIR / "human_relaxed_peak_recovery_summary_local_peak_r2.csv", index=False)

    metadata = {
        "run_started": now(),
        "n_human_score_samples": len(sample_idx),
        "n_human_relaxed_loci": int(human_loci.shape[0]),
        "n_recovered_embedding_traits": int(trait_groups.shape[0]),
        "local_window_bp_each_side": WINDOW_BP,
        "roughly_equivalent_abs_delta_neglog10_p": ROUGH_DELTA_LOG10,
        "marker_r2_definition": "Partial R2 from marker dosage after residualizing phenotype and marker on intercept plus 5 genotype PCs; kinship is not included in this R2 summary.",
        "effective_markers": int(effective_info["Me"]),
        "effective_bonferroni_threshold": threshold,
        "genotype_load_seconds": load_elapsed,
        "effective_tests_seconds": effective_elapsed,
        "pc_seconds": pc_elapsed,
        "loco_kinship_seconds": loco_elapsed,
    }
    (OUT_DIR / "local_peak_r2_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
