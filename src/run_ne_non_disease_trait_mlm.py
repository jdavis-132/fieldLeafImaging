#!/usr/bin/env python3
"""Run PANICLE LOCO MLM/LRT for available Nebraska non-disease traits."""

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
OUT_DIR = ROOT / "output" / "reframing_results" / "ne_non_disease_trait_mlm_lrt"


def winsorize(values: np.ndarray, p: float = 0.01) -> np.ndarray:
    lo, hi = np.nanquantile(values, [p, 1 - p])
    return np.minimum(np.maximum(values, lo), hi)


def map_gwas_ids(df: pd.DataFrame) -> pd.DataFrame:
    conv = pd.read_csv(ROOT / "data" / "genotype_conversion_table.tsv", sep="\t", header=None, names=["from", "to"])
    conv["from"] = conv["from"].astype(str).str.replace(" ", "", regex=False)
    conv = conv.drop_duplicates("from")
    out = df.copy()
    out["genotype"] = out["genotype"].astype(str).str.replace(" ", "", regex=False)
    out = out.merge(conv, left_on="genotype", right_on="from", how="left")
    out["genotype"] = out["to"].fillna(out["genotype"]).astype(str).str.replace(r"^ExPVP_", "", regex=True)
    return out.drop(columns=["from", "to"])


def genotype_blues(df: pd.DataFrame, trait: str) -> pd.Series:
    df = df.dropna(subset=[trait, "range", "row", "genotype"]).copy()
    df[trait] = winsorize(df[trait].to_numpy(dtype=float))
    genotypes = sorted(df["genotype"].astype(str).unique())
    pieces = [np.ones((len(df), 1)), df[["range", "row"]].to_numpy(dtype=float)]
    for genotype in genotypes[1:]:
        pieces.append((df["genotype"].to_numpy(dtype=str) == genotype).astype(float).reshape(-1, 1))
    x = np.hstack(pieces)
    beta, *_ = np.linalg.lstsq(x, df[trait].to_numpy(dtype=float), rcond=None)
    values = np.zeros(len(genotypes), dtype=float) + beta[0]
    if len(genotypes) > 1:
        values[1:] += beta[3:]
    values = winsorize(values)
    return pd.Series(values, index=genotypes, name=trait)


def build_phenotypes() -> pd.DataFrame:
    field = pd.read_csv(ROOT / "data" / "ne2025" / "SbDiv_ne2025_fieldindex.csv").rename(columns=lambda c: c.strip())
    field["genotype"] = field["genotype"].astype(str).str.replace(" ", "", regex=False)

    ft = pd.read_csv(ROOT / "data" / "manual" / "SbDiv_ne2025_FT_clean.csv")
    ft = ft.merge(field, on="plotNumber", how="left")
    ft = map_gwas_ids(ft)
    days = genotype_blues(ft, "days_to_flower")

    leaf = pd.read_csv(ROOT / "data" / "manual" / "ne2025_exg_percent_disease_area_updated.csv")
    leaf = leaf.dropna(subset=["plotNumber", "leaf_pixels"]).copy()
    leaf["plotNumber"] = leaf["plotNumber"].astype(int)
    leaf = leaf.merge(field, on="plotNumber", how="left")
    leaf = map_gwas_ids(leaf)
    leaf["log_leaf_pixels"] = np.log(leaf["leaf_pixels"].astype(float))
    leaf_pixels = genotype_blues(leaf, "leaf_pixels")
    log_leaf_pixels = genotype_blues(leaf, "log_leaf_pixels")

    phenotypes = pd.concat([days, leaf_pixels, log_leaf_pixels], axis=1).reset_index(names="genotype")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phenotypes.to_csv(OUT_DIR / "ne_non_disease_trait_blues.csv", index=False)
    return phenotypes


def result_rows(trait: str, result, marker_df: pd.DataFrame, threshold: float) -> tuple[dict, pd.DataFrame]:
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
    sig = marker_df.iloc[sig_idx][["MARKER", "CHROM", "POS", "REF", "ALT"]].copy()
    if not sig.empty:
        sig.insert(0, "trait", trait)
        sig["effect"] = effects[sig_idx]
        sig["se"] = ses[sig_idx]
        sig["p_value"] = p[sig_idx]
        sig["q_value_within_trait"] = q[sig_idx]
    return summary, sig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phenotypes = build_phenotypes()

    (loaded, _) = timed(
        "genotype cache load",
        lambda: load_genotype_file(GENOTYPE_PREFIX, file_format="plink", precompute_alleles=False),
    )
    geno, genome_ids, geno_map = loaded
    genome_ids = list(genome_ids)
    marker_df = geno_map.to_dataframe()

    effective_info, _ = load_effective_tests(geno, geno_map, OUT_DIR, recompute=False)
    threshold = 0.05 / int(effective_info["Me"])
    metadata = {
        "run_started": now(),
        "n_markers": geno.n_markers,
        "effective_markers": int(effective_info["Me"]),
        "effective_bonferroni_threshold": threshold,
        "traits": ["days_to_flower", "leaf_pixels", "log_leaf_pixels"],
        "model": "BLUEs from trait ~ range + row + genotype; PANICLE LOCO MLM/LRT with 5 PCs",
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    summaries = []
    sig_frames = []
    for trait in metadata["traits"]:
        pheno = phenotypes.set_index("genotype")[trait].reindex(genome_ids)
        keep = pheno.notna().to_numpy()
        sample_idx = np.flatnonzero(keep)
        y = pheno.iloc[sample_idx].to_numpy(dtype=float).reshape(-1, 1)
        sub_geno = geno.subset_individuals(sample_idx.tolist())
        print(f"[{now()}] {trait}: {len(sample_idx)} samples", flush=True)
        pcs, _ = timed(f"{trait} compute 5 PCs", lambda: PANICLE_PCA(M=sub_geno, pcs_keep=5, verbose=False))
        loco, _ = timed(f"{trait} compute LOCO kinship", lambda: PANICLE_K_VanRaden_LOCO(sub_geno, geno_map, verbose=False))
        results, elapsed = timed(
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
        summary, sig = result_rows(trait, results[trait], marker_df, threshold)
        summary["n_samples"] = int(len(sample_idx))
        summary["elapsed_seconds"] = elapsed
        summaries.append(summary)
        if not sig.empty:
            sig_frames.append(sig)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "ne_non_disease_trait_gwas_summary.csv", index=False)
    sig_df = pd.concat(sig_frames, ignore_index=True) if sig_frames else pd.DataFrame()
    sig_df.to_csv(OUT_DIR / "ne_non_disease_trait_significant_hits.csv", index=False)
    if not sig_df.empty:
        sig_loci, loci = build_loci(sig_df, window_bp=200_000)
        sig_loci.to_csv(OUT_DIR / "ne_non_disease_trait_significant_hits_with_loci.csv", index=False)
        loci.to_csv(OUT_DIR / "ne_non_disease_trait_gwas_loci.csv", index=False)
    metadata["run_finished"] = now()
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary_df.to_string(index=False))
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
