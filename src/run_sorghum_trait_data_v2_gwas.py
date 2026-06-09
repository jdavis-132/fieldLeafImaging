#!/usr/bin/env python3
"""Run comparable PANICLE MLM/LRT GWAS for sorghum_trait_data_v2 trait-environment BLUEs."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
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
INPUT_DIR = ROOT / "output" / "reframing_results" / "sorghum_trait_data_v2_input" / "sorghum_trait_data_v2"
OUT_DIR = ROOT / "output" / "reframing_results" / "sorghum_trait_data_v2_gwas"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")


def winsorize(values: np.ndarray, p: float = 0.01) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.isfinite(values).sum() < 3:
        return values
    lo, hi = np.nanquantile(values, [p, 1 - p])
    return np.minimum(np.maximum(values, lo), hi)


def load_conversion() -> pd.DataFrame:
    conv = pd.read_csv(ROOT / "data" / "genotype_conversion_table.tsv", sep="\t", header=None, names=["from", "to"])
    conv["from"] = conv["from"].astype(str).str.replace(" ", "", regex=False)
    conv = conv.drop_duplicates("from")
    return conv


def map_gwas_ids(df: pd.DataFrame, conv: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["genotype"] = out["genotype"].astype(str).str.replace(" ", "", regex=False)
    out = out.merge(conv, left_on="genotype", right_on="from", how="left")
    out["genotype"] = out["to"].fillna(out["genotype"]).astype(str).str.replace(r"^ExPVP_", "", regex=True)
    return out.drop(columns=["from", "to"])


def design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    pieces = [np.ones((len(df), 1), dtype=float)]
    labels = ["intercept"]

    for col in ["row", "col"]:
        if col in df.columns and df[col].notna().any():
            x = df[col].astype(float).to_numpy()
            x = x - np.nanmean(x)
            x = np.nan_to_num(x, nan=0.0)
            pieces.append(x.reshape(-1, 1))
            labels.append(col)

    for col in ["rep", "treatment"]:
        if col in df.columns and df[col].notna().any() and df[col].astype(str).nunique() > 1:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True, dtype=float)
            if not dummies.empty:
                pieces.append(dummies.to_numpy(dtype=float))
                labels.extend(dummies.columns.tolist())

    genotypes = sorted(df["genotype"].astype(str).unique())
    for genotype in genotypes[1:]:
        pieces.append((df["genotype"].astype(str).to_numpy() == genotype).astype(float).reshape(-1, 1))
        labels.append(f"genotype:{genotype}")

    return np.hstack(pieces), genotypes


def blue_for_trait_env(df: pd.DataFrame, trait_name: str) -> tuple[pd.Series, dict]:
    df = df.dropna(subset=["value", "genotype"]).copy()
    df["value"] = winsorize(df["value"].to_numpy(dtype=float))
    x, genotypes = design_matrix(df)
    beta, *_ = np.linalg.lstsq(x, df["value"].to_numpy(dtype=float), rcond=None)
    values = np.zeros(len(genotypes), dtype=float) + beta[0]
    if len(genotypes) > 1:
        genotype_start = len(beta) - (len(genotypes) - 1)
        values[1:] += beta[genotype_start:]
    values = winsorize(values)
    series = pd.Series(values, index=genotypes, name=trait_name)
    meta = {
        "trait": trait_name,
        "n_plot_observations": int(len(df)),
        "n_genotypes_blue": int(len(genotypes)),
        "model": "value ~ row + col + rep + treatment + genotype, with row/col numeric and rep/treatment/genotype fixed effects where present",
    }
    return series, meta


def build_blues(genome_ids: list[str], min_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = pd.read_csv(INPUT_DIR / "observations.tsv", sep="\t", low_memory=False)
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    obs = obs.dropna(subset=["value"])
    obs = map_gwas_ids(obs, load_conversion())

    genome_set = set(genome_ids)
    blue_series = []
    meta_rows = []
    for (env_id, canonical), group in obs.groupby(["env_id", "canonical_name"], sort=True):
        trait_name = f"{safe_name(env_id)}__{safe_name(canonical)}"
        series, meta = blue_for_trait_env(group, trait_name)
        aligned = series.reindex(genome_ids)
        n_samples = int(aligned.notna().sum())
        meta.update(
            {
                "env_id": env_id,
                "canonical_name": canonical,
                "n_gwas_samples": n_samples,
                "n_gwas_samples_in_observation_rows": int(group["genotype"].isin(genome_set).sum()),
            }
        )
        meta_rows.append(meta)
        if n_samples >= min_samples:
            blue_series.append(aligned.rename(trait_name))

    blues = pd.concat(blue_series, axis=1) if blue_series else pd.DataFrame(index=genome_ids)
    blues.index.name = "genotype"
    meta_df = pd.DataFrame(meta_rows)
    return blues, meta_df


def result_frames_for_trait(trait: str, result, marker_df: pd.DataFrame, threshold: float, top_k: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    p = np.asarray(result.pvalues, dtype=float)
    effects = np.asarray(result.effects, dtype=float)
    ses = np.asarray(result.se, dtype=float)
    q = bh_qvalues(p)
    sig_idx = np.flatnonzero(np.isfinite(p) & (p < threshold))

    env_id, canonical_name = trait.split("__", 1)
    summary = {
        "trait": trait,
        "env_id": env_id,
        "canonical_name": canonical_name,
        "n_markers_tested": int(np.isfinite(p).sum()),
        "min_p": float(np.nanmin(p)),
        "n_significant_effective_bonferroni": int(len(sig_idx)),
        "n_q_lt_0_05_within_trait": int(np.nansum(q < 0.05)),
    }

    base_cols = ["MARKER", "CHROM", "POS", "REF", "ALT"]
    sig_df = marker_df.iloc[sig_idx][base_cols].copy()
    if not sig_df.empty:
        sig_df.insert(0, "trait", trait)
        sig_df["env_id"] = env_id
        sig_df["canonical_name"] = canonical_name
        sig_df["effect"] = effects[sig_idx]
        sig_df["se"] = ses[sig_idx]
        sig_df["p_value"] = p[sig_idx]
        sig_df["q_value_within_trait"] = q[sig_idx]

    k = min(top_k, p.size)
    top_idx = np.argpartition(p, k - 1)[:k]
    top_idx = top_idx[np.argsort(p[top_idx])]
    top_df = marker_df.iloc[top_idx][base_cols].copy()
    top_df.insert(0, "trait", trait)
    top_df["env_id"] = env_id
    top_df["canonical_name"] = canonical_name
    top_df["effect"] = effects[top_idx]
    top_df["se"] = ses[top_idx]
    top_df["p_value"] = p[top_idx]
    top_df["q_value_within_trait"] = q[top_idx]
    top_df["passes_effective_bonferroni"] = p[top_idx] < threshold
    return summary, sig_df, top_df


def concatenate_chunk_outputs(chunk_dir: Path, out_dir: Path) -> None:
    for name in ["summary", "significant_hits", "top_hits"]:
        frames = []
        for path in sorted(chunk_dir.glob(f"group_*_{name}.csv")):
            if path.stat().st_size:
                frames.append(pd.read_csv(path))
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out_dir / f"all_trait_env_{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--min-samples", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--max-traits-per-group", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    chunk_dir = out_dir / "groups"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    (loaded, load_elapsed) = timed(
        "genotype cache load",
        lambda: load_genotype_file(GENOTYPE_PREFIX, file_format="plink", precompute_alleles=False),
    )
    geno, genome_ids, geno_map = loaded
    genome_ids = list(genome_ids)
    marker_df = geno_map.to_dataframe()

    blues_path = out_dir / "sorghum_trait_data_v2_trait_env_blues.csv"
    meta_path = out_dir / "sorghum_trait_data_v2_trait_env_blue_metadata.csv"
    if blues_path.exists() and meta_path.exists() and not args.force:
        blues = pd.read_csv(blues_path).set_index("genotype")
        blue_meta = pd.read_csv(meta_path)
    else:
        blues, blue_meta = build_blues(genome_ids, args.min_samples)
        blues.reset_index().to_csv(blues_path, index=False)
        blue_meta.to_csv(meta_path, index=False)

    traits = blues.columns.tolist()
    if not traits:
        raise ValueError("No traits passed the minimum sample threshold.")

    effective_info, effective_elapsed = load_effective_tests(geno, geno_map, out_dir, recompute=False)
    threshold = 0.05 / int(effective_info["Me"])

    groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for trait in traits:
        idx = tuple(np.flatnonzero(blues[trait].notna().to_numpy()).tolist())
        groups[idx].append(trait)

    group_items = sorted(groups.items(), key=lambda kv: (-len(kv[1]), -len(kv[0]), kv[1][0]))
    run_items = []
    for sample_idx_tuple, group_traits in group_items:
        for start in range(0, len(group_traits), args.max_traits_per_group):
            run_items.append((sample_idx_tuple, group_traits[start : start + args.max_traits_per_group]))
    metadata = {
        "run_started": now(),
        "input_dir": str(INPUT_DIR),
        "n_samples_genotype_cache": geno.n_individuals,
        "n_markers": geno.n_markers,
        "n_trait_envs_total": int(blue_meta.shape[0]),
        "n_trait_envs_tested": int(len(traits)),
        "min_samples": args.min_samples,
        "n_sample_mask_groups": int(len(group_items)),
        "n_panicle_trait_batches": int(len(run_items)),
        "max_traits_per_group": int(args.max_traits_per_group),
        "effective_markers": int(effective_info["Me"]),
        "effective_bonferroni_threshold": threshold,
        "genotype_load_seconds": load_elapsed,
        "effective_tests_seconds": effective_elapsed,
        "association_model": "PANICLE multi-trait LOCO MLM with 5 PCs and LRT refinement",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    completed = 0
    cached_sample_idx_tuple = None
    cached_sample_idx = None
    cached_sub_geno = None
    cached_pcs = None
    cached_pc_elapsed = None
    cached_loco = None
    cached_loco_elapsed = None
    for group_idx, (sample_idx_tuple, group_traits) in enumerate(run_items, start=1):
        stem = f"group_{group_idx:03d}"
        done_path = chunk_dir / f"{stem}.done"
        summary_path = chunk_dir / f"{stem}_summary.csv"
        sig_path = chunk_dir / f"{stem}_significant_hits.csv"
        top_path = chunk_dir / f"{stem}_top_hits.csv"
        if done_path.exists() and summary_path.exists() and not args.force:
            completed += 1
            continue

        print(
            f"[{now()}] Starting {stem}/{len(run_items)}: {len(group_traits)} traits, {len(sample_idx_tuple)} samples",
            flush=True,
        )
        if cached_sample_idx_tuple != sample_idx_tuple:
            sample_idx = list(sample_idx_tuple)
            sub_geno = geno.subset_individuals(sample_idx)
            cached_pcs, cached_pc_elapsed = timed(
                f"{stem} compute 5 PCs",
                lambda: PANICLE_PCA(M=sub_geno, pcs_keep=5, verbose=False),
            )
            cached_loco, cached_loco_elapsed = timed(
                f"{stem} compute LOCO kinship",
                lambda: PANICLE_K_VanRaden_LOCO(sub_geno, geno_map, verbose=False),
            )
            cached_sample_idx_tuple = sample_idx_tuple
            cached_sample_idx = sample_idx
            cached_sub_geno = sub_geno

        sample_idx = cached_sample_idx
        sub_geno = cached_sub_geno
        pcs = cached_pcs
        pc_elapsed = cached_pc_elapsed
        loco = cached_loco
        loco_elapsed = cached_loco_elapsed
        y_matrix = blues.iloc[sample_idx][group_traits].to_numpy(dtype=float)
        results, assoc_elapsed = timed(
            f"{stem} LOCO MLM + LRT",
            lambda: PANICLE_MLM_LOCO_MULTI(
                phe=y_matrix,
                geno=sub_geno,
                map_data=geno_map,
                trait_names=group_traits,
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

        summary_rows = []
        sig_frames = []
        top_frames = []
        for trait in group_traits:
            summary, sig_df, top_df = result_frames_for_trait(trait, results[trait], marker_df, threshold, args.top_k)
            summary["group"] = group_idx
            summary["n_samples"] = len(sample_idx)
            summary["pc_seconds"] = pc_elapsed
            summary["loco_kinship_seconds"] = loco_elapsed
            summary["association_seconds_group"] = assoc_elapsed
            summary["association_seconds_per_trait"] = assoc_elapsed / len(group_traits)
            summary_rows.append(summary)
            if not sig_df.empty:
                sig_frames.append(sig_df)
            top_frames.append(top_df)

        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        if sig_frames:
            pd.concat(sig_frames, ignore_index=True).to_csv(sig_path, index=False)
        else:
            pd.DataFrame(
                columns=[
                    "trait",
                    "env_id",
                    "canonical_name",
                    "MARKER",
                    "CHROM",
                    "POS",
                    "REF",
                    "ALT",
                    "effect",
                    "se",
                    "p_value",
                    "q_value_within_trait",
                ]
            ).to_csv(sig_path, index=False)
        pd.concat(top_frames, ignore_index=True).to_csv(top_path, index=False)
        done_path.write_text(f"{now()}\n")
        completed += 1
        concatenate_chunk_outputs(chunk_dir, out_dir)
        print(f"[{now()}] Finished {stem}; completed {completed}/{len(run_items)}", flush=True)

    concatenate_chunk_outputs(chunk_dir, out_dir)
    sig_all_path = out_dir / "all_trait_env_significant_hits.csv"
    if sig_all_path.exists():
        sig = pd.read_csv(sig_all_path)
        sig_loci, loci = build_loci(sig, window_bp=200_000)
        sig_loci.to_csv(out_dir / "all_trait_env_significant_hits_with_loci.csv", index=False)
        loci.to_csv(out_dir / "trait_env_gwas_loci.csv", index=False)
        reps = sig_loci.sort_values("p_value").groupby(["trait", "locus_id"], as_index=False).first()
        reps["neg_log10_p"] = -np.log10(reps["p_value"])
        reps.to_csv(out_dir / "trait_env_locus_representative_signals.csv", index=False)

    metadata["run_finished"] = now()
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[{now()}] Complete. Outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
