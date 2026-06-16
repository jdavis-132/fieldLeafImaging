#!/usr/bin/env python3
"""Run full-marker LOCO MLM with LRT refinement for all SAM3 embeddings.

Outputs are compact and resumable:
  - per-trait summaries
  - markers passing the effective-marker Bonferroni threshold
  - top K markers per trait

The script intentionally does not write full all-marker p-value tables for all
2,048 traits.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from panicle.association.mlm_loco import PANICLE_MLM_LOCO_MULTI
from panicle.data.loaders import load_genotype_file
from panicle.matrix.kinship_loco import PANICLE_K_VanRaden_LOCO
from panicle.matrix.pca import PANICLE_PCA
from panicle.utils.effective_tests import estimate_effective_tests_from_genotype


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "output" / "conditional_panicle" / "condition_exg"
GENOTYPE_PREFIX = DATASET_DIR / "condition_exg_plink"
DEFAULT_OUT_DIR = ROOT / "output" / "reframing_results" / "all_sam3_full_mlm_lrt"
EFFECTIVE_TESTS_JSON = (
    ROOT
    / "output"
    / "reframing_results"
    / "all_sam3_full_mlm_benchmark"
    / "effective_tests_condition_exg_plink.json"
)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def timed(label: str, fn):
    t0 = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - t0
    print(f"[{now()}] {label}: {elapsed:.2f}s", flush=True)
    return value, elapsed


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return q
    valid_idx = np.flatnonzero(valid)
    order = valid_idx[np.argsort(p[valid_idx])]
    m = order.size
    ranked = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(adjusted, 1.0)
    return q


def load_effective_tests(geno, geno_map, out_dir: Path, recompute: bool) -> tuple[dict, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    local_cache = out_dir / "effective_tests_condition_exg_plink.json"
    cache_path = local_cache if local_cache.exists() else EFFECTIVE_TESTS_JSON
    if cache_path.exists() and not recompute:
        t0 = time.perf_counter()
        info = json.loads(cache_path.read_text())
        elapsed = time.perf_counter() - t0
        if cache_path != local_cache:
            local_cache.write_text(json.dumps(info, indent=2))
        print(f"[{now()}] effective tests cache load: {elapsed:.2f}s", flush=True)
        return info, elapsed

    info, elapsed = timed(
        "effective tests compute",
        lambda: estimate_effective_tests_from_genotype(geno, geno_map, ncpus=1),
    )
    local_cache.write_text(json.dumps(info, indent=2))
    return info, elapsed


def load_sam3_phenotypes(genome_ids: list[str], traits: list[str]) -> pd.DataFrame:
    usecols = ["location", "genotype", *traits]
    blues = pd.read_csv(ROOT / "data" / "blues_all.csv", usecols=usecols)
    ne_blues = blues.loc[blues["location"].eq("NE"), ["genotype", *traits]]
    pheno = ne_blues.set_index("genotype").reindex(genome_ids)
    missing = pheno[traits].isna().any(axis=1).sum()
    if missing:
        raise ValueError(f"{missing} aligned samples have missing SAM3 embedding values")
    return pheno[traits]


def get_all_embedding_traits() -> list[str]:
    cols = pd.read_csv(ROOT / "data" / "blues_all.csv", nrows=0).columns
    return [
        c
        for c in cols
        if c.startswith("embedding_mean_") or c.startswith("embedding_std_")
    ]


def result_frames_for_trait(
    trait: str,
    result,
    marker_df: pd.DataFrame,
    threshold: float,
    top_k: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    p = np.asarray(result.pvalues, dtype=float)
    effects = np.asarray(result.effects, dtype=float)
    ses = np.asarray(result.se, dtype=float)
    sig_mask = np.isfinite(p) & (p < threshold)
    q = bh_qvalues(p)

    summary = {
        "trait": trait,
        "embedding_stat": "mean" if trait.startswith("embedding_mean_") else "std",
        "embedding_index": int(trait.rsplit("_", 1)[1]),
        "n_markers_tested": int(np.isfinite(p).sum()),
        "min_p": float(np.nanmin(p)),
        "n_significant_effective_bonferroni": int(sig_mask.sum()),
        "n_q_lt_0_05_within_trait": int(np.nansum(q < 0.05)),
    }

    base_cols = ["MARKER", "CHROM", "POS", "REF", "ALT"]
    sig_idx = np.flatnonzero(sig_mask)
    sig_df = marker_df.iloc[sig_idx][base_cols].copy()
    if not sig_df.empty:
        sig_df.insert(0, "trait", trait)
        sig_df["embedding_stat"] = summary["embedding_stat"]
        sig_df["embedding_index"] = summary["embedding_index"]
        sig_df["effect"] = effects[sig_idx]
        sig_df["se"] = ses[sig_idx]
        sig_df["p_value"] = p[sig_idx]
        sig_df["q_value_within_trait"] = q[sig_idx]

    k = min(top_k, p.size)
    top_idx = np.argpartition(p, k - 1)[:k]
    top_idx = top_idx[np.argsort(p[top_idx])]
    top_df = marker_df.iloc[top_idx][base_cols].copy()
    top_df.insert(0, "trait", trait)
    top_df["embedding_stat"] = summary["embedding_stat"]
    top_df["embedding_index"] = summary["embedding_index"]
    top_df["effect"] = effects[top_idx]
    top_df["se"] = ses[top_idx]
    top_df["p_value"] = p[top_idx]
    top_df["q_value_within_trait"] = q[top_idx]
    top_df["passes_effective_bonferroni"] = p[top_idx] < threshold
    return summary, sig_df, top_df


def concatenate_outputs(out_dir: Path) -> None:
    chunks = sorted((out_dir / "chunks").glob("chunk_*_summary.csv"))
    if not chunks:
        return
    for name in ["summary", "significant_hits", "top_hits"]:
        frames = []
        for path in sorted((out_dir / "chunks").glob(f"chunk_*_{name}.csv")):
            if path.stat().st_size:
                frames.append(pd.read_csv(path))
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out_dir / f"all_traits_{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--n-pcs", type=int, default=5)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--max-line", type=int, default=1000)
    parser.add_argument("--lrt-batch-size", type=int, default=2048)
    parser.add_argument("--lrt-solver", default="GEMMA")
    parser.add_argument("--recompute-effective-tests", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run completed chunks.")
    args = parser.parse_args()

    out_dir = args.out_dir
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    run_started = now()
    all_traits = get_all_embedding_traits()
    if not all_traits:
        raise ValueError("No SAM3 embedding traits found in data/blues_all.csv")

    (loaded, load_elapsed) = timed(
        "genotype cache load",
        lambda: load_genotype_file(
            GENOTYPE_PREFIX,
            file_format="plink",
            precompute_alleles=False,
        ),
    )
    geno, genome_ids, geno_map = loaded
    genome_ids = list(genome_ids)
    marker_df = geno_map.to_dataframe()
    print(
        f"[{now()}] matrix: {geno.n_individuals} samples x {geno.n_markers} markers; "
        f"{len(all_traits)} SAM3 traits",
        flush=True,
    )

    effective_info, effective_elapsed = load_effective_tests(
        geno, geno_map, out_dir, args.recompute_effective_tests
    )
    me = int(effective_info["Me"])
    threshold = 0.05 / me
    print(f"[{now()}] effective markers Me={me}; threshold={threshold:.3e}", flush=True)

    pcs, pc_elapsed = timed(
        f"compute {args.n_pcs} PCs",
        lambda: PANICLE_PCA(M=geno, pcs_keep=args.n_pcs, verbose=False),
    )
    loco, loco_elapsed = timed(
        "compute LOCO kinship",
        lambda: PANICLE_K_VanRaden_LOCO(geno, geno_map, verbose=False),
    )

    metadata = {
        "run_started": run_started,
        "n_samples": geno.n_individuals,
        "n_markers": geno.n_markers,
        "n_traits": len(all_traits),
        "chunk_size": args.chunk_size,
        "top_k": args.top_k,
        "n_pcs": args.n_pcs,
        "effective_markers": me,
        "effective_bonferroni_threshold": threshold,
        "genotype_load_seconds": load_elapsed,
        "effective_tests_seconds": effective_elapsed,
        "pc_seconds": pc_elapsed,
        "loco_kinship_seconds": loco_elapsed,
        "lrt_refinement": True,
        "lrt_solver": args.lrt_solver,
        "lrt_batch_size": args.lrt_batch_size,
        "cpu": args.cpu,
        "max_line": args.max_line,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    total_chunks = int(np.ceil(len(all_traits) / args.chunk_size))
    completed = 0
    for chunk_idx, start in enumerate(range(0, len(all_traits), args.chunk_size), start=1):
        traits = all_traits[start : start + args.chunk_size]
        stem = f"chunk_{chunk_idx:03d}"
        summary_path = chunk_dir / f"{stem}_summary.csv"
        sig_path = chunk_dir / f"{stem}_significant_hits.csv"
        top_path = chunk_dir / f"{stem}_top_hits.csv"
        done_path = chunk_dir / f"{stem}.done"
        if done_path.exists() and summary_path.exists() and not args.force:
            print(f"[{now()}] Skipping completed {stem} ({len(traits)} traits)", flush=True)
            completed += 1
            continue

        print(
            f"[{now()}] Starting {stem}/{total_chunks}: {traits[0]}..{traits[-1]} "
            f"({len(traits)} traits)",
            flush=True,
        )
        pheno = load_sam3_phenotypes(genome_ids, traits)
        y_matrix = pheno[traits].to_numpy(dtype=float)
        results, elapsed = timed(
            f"{stem} multi-trait LOCO MLM + LRT",
            lambda: PANICLE_MLM_LOCO_MULTI(
                phe=y_matrix,
                geno=geno,
                map_data=geno_map,
                trait_names=traits,
                loco_kinship=loco,
                CV=pcs,
                maxLine=args.max_line,
                cpu=args.cpu,
                lrt_refinement=True,
                lrt_solver=args.lrt_solver,
                lrt_batch_size=args.lrt_batch_size,
                verbose=False,
            ),
        )

        summary_rows = []
        sig_frames = []
        top_frames = []
        for trait in traits:
            summary, sig_df, top_df = result_frames_for_trait(
                trait, results[trait], marker_df, threshold, args.top_k
            )
            summary["chunk"] = chunk_idx
            summary["chunk_elapsed_seconds"] = elapsed
            summary["chunk_elapsed_seconds_per_trait"] = elapsed / len(traits)
            summary["effective_markers"] = me
            summary["effective_bonferroni_threshold"] = threshold
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
                    "embedding_stat",
                    "embedding_index",
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
        concatenate_outputs(out_dir)
        print(f"[{now()}] Finished {stem}; completed {completed}/{total_chunks}", flush=True)

    concatenate_outputs(out_dir)
    metadata["run_finished"] = now()
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[{now()}] All chunks complete. Outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
