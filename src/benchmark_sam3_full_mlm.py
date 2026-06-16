#!/usr/bin/env python3
"""Benchmark full-marker PANICLE LOCO MLM for SAM3 embeddings.

The benchmark uses the cached 895-sample PLINK genotype matrix, computes or
loads effective marker number, computes shared PCs and LOCO kinship, then runs
full-marker LOCO MLM for one or more SAM3 embeddings. Output is compact: no
full all-marker p-value table is written unless explicitly requested later.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from panicle.association.mlm_loco import PANICLE_MLM_LOCO, PANICLE_MLM_LOCO_MULTI
from panicle.data.loaders import load_genotype_file
from panicle.matrix.kinship_loco import PANICLE_K_VanRaden_LOCO
from panicle.matrix.pca import PANICLE_PCA
from panicle.utils.effective_tests import estimate_effective_tests_from_genotype


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "output" / "conditional_panicle" / "condition_exg"
OUT_DIR = ROOT / "output" / "reframing_results" / "all_sam3_full_mlm_benchmark"
GENOTYPE_PREFIX = DATASET_DIR / "condition_exg_plink"
EFFECTIVE_TESTS_JSON = OUT_DIR / "effective_tests_condition_exg_plink.json"


def timed(label: str, fn):
    t0 = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - t0
    print(f"{label}: {elapsed:.2f}s", flush=True)
    return value, elapsed


def load_sam3_phenotypes(genome_ids: list[str], traits: list[str]) -> pd.DataFrame:
    usecols = ["location", "genotype", *traits]
    blues = pd.read_csv(ROOT / "data" / "blues_all.csv", usecols=usecols)
    ne_blues = blues.loc[blues["location"].eq("NE"), ["genotype", *traits]]
    pheno = ne_blues.set_index("genotype").reindex(genome_ids)
    missing = pheno[traits].isna().any(axis=1).sum()
    if missing:
        raise ValueError(f"{missing} aligned samples have missing SAM3 embedding values")
    return pheno[traits]


def get_effective_tests(geno, geno_map, recompute: bool) -> tuple[dict, float]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if EFFECTIVE_TESTS_JSON.exists() and not recompute:
        t0 = time.perf_counter()
        info = json.loads(EFFECTIVE_TESTS_JSON.read_text())
        elapsed = time.perf_counter() - t0
        print(f"effective tests cache load: {elapsed:.2f}s", flush=True)
        return info, elapsed

    info, elapsed = timed(
        "effective tests compute",
        lambda: estimate_effective_tests_from_genotype(geno, geno_map, ncpus=1),
    )
    EFFECTIVE_TESTS_JSON.write_text(json.dumps(info, indent=2))
    return info, elapsed


def top_hits_for_result(res, marker_df: pd.DataFrame, trait: str, threshold: float, top_k: int) -> pd.DataFrame:
    p = np.asarray(res.pvalues, dtype=float)
    k = min(top_k, p.size)
    top_idx = np.argpartition(p, k - 1)[:k]
    top_idx = top_idx[np.argsort(p[top_idx])]
    out = marker_df.iloc[top_idx][["MARKER", "CHROM", "POS", "REF", "ALT"]].copy()
    out.insert(0, "trait", trait)
    out["effect"] = np.asarray(res.effects, dtype=float)[top_idx]
    out["se"] = np.asarray(res.se, dtype=float)[top_idx]
    out["p_value"] = p[top_idx]
    out["passes_effective_bonferroni"] = out["p_value"] < threshold
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traits",
        default="embedding_std_976",
        help="Comma-separated SAM3 embedding columns to benchmark.",
    )
    parser.add_argument("--n-pcs", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--max-line", type=int, default=1000)
    parser.add_argument("--lrt-refinement", action="store_true")
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--recompute-effective-tests", action="store_true")
    args = parser.parse_args()

    traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
        f"matrix: {geno.n_individuals} samples x {geno.n_markers} markers; "
        f"traits={','.join(traits)}",
        flush=True,
    )

    effective_info, effective_elapsed = get_effective_tests(
        geno, geno_map, args.recompute_effective_tests
    )
    me = int(effective_info["Me"])
    threshold = 0.05 / me
    print(f"effective markers Me={me}; threshold={threshold:.3e}", flush=True)

    pcs, pc_elapsed = timed(
        f"compute {args.n_pcs} PCs",
        lambda: PANICLE_PCA(M=geno, pcs_keep=args.n_pcs, verbose=False),
    )
    loco, loco_elapsed = timed(
        "compute LOCO kinship",
        lambda: PANICLE_K_VanRaden_LOCO(geno, geno_map, verbose=False),
    )
    pheno = load_sam3_phenotypes(genome_ids, traits)

    if args.multi:
        y_matrix = pheno[traits].to_numpy(dtype=float)

        results, gwas_elapsed = timed(
            "multi-trait LOCO MLM",
            lambda: PANICLE_MLM_LOCO_MULTI(
                phe=y_matrix,
                geno=geno,
                map_data=geno_map,
                trait_names=traits,
                loco_kinship=loco,
                CV=pcs,
                maxLine=args.max_line,
                cpu=args.cpu,
                lrt_refinement=args.lrt_refinement,
                verbose=False,
            ),
        )
    else:
        results = {}
        t0 = time.perf_counter()
        for trait in traits:
            y = pheno[trait].to_numpy(dtype=float)
            phe = np.column_stack([np.asarray(genome_ids, dtype=object), y])
            results[trait] = PANICLE_MLM_LOCO(
                phe=phe,
                geno=geno,
                map_data=geno_map,
                loco_kinship=loco,
                CV=pcs,
                maxLine=args.max_line,
                cpu=args.cpu,
                lrt_refinement=args.lrt_refinement,
                verbose=False,
            )
        gwas_elapsed = time.perf_counter() - t0
        print(f"per-trait LOCO MLM loop: {gwas_elapsed:.2f}s", flush=True)

    summary_rows = []
    hit_frames = []
    for trait, res in results.items():
        p = np.asarray(res.pvalues, dtype=float)
        sig = p < threshold
        summary_rows.append(
            {
                "trait": trait,
                "n_samples": geno.n_individuals,
                "n_markers": geno.n_markers,
                "effective_markers": me,
                "effective_bonferroni_threshold": threshold,
                "min_p": float(np.nanmin(p)),
                "n_significant_effective_bonferroni": int(np.sum(sig)),
                "gwas_elapsed_seconds_total": gwas_elapsed,
                "gwas_elapsed_seconds_per_trait": gwas_elapsed / len(traits),
                "genotype_load_seconds": load_elapsed,
                "effective_tests_seconds": effective_elapsed,
                "pc_seconds": pc_elapsed,
                "loco_kinship_seconds": loco_elapsed,
                "lrt_refinement": bool(args.lrt_refinement),
                "multi": bool(args.multi),
                "cpu": int(args.cpu),
                "max_line": int(args.max_line),
            }
        )
        hit_frames.append(top_hits_for_result(res, marker_df, trait, threshold, args.top_k))

    suffix = "multi" if args.multi else "single"
    lrt = "lrt" if args.lrt_refinement else "wald"
    trait_tag = f"{len(traits)}traits"
    summary_path = OUT_DIR / f"benchmark_{trait_tag}_{suffix}_{lrt}_summary.csv"
    top_path = OUT_DIR / f"benchmark_{trait_tag}_{suffix}_{lrt}_top_hits.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.concat(hit_frames, ignore_index=True).to_csv(top_path, index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Wrote {summary_path}")
    print(f"Wrote {top_path}")


if __name__ == "__main__":
    main()
