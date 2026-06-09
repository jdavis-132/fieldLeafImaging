#!/usr/bin/env python3
"""Run severity-conditioned PANICLE MLM tests for existing embedding GWAS hits.

This computes PCs and LOCO kinship from the genome-wide filtered PLINK files
for each analysis set, but tests only the previously RMIP-supported
marker-embedding associations using exact marker IDs from the filtered VCF.
The model is equivalent to the full conditional MLM for these markers without
writing full-genome all-marker result tables.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from panicle.association.mlm_loco import PANICLE_MLM_LOCO
from panicle.data.loaders import load_genotype_file
from panicle.matrix.kinship_loco import PANICLE_K_VanRaden_LOCO
from panicle.matrix.pca import PANICLE_PCA


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "conditional_panicle"
HITS_FILE = ROOT / "figures" / "supplemental" / "all_sig_hits_embeddings.csv"


DATASETS = {
    "condition_exg": {
        "covariates": ["percentUnhealthy"],
    },
    "condition_human_exg": {
        "covariates": ["human_score", "percentUnhealthy"],
    },
}


def embedding_label_to_column(label: str) -> str:
    match = re.fullmatch(r"\s*(\d+)\s+\((Mean|SD)\)\s*", label)
    if not match:
        raise ValueError(f"Could not parse embedding label: {label!r}")
    emb_num, stat = match.groups()
    return f"embedding_{'mean' if stat == 'Mean' else 'std'}_{emb_num}"


def bh_qvalues(pvalues: pd.Series) -> pd.Series:
    p = pvalues.astype(float)
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    m = valid.shape[0]
    if m == 0:
        return q
    ranked = valid * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked.iloc[::-1]).iloc[::-1].clip(upper=1)
    q.loc[adjusted.index] = adjusted
    return q


def write_hit_region_file(hits: pd.DataFrame, path: Path) -> None:
    seen = set()
    with path.open("w") as handle:
        for marker in hits["Marker"]:
            if marker in seen:
                continue
            seen.add(marker)
            chrom, pos, _ref, _alt = marker.split(":", 3)
            handle.write(f"{chrom}\t{pos}\t{pos}\n")


def make_hit_vcf(dataset: str, region_file: Path) -> Path:
    dataset_dir = OUT_DIR / dataset
    source_vcf = dataset_dir / f"{dataset}.vcf.gz"
    hit_vcf = dataset_dir / f"{dataset}_hit_positions.vcf.gz"
    subprocess.run(
        [
            "bcftools",
            "view",
            "-R",
            str(region_file),
            str(source_vcf),
            "-Oz",
            "-o",
            str(hit_vcf),
        ],
        check=True,
    )
    subprocess.run(["tabix", "-f", "-p", "vcf", str(hit_vcf)], check=True)
    return hit_vcf


def marker_ids(geno_map) -> np.ndarray:
    df = geno_map.to_dataframe()
    return df["MARKER"].astype(str).to_numpy()


def align_frame(path: Path, ids: list[str], columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.set_index("genotype").reindex(ids)
    missing = out[columns].isna().any(axis=1).sum()
    if missing:
        raise ValueError(f"{path} has {missing} rows with missing values after alignment")
    return out[columns]


def run_dataset(dataset: str, hits: pd.DataFrame) -> pd.DataFrame:
    dataset_dir = OUT_DIR / dataset
    genome_plink = dataset_dir / f"{dataset}_plink"
    hit_vcf = make_hit_vcf(dataset, OUT_DIR / "hit_regions.tsv")

    print(f"\n[{dataset}] loading genome-wide genotype matrix from {genome_plink}", flush=True)
    genome_geno, genome_ids, genome_map = load_genotype_file(
        genome_plink,
        file_format="plink",
        precompute_alleles=False,
    )
    print(
        f"[{dataset}] genome matrix: {genome_geno.n_individuals} samples x "
        f"{genome_geno.n_markers} markers",
        flush=True,
    )

    print(f"[{dataset}] computing 5 PCs", flush=True)
    pcs = PANICLE_PCA(M=genome_geno, pcs_keep=5, verbose=False)

    print(f"[{dataset}] computing LOCO kinship", flush=True)
    loco = PANICLE_K_VanRaden_LOCO(genome_geno, genome_map, verbose=False)

    print(f"[{dataset}] loading hit-marker genotype matrix", flush=True)
    hit_geno, hit_ids, hit_map = load_genotype_file(
        hit_vcf,
        file_format="vcf",
        precompute_alleles=False,
    )
    if list(hit_ids) != list(genome_ids):
        raise ValueError(f"{dataset}: hit VCF sample order does not match genome VCF")

    hit_marker_ids = marker_ids(hit_map)
    hit_marker_to_index = {marker: idx for idx, marker in enumerate(hit_marker_ids)}

    covariate_cols = DATASETS[dataset]["covariates"]
    covariates = align_frame(
        dataset_dir / f"{dataset}_severity_covariates.csv",
        genome_ids,
        covariate_cols,
    )
    cv = np.column_stack([pcs, covariates.to_numpy(dtype=float)])

    phenotypes = align_frame(
        dataset_dir / f"{dataset}_embedding_phenotypes.csv",
        genome_ids,
        sorted(set(hits["embedding_col"])),
    )

    rows = []
    for hit in hits.itertuples(index=False):
        row = {
            "dataset": dataset,
            "marker": hit.Marker,
            "embedding": hit.Embedding,
            "embedding_col": hit.embedding_col,
            "rmip": hit.RMIP,
            "covariates": "+".join(covariate_cols),
            "n_samples": len(genome_ids),
        }

        marker_idx = hit_marker_to_index.get(hit.Marker)
        if marker_idx is None:
            rows.append({**row, "status": "marker_not_in_filtered_vcf"})
            continue

        marker_geno = hit_geno.get_columns([marker_idx])
        marker_map = hit_map.to_dataframe().iloc[[marker_idx]].reset_index(drop=True)
        marker_map.attrs.clear()
        y = phenotypes[hit.embedding_col].to_numpy(dtype=float)
        phe = np.column_stack([np.asarray(genome_ids, dtype=object), y])

        result = PANICLE_MLM_LOCO(
            phe=phe,
            geno=marker_geno,
            map_data=marker_map,
            loco_kinship=loco,
            CV=cv,
            verbose=False,
            lrt_refinement=True,
        )
        res_df = result.to_dataframe()
        rows.append(
            {
                **row,
                "status": "tested",
                "effect": float(res_df.loc[0, "Effect"]),
                "se": float(res_df.loc[0, "SE"]),
                "p_value": float(res_df.loc[0, "P-value"]),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    hits = pd.read_csv(HITS_FILE)
    hits["embedding_col"] = hits["Embedding"].map(embedding_label_to_column)
    write_hit_region_file(hits, OUT_DIR / "hit_regions.tsv")

    all_results = []
    for dataset in DATASETS:
        all_results.append(run_dataset(dataset, hits))

    results = pd.concat(all_results, ignore_index=True)
    for dataset, idx in results.groupby("dataset").groups.items():
        tested = results.loc[idx, "status"].eq("tested")
        results.loc[idx[tested], "q_value"] = bh_qvalues(results.loc[idx[tested], "p_value"])

    out = OUT_DIR / "targeted_conditional_panicle_results.csv"
    results.to_csv(out, index=False)

    summary = (
        results.assign(
            p_lt_0_05=results["p_value"] < 0.05,
            q_lt_0_05=results["q_value"] < 0.05,
        )
        .groupby(["dataset", "status"], dropna=False)
        .agg(
            n_associations=("marker", "size"),
            n_p_lt_0_05=("p_lt_0_05", "sum"),
            n_q_lt_0_05=("q_lt_0_05", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "targeted_conditional_panicle_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
