#!/usr/bin/env python3
"""Calculate DINOv2 embedding BLUEs using the same fixed-effect model as LV_quantGen.R."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("/home/james/leaf_imaging/dinov2_20260522.csv"),
        help="DINOv2 crop-level embeddings CSV.",
    )
    parser.add_argument(
        "--field-index",
        type=Path,
        default=ROOT / "data" / "ne2025" / "SbDiv_ne2025_fieldindex.csv",
        help="Nebraska field index CSV with plotNumber, range, row, and genotype.",
    )
    parser.add_argument(
        "--genotype-alignment",
        type=Path,
        default=ROOT / "data" / "ne2025" / "genotype_alignment_reseq.csv",
        help="Two-column genotype name alignment file.",
    )
    parser.add_argument(
        "--genotype-conversion",
        type=Path,
        default=ROOT / "data" / "genotype_conversion_table.tsv",
        help="Two-column genotype conversion table used to match GWAS genotype IDs.",
    )
    parser.add_argument(
        "--images-keep",
        type=Path,
        default=ROOT / "data" / "ne2025" / "images_keep_all.csv",
        help="One-image-id-per-line file. First line is skipped to match LV_quantGen.R.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "dinov2_20260522_blues",
        help="Output directory.",
    )
    parser.add_argument("--winsor-strength", type=float, default=0.01)
    return parser.parse_args()


def crop_image_id(path: str) -> str:
    name = Path(path).name
    name = re.sub(r"-05_00_[0-9]+\.(png|npz)$", "", name)
    name = re.sub(r"-05_00\.jpg$", "", name)
    name = re.sub(r"-05_00$", "", name)
    return name


def winsorize_matrix(values: np.ndarray, lower_prob: float, upper_prob: float) -> np.ndarray:
    lower = np.nanquantile(values, lower_prob, axis=0)
    upper = np.nanquantile(values, upper_prob, axis=0)
    return np.minimum(np.maximum(values, lower), upper)


def read_and_join_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    embeddings = pd.read_csv(args.embeddings)
    trait_cols = [col for col in embeddings.columns if col.startswith("mean_") or col.startswith("std_")]
    if not trait_cols:
        raise ValueError(f"No DINOv2 mean_/std_ columns found in {args.embeddings}")

    keep = pd.read_csv(args.images_keep, header=None, skiprows=1, names=["image_id"])["image_id"].astype(str)
    keep_set = set(keep)

    embeddings = embeddings.loc[~embeddings["image_path"].astype(str).str.contains("cropped_transparent_bg")].copy()
    embeddings["plotNumber"] = embeddings["image_path"].map(lambda x: int(Path(str(x)).name.split("_", 1)[0]))
    embeddings["image_id"] = embeddings["image_path"].map(crop_image_id)
    embeddings = embeddings.loc[embeddings["image_id"].isin(keep_set), ["image_path", "plotNumber", *trait_cols]]

    field = pd.read_csv(args.field_index)
    field = field.rename(columns={col: col.strip() for col in field.columns})
    field["genotype"] = field["genotype"].astype(str).str.replace(" ", "", regex=False)

    combined = embeddings.merge(field, on="plotNumber", how="left", validate="many_to_one")
    combined = combined.dropna(subset=["genotype", "range", "row"]).copy()
    return combined, trait_cols


def build_design(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    genotype_levels = sorted(df["genotype"].astype(str).unique())
    if not genotype_levels:
        raise ValueError("No genotypes available after joining embeddings to field index")

    pieces = [
        np.ones((len(df), 1), dtype=float),
        df[["range", "row"]].to_numpy(dtype=float),
    ]
    for genotype in genotype_levels[1:]:
        pieces.append((df["genotype"].to_numpy(dtype=str) == genotype).astype(float).reshape(-1, 1))
    return np.hstack(pieces), genotype_levels


def calculate_blues(
    df: pd.DataFrame,
    trait_cols: list[str],
    winsor_strength: float,
    genotype_alignment_path: Path,
) -> pd.DataFrame:
    y = df[trait_cols].to_numpy(dtype=float)
    if np.isnan(y).any():
        missing = int(np.isnan(y).sum())
        raise ValueError(f"DINOv2 embedding matrix contains {missing} missing values; trait-specific lm() handling is not implemented")

    y_winsor = winsorize_matrix(y, winsor_strength, 1 - winsor_strength)
    x, genotype_levels = build_design(df)
    beta, *_ = np.linalg.lstsq(x, y_winsor, rcond=None)

    intercept = beta[0, :]
    genotype_effects = np.zeros((len(genotype_levels), len(trait_cols)), dtype=float)
    if len(genotype_levels) > 1:
        genotype_effects[1:, :] = beta[3:, :]
    blue_values = genotype_effects + intercept

    out = pd.DataFrame(blue_values, columns=trait_cols)
    out.insert(0, "genotype", genotype_levels)

    alignment = pd.read_csv(genotype_alignment_path)
    alignment = alignment.rename(columns={alignment.columns[0]: "genotype_idx", alignment.columns[1]: "genotype_markers"})
    alignment["genotype_idx"] = alignment["genotype_idx"].astype(str).str.replace(" ", "", regex=False)
    alignment = alignment.drop_duplicates(subset=["genotype_idx"])

    out = out.merge(alignment, left_on="genotype", right_on="genotype_idx", how="left")
    out["genotype"] = out["genotype_markers"].fillna(out["genotype"])
    out = out.drop(columns=["genotype_idx", "genotype_markers"])
    out = out.loc[out["genotype"].notna()].copy()

    values = winsorize_matrix(out[trait_cols].to_numpy(dtype=float), winsor_strength, 1 - winsor_strength)
    out.loc[:, trait_cols] = values
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    combined, trait_cols = read_and_join_inputs(args)
    blues = calculate_blues(combined, trait_cols, args.winsor_strength, args.genotype_alignment)

    renamed = blues.rename(
        columns={
            **{f"mean_{i}": f"dinov2_mean_{i}" for i in range(1024)},
            **{f"std_{i}": f"dinov2_std_{i}" for i in range(1024)},
        }
    )
    blues_all_compatible = renamed.copy()
    blues_all_compatible.insert(0, "location", "NE")

    raw_path = args.out_dir / "dinov2_20260522_blues.csv"
    compatible_path = args.out_dir / "dinov2_20260522_blues_all_compatible.csv"
    matched_compatible_path = args.out_dir / "dinov2_20260522_blues_all_compatible_matched_ne_genotypes.csv"
    gwas_compatible_path = args.out_dir / "dinov2_20260522_blues_all_compatible_gwas_ids.csv"
    keep_path = args.out_dir / "dinov2_20260522_genotypes_keep.txt"
    summary_path = args.out_dir / "dinov2_20260522_blues_summary.csv"

    blues.to_csv(raw_path, index=False)
    blues_all_compatible.to_csv(compatible_path, index=False)
    ne_blues_genotypes = set(
        pd.read_csv(ROOT / "data" / "blues_all.csv", usecols=["location", "genotype"])
        .query("location == 'NE'")["genotype"]
        .astype(str)
    )
    matched = blues_all_compatible.copy()
    matched["genotype"] = matched["genotype"].str.replace(r"^ExPVP_", "", regex=True)
    matched = matched.loc[matched["genotype"].isin(ne_blues_genotypes)].copy()
    matched.to_csv(matched_compatible_path, index=False)

    conversion = pd.read_csv(
        args.genotype_conversion,
        sep="\t",
        header=None,
        names=["genotype_idx", "genotype_markers"],
    )
    conversion["genotype_idx"] = conversion["genotype_idx"].astype(str).str.replace(" ", "", regex=False)
    conversion = conversion.drop_duplicates(subset=["genotype_idx"])
    gwas_matched = blues_all_compatible.copy()
    gwas_matched["genotype"] = gwas_matched["genotype"].str.replace(r"^ExPVP_", "", regex=True)
    gwas_matched = gwas_matched.merge(conversion, left_on="genotype", right_on="genotype_idx", how="left")
    gwas_matched["genotype"] = gwas_matched["genotype_markers"].fillna(gwas_matched["genotype"])
    gwas_matched = gwas_matched.drop(columns=["genotype_idx", "genotype_markers"])
    gwas_matched = gwas_matched.loc[gwas_matched["genotype"].isin(ne_blues_genotypes)].copy()
    gwas_matched.to_csv(gwas_compatible_path, index=False)
    pd.Series(blues["genotype"].drop_duplicates()).to_csv(keep_path, index=False, header=False)

    summary = pd.DataFrame(
        [
            {
                "input_embedding_rows": int(pd.read_csv(args.embeddings, usecols=["image_path"]).shape[0]),
                "joined_kept_crop_rows": int(combined.shape[0]),
                "n_traits": int(len(trait_cols)),
                "n_output_genotypes": int(blues.shape[0]),
                "winsor_strength": args.winsor_strength,
                "model": "trait ~ range + row + genotype",
                "raw_blues": str(raw_path),
                "blues_all_compatible": str(compatible_path),
                "blues_all_compatible_matched_ne_genotypes": str(matched_compatible_path),
                "blues_all_compatible_gwas_ids": str(gwas_compatible_path),
                "n_matched_ne_genotypes": int(matched["genotype"].nunique()),
                "n_gwas_matched_ne_genotypes": int(gwas_matched["genotype"].nunique()),
            }
        ]
    )
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print(f"Wrote {raw_path}")
    print(f"Wrote {compatible_path}")
    print(f"Wrote {matched_compatible_path}")
    print(f"Wrote {gwas_compatible_path}")


if __name__ == "__main__":
    main()
