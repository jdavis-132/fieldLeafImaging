#!/usr/bin/env python3
"""
Calculate the percent of leaf pixels below the ExG threshold for a directory of images.
Leaf pixels are identified from a paired mask directory (same basename, white=leaf, black=background).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

THRESHOLD_FILE = Path('/home/schnable/Documents/fieldLeafImaging/src/threshold_info.json')
with THRESHOLD_FILE.open("r") as f:
    threshold_config = json.load(f)

EXG_P20_THRESHOLD = threshold_config["thresholds"]["ExG"]["P20"]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    result = np.full_like(numerator, fill_value, dtype=float)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    return result


def calculate_exg(image_rgb: np.ndarray) -> np.ndarray:
    r = image_rgb[:, :, 0].astype(float)
    g = image_rgb[:, :, 1].astype(float)
    b = image_rgb[:, :, 2].astype(float)

    sum_rgb = r + g + b
    r = safe_divide(r, sum_rgb, fill_value=0)
    g = safe_divide(g, sum_rgb, fill_value=0)
    b = safe_divide(b, sum_rgb, fill_value=0)

    return 2 * g - r - b


def find_mask(mask_dir: Path, image_stem: str) -> Path | None:
    """Return the first mask file in mask_dir whose stem matches image_stem."""
    for ext in IMAGE_EXTENSIONS:
        candidate = mask_dir / f"{image_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_leaf_mask(mask_path: Path) -> np.ndarray:
    """Load a white/black mask image and return a boolean array (True = leaf)."""
    mask_img = Image.open(mask_path).convert("RGB")
    mask_arr = np.array(mask_img)
    # White pixels ([255, 255, 255]) are leaf; black ([0, 0, 0]) are background.
    # Channel order doesn't matter for pure white/black images.
    return mask_arr[:, :, 0] > 0


def process_image(image_path: Path, mask_path: Path) -> dict | None:
    try:
        img = Image.open(image_path).convert("RGB")
    except OSError as exc:
        print(f"  Skipping {image_path.name}: {exc}")
        return None

    image_rgb = np.array(img)

    try:
        leaf_mask = load_leaf_mask(mask_path)
    except OSError as exc:
        print(f"  Skipping {image_path.name}: cannot read mask {mask_path.name}: {exc}")
        return None

    leaf_pixels = int(np.sum(leaf_mask))
    if leaf_pixels == 0:
        return None

    exg = calculate_exg(image_rgb)
    below_threshold_mask = (exg < EXG_P20_THRESHOLD) & leaf_mask
    below_threshold_pixels = int(np.sum(below_threshold_mask))
    below_threshold_pct = (below_threshold_pixels / leaf_pixels) * 100

    return {
        "image_name": image_path.name,
        "leaf_pixels": leaf_pixels,
        "below_threshold_pixels": below_threshold_pixels,
        "ExG_P20_below_threshold_pct": below_threshold_pct,
        "ExG_P20_threshold": EXG_P20_THRESHOLD,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate percent of leaf pixels below the ExG threshold."
    )
    parser.add_argument(
        "--images", required=True, type=Path,
        help="Directory of input images to process."
    )
    parser.add_argument(
        "--masks", required=True, type=Path,
        help="Directory of mask images (same basename as input images; white=leaf, black=background)."
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Path for the output CSV file."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.images.exists():
        raise SystemExit(f"Images directory not found: {args.images}")
    if not args.masks.exists():
        raise SystemExit(f"Masks directory not found: {args.masks}")

    images = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"No images found in {args.images}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped_no_mask = 0
    skipped_no_leaf = 0

    for i, img_path in enumerate(images, 1):
        if i % 25 == 0 or i == 1:
            print(f"[{i:3d}/{len(images)}] {img_path.name}")

        mask_path = find_mask(args.masks, img_path.stem)
        if mask_path is None:
            print(f"  No mask found for {img_path.name}, skipping.")
            skipped_no_mask += 1
            continue

        result = process_image(img_path, mask_path)
        if result is None:
            skipped_no_leaf += 1
            continue
        rows.append(result)

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "leaf_pixels",
                "below_threshold_pixels",
                "ExG_P20_below_threshold_pct",
                "ExG_P20_threshold",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print(f"Images processed: {len(rows)}")
    print(f"Images skipped (no mask):      {skipped_no_mask}")
    print(f"Images skipped (no leaf pixels): {skipped_no_leaf}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
