#!/usr/bin/env python3
"""
Generate ExG P20 disease overlays and scores for leaf masks trimmed images.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


THRESHOLD_FILE = Path(__file__).resolve().parents[1] / "threshold_info.json"
with THRESHOLD_FILE.open("r") as f:
    threshold_config = json.load(f)

EXG_P20_THRESHOLD = threshold_config["thresholds"]["ExG"]["P20"]

INPUT_DIR = Path(
    "/home/schnablelab/project2_fieldLeafImaging/Analysis/disease_threshold_analysis/"
    "ne2025_all6080_viz2400_alabama_thresholds/final_lincoln_1.16.26/"
    "leaf masks trimmed"
)
OUTPUT_DIR = Path(
    "/home/schnablelab/project2_fieldLeafImaging/Analysis/disease_threshold_analysis/"
    "ne2025_all6080_viz2400_alabama_thresholds/final_lincoln_1.16.26/"
    "leaf_masks_trimmed_exg_p20_overlays"
)
OUTPUT_CSV = Path(
    "/home/schnablelab/project2_fieldLeafImaging/Analysis/disease_threshold_analysis/"
    "ne2025_all6080_viz2400_alabama_thresholds/final_lincoln_1.16.26/"
    "leaf_masks_trimmed_exg_p20_scores.csv"
)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Safely divide arrays, handling division by zero."""
    result = np.full_like(numerator, fill_value, dtype=float)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    return result


def calculate_exg(image_rgb: np.ndarray) -> np.ndarray:
    """Calculate ExG (Excess Green) vegetation index."""
    r = image_rgb[:, :, 0].astype(float)
    g = image_rgb[:, :, 1].astype(float)
    b = image_rgb[:, :, 2].astype(float)

    sum_rgb = r + g + b
    r = safe_divide(r, sum_rgb, fill_value=0)
    g = safe_divide(g, sum_rgb, fill_value=0)
    b = safe_divide(b, sum_rgb, fill_value=0)

    return 2 * g - r - b


def create_leaf_mask(image_rgb: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Create binary mask to separate leaf from background."""
    gray = np.mean(image_rgb, axis=2)
    return gray > threshold


def overlay_disease(image_rgb: np.ndarray, disease_mask: np.ndarray) -> np.ndarray:
    """Create overlay image with diseased pixels highlighted in red."""
    overlay = image_rgb.copy()
    overlay[disease_mask] = [255, 0, 0]
    return overlay


def process_image(image_path: Path) -> dict[str, float] | None:
    """Calculate disease score and save overlay for a single image."""
    try:
        img = Image.open(image_path).convert("RGB")
    except OSError as exc:
        print(f"  Skipping {image_path.name}: {exc}")
        return None
    image_rgb = np.array(img)

    leaf_mask = create_leaf_mask(image_rgb)
    leaf_pixels = int(np.sum(leaf_mask))
    if leaf_pixels == 0:
        return None

    exg = calculate_exg(image_rgb)
    disease_mask = (exg < EXG_P20_THRESHOLD) & leaf_mask
    disease_pixels = int(np.sum(disease_mask))
    disease_pct = (disease_pixels / leaf_pixels) * 100

    overlay = overlay_disease(image_rgb, disease_mask)
    out_path = OUTPUT_DIR / f"{image_path.stem}_overlay.png"
    Image.fromarray(overlay).save(out_path)

    return {
        "image_name": image_path.name,
        "leaf_pixels": leaf_pixels,
        "disease_pixels": disease_pixels,
        "ExG_P20_disease_pct": disease_pct,
        "ExG_P20_threshold": EXG_P20_THRESHOLD,
    }


def generate_overlays() -> None:
    """Generate overlays and CSV scores for all leaf masks trimmed images."""
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input dir not found: {INPUT_DIR}")

    images = sorted(INPUT_DIR.glob("*.png"))
    if not images:
        raise SystemExit(f"No images found in {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    skipped = 0

    for i, img_path in enumerate(images, 1):
        if i % 25 == 0 or i == 1:
            print(f"[{i:3d}/{len(images)}] {img_path.name}")

        result = process_image(img_path)
        if result is None:
            skipped += 1
            continue
        rows.append(result)

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "leaf_pixels",
                "disease_pixels",
                "ExG_P20_disease_pct",
                "ExG_P20_threshold",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print("ExG P20 overlay generation complete")
    print("=" * 80)
    print(f"Images processed: {len(rows)}")
    print(f"Images skipped (no leaf pixels): {skipped}")
    print(f"Overlays saved to: {OUTPUT_DIR}")
    print(f"Scores saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    generate_overlays()
