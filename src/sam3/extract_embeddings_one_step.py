#!/usr/bin/env python3
"""Segment raw leaf images, crop in memory, and write SAM3 crop embeddings.

This is a single-step version of the older preprocessing plus
``src/sam3/extract_embeddings.py`` workflow. It does not write masks, crops, or
normalized intermediates. The output is one CSV row per generated crop.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder import segment_leaf  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def default_model_path() -> str:
    local = Path("/home/james/leaf_imaging/SAM3")
    if local.exists():
        return str(local)
    return "src/sam3"


def find_images(pattern: str) -> list[Path]:
    paths = [
        Path(p)
        for p in glob.glob(pattern, recursive=True)
        if Path(p).suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths)


def combine_masks(masks: torch.Tensor) -> np.ndarray | None:
    if len(masks) == 0:
        return None
    masks_np = masks.detach().cpu().numpy()
    combined = np.any(masks_np > 0, axis=0)
    return combined.astype(np.uint8)


def crop_name(image_path: Path, crop_index: int) -> str:
    return f"{image_path.stem}_{crop_index}.png"


def crops_from_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    step: int,
    x_dim: int,
    y_dim: int,
) -> list[np.ndarray]:
    """Return aligned BGR crops matching the legacy crop geometry."""
    img_height, img_width = image_bgr.shape[:2]
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        raise ValueError("No non-zero pixels found in mask")

    points = np.column_stack((x_coords, y_coords))
    pca = PCA(n_components=2)
    pca.fit(points)
    principal_axis = pca.components_[0]
    perpendicular_axis = pca.components_[1]
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)

    temp_principal_proj = np.dot(points - [center_x, center_y], principal_axis)
    temp_perpendicular_proj = np.dot(points - [center_x, center_y], perpendicular_axis)
    principal_extent = np.max(temp_principal_proj) - np.min(temp_principal_proj)
    perpendicular_extent = np.max(temp_perpendicular_proj) - np.min(temp_perpendicular_proj)
    if perpendicular_extent > principal_extent:
        principal_axis, perpendicular_axis = perpendicular_axis, principal_axis

    projections = np.dot(points - [center_x, center_y], principal_axis)
    min_proj = np.min(projections)
    max_proj = np.max(projections)
    start_point = np.array([center_x, center_y]) + min_proj * principal_axis
    total_distance = max_proj - min_proj
    all_projections = np.dot(points - [center_x, center_y], principal_axis)

    half_x = x_dim / 2
    half_y = y_dim / 2
    local_corners = np.array(
        [[-half_x, -half_y], [half_x, -half_y], [half_x, half_y], [-half_x, half_y]],
        dtype=np.float32,
    )
    dst_points = np.array(
        [[0, 0], [x_dim - 1, 0], [x_dim - 1, y_dim - 1], [0, y_dim - 1]],
        dtype=np.float32,
    )

    def calculate_corners(center: np.ndarray) -> np.ndarray:
        corners = [
            center + lc[0] * principal_axis + lc[1] * perpendicular_axis
            for lc in local_corners
        ]
        return np.array(corners, dtype=np.float32)

    def corners_within_bounds(corners: np.ndarray) -> bool:
        return bool(
            np.all(corners[:, 0] >= 0)
            and np.all(corners[:, 0] < img_width)
            and np.all(corners[:, 1] >= 0)
            and np.all(corners[:, 1] < img_height)
        )

    crops: list[np.ndarray] = []
    current_distance = 0
    while current_distance + x_dim <= total_distance:
        window_center_on_axis = start_point + (current_distance + x_dim / 2) * principal_axis
        window_min_proj = min_proj + current_distance
        window_max_proj = min_proj + current_distance + x_dim
        in_window = (all_projections >= window_min_proj) & (all_projections <= window_max_proj)
        if not np.any(in_window):
            current_distance += step
            continue

        window_points = points[in_window]
        perp_projections = np.dot(window_points - [center_x, center_y], perpendicular_axis)
        mean_perp_offset = np.mean(perp_projections)
        window_center = window_center_on_axis + mean_perp_offset * perpendicular_axis
        corners = calculate_corners(window_center)

        if corners_within_bounds(corners):
            transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)
            crops.append(cv2.warpPerspective(image_bgr, transform_matrix, (x_dim, y_dim)))

        current_distance += step

    return crops


class OneStepSam3Extractor:
    def __init__(self, model_path: str, device: str, dtype: str = "float32") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but unavailable; falling back to CPU")
            device = "cpu"
        self.device = device
        self.processor = Sam3Processor.from_pretrained(model_path)
        self.model = Sam3Model.from_pretrained(model_path)
        if dtype == "float16" and self.device == "cuda":
            self.model = self.model.half()
        self.model = self.model.to(self.device).eval()

    def sam3_mask(
        self,
        image_rgb: Image.Image,
        prompt: str,
        threshold: float,
        mask_threshold: float,
    ) -> np.ndarray | None:
        inputs = self.processor(images=image_rgb, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        return combine_masks(results["masks"])

    def embedding(self, crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(crop_rgb)
        inputs = self.processor(images=image, text="", return_tensors="pt").to(self.device)
        with torch.inference_mode():
            vision_outputs = self.model.get_vision_features(pixel_values=inputs.pixel_values)
        features = vision_outputs.last_hidden_state

        if features.dim() == 4:
            if features.shape[-1] >= features.shape[1]:
                # Current transformers SAM3 exposes vision states as [B, H, W, C].
                pooled_mean = features.mean(dim=[1, 2])
                pooled_std = features.std(dim=[1, 2])
            else:
                pooled_mean = features.mean(dim=[2, 3])
                pooled_std = features.std(dim=[2, 3])
        elif features.dim() == 3:
            pooled_mean = features.mean(dim=1)
            pooled_std = features.std(dim=1)
        elif features.dim() == 2:
            pooled_mean = features
            pooled_std = torch.zeros_like(features)
        else:
            pooled_mean = features.flatten(1)
            pooled_std = torch.zeros_like(pooled_mean)

        mean = pooled_mean.squeeze().float().cpu().numpy()
        std = pooled_std.squeeze().float().cpu().numpy()
        return np.ravel(mean), np.ravel(std)


def valid_mask(mask: np.ndarray | None, min_pixels: int, max_pixels: int) -> bool:
    if mask is None:
        return False
    pixels = int(np.sum(mask > 0))
    return min_pixels <= pixels <= max_pixels


def process_image(
    image_path: Path,
    extractor: OneStepSam3Extractor,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return [], {
            "image_path": str(image_path),
            "status": "failed_read",
            "segmentation_method": "None",
            "mask_pixels": 0,
            "n_crops": 0,
        }

    mask = segment_leaf.process_single(
        image_path,
        tolerance1=args.tolerance1,
        tolerance2=args.tolerance2,
        down_from_top=args.down_from_top,
        up_from_bottom=args.up_from_bottom,
        card_height=args.card_height,
        card_width=args.card_width,
        trim_left=args.trim_left,
        trim_right=args.trim_right,
    )
    segmentation_method = "CV2"

    if not valid_mask(mask, args.mask_pixels_min, args.mask_pixels_max):
        if args.no_sam3_fallback:
            mask = None
        else:
            image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            mask = extractor.sam3_mask(
                image_rgb,
                prompt=args.prompt,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
            )
            segmentation_method = "SAM3"

    if not valid_mask(mask, args.mask_pixels_min, args.mask_pixels_max):
        return [], {
            "image_path": str(image_path),
            "status": "failed_segmentation",
            "segmentation_method": segmentation_method,
            "mask_pixels": int(np.sum(mask > 0)) if mask is not None else 0,
            "n_crops": 0,
        }

    crops = crops_from_mask(
        image_bgr=image_bgr,
        mask=mask,
        step=args.step,
        x_dim=args.crop_width,
        y_dim=args.crop_height,
    )
    if not crops:
        return [], {
            "image_path": str(image_path),
            "status": "failed_cropping",
            "segmentation_method": segmentation_method,
            "mask_pixels": int(np.sum(mask > 0)),
            "n_crops": 0,
        }

    rows: list[dict[str, object]] = []
    for crop_index, crop_bgr in enumerate(crops):
        mean, std = extractor.embedding(crop_bgr)
        row: dict[str, object] = {
            "image_path": crop_name(image_path, crop_index),
            "source_image_path": str(image_path),
            "crop_index": crop_index,
            "segmentation_method": segmentation_method,
            "mask_pixels": int(np.sum(mask > 0)),
        }
        row.update({f"embedding_mean_{i}": float(v) for i, v in enumerate(mean)})
        row.update({f"embedding_std_{i}": float(v) for i, v in enumerate(std)})
        rows.append(row)

    return rows, {
        "image_path": str(image_path),
        "status": "ok",
        "segmentation_method": segmentation_method,
        "mask_pixels": int(np.sum(mask > 0)),
        "n_crops": len(crops),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_pattern", help="Raw image glob, e.g. 'data/ne2025/device*/*.jpg'")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Final embedding CSV")
    parser.add_argument(
        "-m",
        "--model",
        default=default_model_path(),
        help="SAM3 model directory",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16"],
        help="Use float16 model weights on CUDA to reduce memory",
    )
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--crop-width", type=int, default=1000)
    parser.add_argument("--crop-height", type=int, default=2000)
    parser.add_argument("--mask-pixels-min", type=int, default=750000)
    parser.add_argument("--mask-pixels-max", type=int, default=7500000)
    parser.add_argument("--tolerance1", type=int, default=50)
    parser.add_argument("--tolerance2", type=int, default=50)
    parser.add_argument("--down-from-top", type=int, default=750)
    parser.add_argument("--up-from-bottom", type=int, default=20)
    parser.add_argument("--trim-left", type=int, default=300)
    parser.add_argument("--trim-right", type=int, default=100)
    parser.add_argument("--card-height", type=int, default=1310)
    parser.add_argument("--card-width", type=int, default=750)
    parser.add_argument("--prompt", default="leaf", help="SAM3 fallback segmentation prompt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--no-sam3-fallback",
        action="store_true",
        help="Use only CV2 segmentation; skip images where CV2 mask fails.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional per-raw-image status CSV. This is a final diagnostic output, not an intermediate.",
    )
    parser.add_argument(
        "--legacy-columns-only",
        action="store_true",
        help="Write only image_path plus embedding columns, matching output/sam3_embeddings.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = find_images(args.image_pattern)
    if not image_paths:
        raise SystemExit(f"No images found for pattern: {args.image_pattern}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(image_paths)} raw images")
    print(f"Loading SAM3 from {args.model}")
    extractor = OneStepSam3Extractor(args.model, args.device, args.dtype)

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for image_path in tqdm(image_paths, desc="Raw images", unit="img"):
        try:
            rows, summary = process_image(image_path, extractor, args)
        except Exception as exc:
            rows = []
            summary = {
                "image_path": str(image_path),
                "status": "failed_exception",
                "segmentation_method": "Unknown",
                "mask_pixels": 0,
                "n_crops": 0,
                "error": str(exc),
            }
        all_rows.extend(rows)
        summaries.append(summary)
        if extractor.device == "cuda" and len(summaries) % 100 == 0:
            torch.cuda.empty_cache()

    if not all_rows:
        raise SystemExit("No embeddings were extracted")

    df = pd.DataFrame(all_rows)
    if args.legacy_columns_only:
        cols = ["image_path"] + [
            c for c in df.columns if c.startswith("embedding_mean_") or c.startswith("embedding_std_")
        ]
        df = df[cols]
    else:
        embedding_cols = [
            c for c in df.columns if c.startswith("embedding_mean_") or c.startswith("embedding_std_")
        ]
        metadata_cols = [c for c in df.columns if c not in embedding_cols]
        df = df[metadata_cols + embedding_cols]
    df.to_csv(args.output, index=False)

    if args.summary_output:
        with args.summary_output.open("w", newline="") as f:
            fieldnames = sorted({k for row in summaries for k in row})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)

    summary_df = pd.DataFrame(summaries)
    status_counts = summary_df["status"].value_counts().to_dict()
    method_counts = summary_df["segmentation_method"].value_counts().to_dict()
    print(f"Wrote {len(df)} crop embeddings to {args.output}")
    print(f"Raw image status counts: {status_counts}")
    print(f"Segmentation method counts: {method_counts}")


if __name__ == "__main__":
    main()
