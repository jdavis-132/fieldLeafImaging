#!/usr/bin/env python3
"""Create a grid image from all images in a folder."""

import argparse
import math
import os

from PIL import Image, ImageOps


THUMB_SIZE = (400, 400)


def _to_pixels(value, unit, dpi):
    """Convert a measurement in inches or mm to pixels at the given DPI."""
    if unit == "mm":
        value = value / 25.4
    return round(value * dpi)


def make_grid(folder, width=None, height=None, unit="in", dpi=300):
    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not files:
        raise ValueError(f"No images found in {folder}")

    n = len(files)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Determine cell size from first image after resizing
    sample = Image.open(os.path.join(folder, files[0]))
    sample.thumbnail(THUMB_SIZE, Image.LANCZOS)
    w, h = sample.size
    sample.close()

    grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))

    for i, fname in enumerate(files):
        img = Image.open(os.path.join(folder, fname))
        img.thumbnail(THUMB_SIZE, Image.LANCZOS)
        r, c = divmod(i, cols)
        grid.paste(img, (c * w, r * h))
        img.close()

    if width is not None or height is not None:
        target_w = _to_pixels(width, unit, dpi) if width is not None else None
        target_h = _to_pixels(height, unit, dpi) if height is not None else None
        orig_w, orig_h = grid.size
        if target_w is None:
            target_w = round(orig_w * target_h / orig_h)
        elif target_h is None:
            target_h = round(orig_h * target_w / orig_w)
        grid = grid.resize((target_w, target_h), Image.LANCZOS)

    folder_name = os.path.basename(folder.rstrip("/"))
    output_path = os.path.join("output", f"{folder_name}_grid.png")
    save_kwargs = {"dpi": (dpi, dpi)} if (width is not None or height is not None) else {}
    grid.save(output_path, **save_kwargs)
    print(f"Saved {n} images in {cols}x{rows} grid to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a grid from images in a folder.")
    parser.add_argument("folder", help="Path to folder containing images")
    parser.add_argument("--width", type=float, default=None, help="Output width in inches or mm")
    parser.add_argument("--height", type=float, default=None, help="Output height in inches or mm")
    parser.add_argument("--unit", choices=["in", "mm"], default="in", help="Unit for --width/--height (default: in)")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution for physical size conversion (default: 300)")
    args = parser.parse_args()
    make_grid(args.folder, width=args.width, height=args.height, unit=args.unit, dpi=args.dpi)
