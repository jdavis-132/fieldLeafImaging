#!/usr/bin/env python3
"""Create a grid image from all images in a folder."""

import argparse
import math
import os

from PIL import Image, ImageOps


THUMB_SIZE = (400, 400)


def make_grid(folder):
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

    folder_name = os.path.basename(folder.rstrip("/"))
    output_path = os.path.join("output", f"{folder_name}_grid.png")
    grid.save(output_path)
    print(f"Saved {n} images in {cols}x{rows} grid to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a grid from images in a folder.")
    parser.add_argument("folder", help="Path to folder containing images")
    args = parser.parse_args()
    make_grid(args.folder)
