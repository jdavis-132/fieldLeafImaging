#!/usr/bin/env bash
# Builds annotation image grids for the top/bottom 16 images per embedding feature.

set -euo pipefail

CSV="output/sam3_embeddings.csv"
OUTBASE="output/embedding_annotation_lowFI"

mkdir -p "$OUTBASE"

python3 - <<'PYEOF'
import csv
import os
import re
import shutil
import subprocess
import sys

CSV      = "output/sam3_embeddings.csv"
OUTBASE  = "output/embedding_annotation_lowFI_highH"
N        = 16

EMBEDDINGS = [
    "embedding_std_488", "embedding_mean_402", "embedding_mean_633",
    "embedding_std_345", "embedding_mean_806", "embedding_std_478",
    "embedding_mean_653", "embedding_mean_372", "embedding_mean_42",
    "embedding_mean_10"
]

KEEP_FILES = {
    "ne2025":   "data/ne2025/images_keep_all.csv",
    "fvsu2025": "data/fvsu2025/image_ids_keep.txt",
    "aamu2025": "data/aamu2025/image_ids_keep.txt",
}

def load_keep_set(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())

keep_sets = {ds: load_keep_set(p) for ds, p in KEEP_FILES.items()}

SUFFIX_RE = re.compile(r"-05_00_\d+\.png$")

def image_id_from_path(image_path):
    return SUFFIX_RE.sub("", os.path.basename(image_path))

def keep_row(row):
    path = row["image_path"]
    for ds, keep in keep_sets.items():
        if ds in path:
            return image_id_from_path(path) in keep
    return False

print(f"Reading {CSV}...")
with open(CSV, newline="") as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader if keep_row(r)]
print(f"  {len(rows)} rows loaded (after keep-list filtering).")

missing_cols = [e for e in EMBEDDINGS if e not in rows[0]]
if missing_cols:
    print(f"ERROR: columns not found in CSV: {missing_cols}", file=sys.stderr)
    sys.exit(1)

for emb in EMBEDDINGS:
    sorted_rows = sorted(rows, key=lambda r: float(r[emb]))
    groups = [("low", sorted_rows[:N]), ("high", sorted_rows[-N:])]

    for level, selected in groups:
        subdir = os.path.join(OUTBASE, emb, level)
        os.makedirs(subdir, exist_ok=True)

        # Copy images
        copied = 0
        for row in selected:
            src = row["image_path"]
            if os.path.exists(src):
                shutil.copy2(src, subdir)
                copied += 1
            else:
                print(f"  WARNING: not found – {src}", file=sys.stderr)
        print(f"[{emb}/{level}] copied {copied}/{N} images")

        # Build grid; script saves to output/<folder_name>_grid.png
        result = subprocess.run(
            ["python3", "src/make_image_grid.py", subdir, "--width", "6.5"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR (make_image_grid): {result.stderr.strip()}", file=sys.stderr)
            continue

        # Relocate to desired path
        grid_src = os.path.join("output", f"{level}_grid.png")
        grid_dst = os.path.join(OUTBASE, f"{emb}_{level}.png")
        if os.path.exists(grid_src):
            shutil.move(grid_src, grid_dst)
            print(f"  -> {grid_dst}")
        else:
            print(f"  WARNING: expected grid not found at {grid_src}", file=sys.stderr)

print("Done.")
PYEOF
