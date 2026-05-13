#!/usr/bin/env bash
# Builds annotation image grids for the top/bottom 16 images per embedding feature.

set -euo pipefail

CSV="output/sam3_embeddings.csv"
OUTBASE="output/embedding_annotation_lowFI"

mkdir -p "$OUTBASE"

python3 - <<'PYEOF'
import csv
import os
import shutil
import subprocess
import sys

CSV      = "output/sam3_embeddings.csv"
OUTBASE  = "output/embedding_annotation_lowFI"
N        = 16

EMBEDDINGS = [
    "embedding_std_416",  "embedding_std_706",  "embedding_std_722",  "embedding_std_654",
    "embedding_std_825",  "embedding_std_992",  "embedding_std_173",  "embedding_mean_964",
    "embedding_std_300",  "embedding_std_302",  "embedding_std_69",   "embedding_std_135",
    "embedding_mean_451", "embedding_mean_799", "embedding_mean_632", "embedding_std_339",
    "embedding_std_390",  "embedding_mean_721", "embedding_std_569",  "embedding_std_719",
    "embedding_std_941",  "embedding_std_866",  "embedding_std_465",  "embedding_std_993",
    "embedding_mean_865", "embedding_std_933",  "embedding_std_262",  "embedding_std_1004",
    "embedding_std_536",  "embedding_std_12",   "embedding_std_716",  "embedding_std_429",
    "embedding_mean_609", "embedding_mean_253", "embedding_std_35",   "embedding_std_36",
    "embedding_std_14",   "embedding_std_672",  "embedding_std_227",  "embedding_std_51",
    "embedding_std_192",  "embedding_std_437",  "embedding_std_190",  "embedding_std_393",
    "embedding_std_198",  "embedding_std_798",  "embedding_std_84",
]

print(f"Reading {CSV}...")
with open(CSV, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f"  {len(rows)} rows loaded.")

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
