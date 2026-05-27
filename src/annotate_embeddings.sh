#!/usr/bin/env bash
# Builds annotation image grids for the top/bottom 16 images per embedding feature.

set -euo pipefail

#CSV="output/sam3_embeddings.csv"
#CSV="output/sam3_embeddings_aamu.csv"
CSV="output/sam3_embeddings_fvsu.csv"
#OUTBASE="output/embedding_annotation_20260512"
#OUTBASE="output/embedding_annotation_lowFI"
#OUTBASE="output/embedding_annotation_lowFI_highH"
#OUTBASE="figures/supplemental/embedding_annotation_al_ga/al"
OUTBASE="figures/supplemental/embedding_annotation_al_ga/ga"
mkdir -p "$OUTBASE"

python3 - <<'PYEOF'
import csv
import os
import re
import shutil
import subprocess
import sys

#CSV      = "output/sam3_embeddings.csv"
CSV="output/sam3_embeddings_aamu.csv"
CSV="output/sam3_embeddings_fvsu.csv"
#OUTBASE  = "output/embedding_annotation_20260512"
#OUTBASE  = "output/embedding_annotation_lowFI"
#OUTBASE="output/embedding_annotation_lowFI_highH"
#OUTBASE="figures/supplemental/embedding_annotation_al_ga/al"
OUTBASE="figures/supplemental/embedding_annotation_al_ga/ga"
N        = 16

EMBEDDINGS = [
"embedding_std_976", "embedding_std_552", "embedding_std_930", "embedding_mean_637", "embedding_std_918", "embedding_std_383",
             "embedding_mean_586", "embedding_mean_968", "embedding_mean_886", "embedding_mean_656", "embedding_mean_698", "embedding_mean_210",
             "embedding_mean_836", "embedding_std_968", "embedding_std_132", "embedding_mean_37",  "embedding_mean_68", "embedding_std_687",
             "embedding_std_793",  "embedding_mean_165", "embedding_mean_582", "embedding_std_821",  "embedding_mean_108", "embedding_mean_119",
             "embedding_mean_989", "embedding_std_839", "embedding_mean_514", "embedding_mean_546", "embedding_std_606", "embedding_mean_984",
             "embedding_mean_792", "embedding_std_981", "embedding_std_270", "embedding_std_594", "embedding_mean_437", "embedding_mean_901",
             "embedding_mean_129", "embedding_std_567", "embedding_mean_930", "embedding_std_983",  "embedding_mean_139", "embedding_std_166",
             "embedding_mean_214", "embedding_std_82",  "embedding_mean_109", "embedding_mean_734", "embedding_std_517"
]
#EMBEDDINGS = ["embedding_std_798", "embedding_std_715", "embedding_std_896", "embedding_std_227", "embedding_std_216", "embedding_std_697", "embedding_std_428", "embedding_std_262",
#"embedding_std_339", "embedding_std_433", "embedding_mean_583", "embedding_mean_511", "embedding_std_999", "embedding_std_69", "embedding_std_992", "embedding_mean_53",
#"embedding_std_300", "embedding_std_786", "embedding_std_415", "embedding_std_66", "embedding_mean_457", "embedding_std_135", "embedding_std_35", "embedding_mean_715",
#"embedding_std_678", "embedding_std_499", "embedding_mean_785", "embedding_std_928", "embedding_std_827", "embedding_std_967", "embedding_std_226", "embedding_std_536", 
#"embedding_mean_850", "embedding_std_498", "embedding_std_64", "embedding_std_71", "embedding_std_192", "embedding_mean_473", "embedding_std_911", "embedding_std_672",
#"embedding_std_965", "embedding_std_48", "embedding_std_393", "embedding_mean_253", "embedding_mean_865", "embedding_std_198", "embedding_std_302" 
#]
#EMBEDDINGS = ["embedding_std_488", "embedding_std_132", "embedding_mean_402", "embedding_mean_633", "embedding_std_345", "embedding_mean_806", "embedding_std_478", "embedding_mean_653",
#"embedding_mean_372", "embedding_mean_42"
#]
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
