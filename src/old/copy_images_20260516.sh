#!/bin/bash
shopt -s globstar nullglob

# Copy images listed in the score file from source to destination,
# converting the _leaf.png suffix to .jpg

LIST_FILE="output/images_to_score_aamu_sample190.txt"
SRC_DIR="data/aamu2025"
DST_DIR="data/aamu2025/images_to_score_20260516"

# Create destination directory if it doesn't exist
mkdir -p "$DST_DIR"

copied=0
missing=0

while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines
    [[ -z "$line" ]] && continue

    # Replace _leaf.png suffix with .jpg
    jpg_name="${line%_leaf.png}.jpg"

    # Glob-expand: find the file anywhere under SRC_DIR
    matches=( "$SRC_DIR"/**/"$jpg_name" )

    if [[ -f "${matches[0]}" ]]; then
        cp "${matches[0]}" "$DST_DIR/$jpg_name"
        ((copied++))
    else
        echo "WARNING: Not found: $jpg_name"
        ((missing++))
    fi

done < "$LIST_FILE"

echo "Done. Copied: $copied | Not found: $missing"
