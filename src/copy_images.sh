 (    dest="data/aamu2025/images_to_score_test" && \
      not_found=0 && found=0 && \
      while IFS= read -r fname; do
        # Convert _leaf.png -> .jpg
        base="${fname%_leaf.png}.jpg"
        src=$(find data/aamu2025/block[0-9]/ -maxdepth 1 -name "$base" 2>/dev/null | head -1)
        if [ -n "$src" ]; then
          cp "$src" "$dest/"
          ((found++))
        else
          echo "NOT FOUND: $base"
          ((not_found++))
        fi
      done < output/aamu2025_images_to_score_30.csv && \
      echo "Copied: $found, Not found: $not_found")
