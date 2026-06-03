"""
For each device directory under data/processed/ne2025/device[0-9]/:
  - Read images from cropped/
  - Read matching mask from masks_cropped/ (same base filename)
  - Multiply each RGB channel by the corresponding mask channel / 255.0
  - Set pixels that are true black [0,0,0] to transparent (alpha=0)
  - Save as RGBA PNG to cropped_transparent_bg/
"""

import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data" / "processed" / "ne2025"

def process_device(device_dir: Path):
    cropped_dir = device_dir / "cropped"
    mask_dir = device_dir / "masks_cropped"
    out_dir = device_dir / "cropped_transparent_bg"

    if not cropped_dir.exists():
        print(f"  Skipping {device_dir.name}: no cropped/ directory")
        return
    if not mask_dir.exists():
        print(f"  Skipping {device_dir.name}: no masks_cropped/ directory")
        return

    out_dir.mkdir(exist_ok=True)

    images = sorted(cropped_dir.glob("*.png")) + sorted(cropped_dir.glob("*.jpg")) + sorted(cropped_dir.glob("*.jpeg"))

    missing_masks = 0
    processed = 0

    for img_path in images:
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            print(f"  WARNING: no mask for {img_path.name}")
            missing_masks += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)   # BGR, uint8
        mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR) # BGR, uint8

        if img is None:
            print(f"  WARNING: could not read image {img_path.name}")
            continue
        if mask is None:
            print(f"  WARNING: could not read mask {mask_path.name}")
            continue

        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Multiply each channel by the corresponding mask channel / 255.0
        img_float = img.astype(np.float32)
        mask_float = mask.astype(np.float32) / 255.0
        blended = np.clip(img_float * mask_float, 0, 255).astype(np.uint8)

        # Build RGBA: alpha=0 where all three channels are 0 (true black), else 255
        alpha = np.where(
            (blended[:, :, 0] == 0) & (blended[:, :, 1] == 0) & (blended[:, :, 2] == 0),
            0, 255
        ).astype(np.uint8)

        rgba = cv2.cvtColor(blended, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha

        out_path = out_dir / (img_path.stem + ".png")
        cv2.imwrite(str(out_path), rgba)
        processed += 1

    print(f"  {device_dir.name}: {processed} images processed, {missing_masks} missing masks")

def main():
    device_dirs = sorted(BASE.glob("device[0-9]"))
    if not device_dirs:
        print(f"No device directories found under {BASE}")
        return

    for device_dir in device_dirs:
        print(f"Processing {device_dir.name}...")
        process_device(device_dir)

    print("Done.")

if __name__ == "__main__":
    main()
