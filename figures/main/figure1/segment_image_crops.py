from __future__ import annotations
import matplotlib.pyplot as plt
import os
import cv2
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile

%matplotlib inline

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

def black_to_transparent(image: np.ndarray | str | Path, output_path: str | Path) -> None:
    """Replace true black [0, 0, 0] pixels with transparent and save as PNG."""
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGBA")
    else:
        # Accept BGR (OpenCV) or RGB numpy array
        if image.shape[2] == 3:
            img = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGBA))
        else:
            img = Image.fromarray(image.astype(np.uint8)).convert("RGBA")
    arr = np.array(img)
    black = (arr[:, :, 0] == 0) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 0)
    arr[black, 3] = 0
    Image.fromarray(arr).save(output_path)


def overlay_disease(image_rgb: np.ndarray, disease_mask: np.ndarray) -> np.ndarray:
    """Create overlay image with diseased pixels highlighted in red."""
    overlay = image_rgb.copy()
    overlay[disease_mask] = [255, 0, 0]
    return overlay

image_crops = ['1688_LeafPhotoA_2025-09-09 09_15_58.077-05_00_1.png', 
              '1338_LeafPhotoA_2025-09-08 10_06_33.224-05_00_0.png', 
              '1314_LeafPhotoA_2025-09-08 09_32_02.288-05_00_2.png', 
              '1064_LeafPhotoA_2025-09-08 12_56_11.568-05_00_1.png', 
              '2836_LeafPhotoA_2025-09-09 10_59_48.765-05_00_3.png', 
              '2087_LeafPhotoA_2025-09-08 12_24_17.487-05_00_2.png']
subdirs = ['device8', 'device4', 'device4', 'device5', 'device2', 'device5']
env = 'ne2025'


for i in range(len(image_crops)):
    image = cv2.imread(str('data/processed/' + env + '/' + subdirs[i] + '/cropped/' + image_crops[i]))
    mask = cv2.imread(str('data/processed/' + env + '/' + subdirs[i] + '/masks_cropped/' + image_crops[i]))
    binary_mask = mask / 255.0
    cv2.imwrite('figures/main/figure1/' + image_crops[i], image * binary_mask)

img_tan = 'figures/main/figure3/2668_LeafPhotoA_2025-09-09 11_41_58.060-05_00_0.png'
img_purple = 'figures/main/figure3/1688_LeafPhotoA_2025-09-09 09_15_58.077-05_00_1.png'
black_to_transparent(img_tan, 'figures/main/figure3/2668_transparent.png')
black_to_transparent(img_purple, 'figures/main/figure3/1688_transparent.png')
    
# image_raw = Image.open('figures/main/figure1/1064_LeafPhotoA_2025-09-08 12_56_11.568-05_00.jpg')
image_raw = cv2.imread('figures/main/figure1/1064_LeafPhotoA_2025-09-08 12_56_11.568-05_00.jpg')
mask = cv2.imread('figures/main/figure1/1064_LeafPhotoA_2025-09-08 12_56_11.568-05_00_mask.png') / 255.0
cv2.imwrite('figures/main/figure1/1064_LeafPhotoA_2025-09-08 12_56_11.568-05_00_segmented.png', image_raw * mask)
image_rgb = image_raw.convert("RGB")

THRESHOLD_FILE = Path("figures/main/figure1/threshold_info.json")
with THRESHOLD_FILE.open("r") as f:
    threshold_config = json.load(f)

EXG_P20_THRESHOLD = threshold_config["thresholds"]["ExG"]["P20"]

exg_image = calculate_exg(np.array(image_rgb))

# Normalize using only unmasked pixels
mask_1ch = mask[:, :, 0]
unmasked = exg_image[mask_1ch == 1].astype(np.float32)
vmin, vmax = unmasked.min(), unmasked.max()

img_norm = exg_image.astype(np.float32)
img_norm = (img_norm - vmin) / (vmax - vmin)

# Apply colormap to get RGBA image
cmap = plt.get_cmap("inferno_r")
img_rgba = cmap(img_norm)  # shape (H, W, 4), values in [0, 1]

# Set masked pixels (where mask == 0) to black
img_rgba[mask_1ch == 0] = [0, 0, 0, 1]  # RGB black, full opacity

# Plot and save
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img_rgba, interpolation="nearest")

sm = plt.cm.ScalarMappable(cmap="inferno_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
plt.colorbar(sm, ax=ax, label="Intensity")
ax.set_title("ExG")
ax.axis("off")

plt.tight_layout()
plt.rcParams.update({'font.size': 14})
plt.savefig("figures/main/figure1/heatmap.svg", dpi=150, bbox_inches="tight")
plt.show()

disease_mask = exg_image < EXG_P20_THRESHOLD
# Create a copy of the RGB image to draw on
overlay = np.array(image_rgb.copy()).astype(np.float32) / 255.0  # normalize if uint8

# Apply red color where bool_array is True
overlay[disease_mask] = [0, 0, 1]  # pure blue
disease_overlay = overlay * mask

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(disease_overlay, interpolation="nearest")
ax.axis("off")

plt.tight_layout()
plt.rcParams.update({'font.size': 14})
plt.savefig("figures/main/figure1/overlay.png", dpi=150, bbox_inches="tight")
plt.show()

gte50pctd_dir = Path('figures/supplemental/gte50pctd/')
images = sorted(gte50pctd_dir.glob("*.jpg"))
for img in images:
    image_raw = cv2.imread(img)
    mask = cv2.imread(str(img).replace('.jpg', '.png')) / 255.0
    cv2.imwrite(str(img).replace('.jpg', '.png').replace('gte50pctd/', 'gte50pctd/segmented/'), image_raw * mask)
