import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def clamp_seed(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """Keep the seed inside the image bounds."""
    x = min(max(0, x), width - 1)
    y = min(max(0, y), height - 1)
    return x, y


def flood_remove(image: np.ndarray, seed: tuple[int, int], tolerance: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Flood fill from the seed, masking pixels within the tolerance and removing them from the image.
    Returns the updated image and the flood mask (uint8, 255 where filled).
    """
    height, width = image.shape[:2]
    seed = clamp_seed(seed[0], seed[1], width, height)

    # Mask for floodFill must be 2 pixels larger than the image in each dimension.
    mask = np.zeros((height + 2, width + 2), np.uint8)
    flags = cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | 4 | (255 << 8)
    lo = (tolerance, tolerance, tolerance)
    up = (tolerance, tolerance, tolerance)

    # The image copy prevents modifying the working image while we harvest the mask.
    cv2.floodFill(image.copy(), mask, seedPoint=seed, newVal=(0, 0, 0), loDiff=lo, upDiff=up, flags=flags)
    fill_mask = mask[1:-1, 1:-1]

    updated = image.copy()
    updated[fill_mask == 255] = 0
    return updated, fill_mask


def largest_component_touching_sides(binary_mask: np.ndarray) -> tuple[int | None, np.ndarray | None]:
    """Find the largest component that touches both the left and right image borders."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    height, width = binary_mask.shape

    best_label = None
    best_area = 0
    for label in range(1, num_labels):  # 0 is background
        left = stats[label, cv2.CC_STAT_LEFT]
        comp_width = stats[label, cv2.CC_STAT_WIDTH]
        area = stats[label, cv2.CC_STAT_AREA]
        touches_left = left == 0
        touches_right = (left + comp_width) >= width

        if touches_left and touches_right and area > best_area:
            best_area = area
            best_label = label

    if best_label is None:
        return None, None

    return best_label, (labels == best_label)

image_path ='figures/supplemental/1151_LeafPhotoA_2025-09-08 16_59_29.011-05_00.jpg'
tolerance1=50
tolerance2=50
down_from_top=750
up_from_bottom=20
trim_left=300
trim_right=100
card_height=1310
card_width=750

image = cv2.imread(str(image_path))
height, width = image.shape[:2]
seed1 = (width // 2, down_from_top)
seed2 = (width // 2, height - up_from_bottom)
points = np.array([seed1, seed2])

working, _ = flood_remove(image, seed1, tolerance1)
working, _ = flood_remove(working, seed2, tolerance2)

working_rgb = cv2.cvtColor(working, cv2.COLOR_BGR2RGB)
plt.imshow(working_rgb)
plt.scatter(points[:, 0], points[:, 1], marker = 'o', s=50, color='cyan')
plt.axis('off')
plt.savefig('figures/supplemental/flood_remove.png', bbox_inches='tight', pad_inches=0)

foreground_mask = np.any(working != 0, axis=2).astype(np.uint8)
_, leaf_mask = largest_component_touching_sides(foreground_mask)

binary_mask = leaf_mask.astype(np.uint8)
image_connected = image * binary_mask[:, :, np.newaxis]
cv2.imwrite('figures/supplemental/largest_connected_component.png', image_connected)

leaf_mask[:, :trim_left] = False
leaf_mask[:, width - trim_right :] = False

# Keep only the largest remaining component after trimming.
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_mask.astype(np.uint8), 
                                                                connectivity=8)
best_label = None
best_area = 0
for label in range(1, num_labels):  # skip background
    area = stats[label, cv2.CC_STAT_AREA]
    if area > best_area:
        best_area = area
        best_label = label

leaf_mask = labels == best_label

binary_mask = (leaf_mask).astype(np.uint8)
image_connected = image * binary_mask[:, :, np.newaxis]
cv2.imwrite('figures/supplemental/final_mask.png', image_connected)
