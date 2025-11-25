import argparse
from pathlib import Path
import cv2
import numpy as np


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
  
def process_single(image_path, tolerance1=50, tolerance2=50, down_from_top=750, up_from_bottom=20):
    """Process a single image and return a binary mask."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image at {image_path}")
        return None

    height, width = image.shape[:2]
    seed1 = (width // 2, down_from_top)
    seed2 = (width // 2, height - up_from_bottom)

    working, _ = flood_remove(image, seed1, tolerance1)
    working, _ = flood_remove(working, seed2, tolerance2)

    foreground_mask = np.any(working != 0, axis=2).astype(np.uint8)
    _, leaf_mask = largest_component_touching_sides(foreground_mask)
    if leaf_mask is None:
        print(f"No component touches both borders after removal for {image_path}")
        return None

    # Trim noisy edges near the borders.
    trim = 100
    if trim * 2 >= width:
        print(f"Image {image_path} is too narrow for trim={trim}")
        return None
    leaf_mask[:, :trim] = False
    leaf_mask[:, width - trim :] = False

    # Keep only the largest remaining component after trimming.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_mask.astype(np.uint8), connectivity=8)
    best_label = None
    best_area = 0
    for label in range(1, num_labels):  # skip background
        area = stats[label, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_label = label
    if best_label is None:
        print(f"No component remains after trimming for {image_path}")
        return None
    leaf_mask = labels == best_label

    # Create binary mask: white (255) for leaf pixels, black (0) for background
    binary_mask = (leaf_mask).astype(np.uint8)
    # cv2.imwrite(str(out_path), binary_mask)
    # print(f"Wrote {out_path}")
    return binary_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment leaf(s) by removing two flood-filled regions and keeping the largest component touching both sides.")
    parser.add_argument("image", type=Path, help="Path to the input image or a directory of images.")
    parser.add_argument("--tolerance1", type=int, default=50, help="Color tolerance for the first flood fill (top-middle seed).")
    parser.add_argument("--tolerance2", type=int, default=50, help="Color tolerance for the second flood fill (bottom-middle seed).")
    parser.add_argument("--down-from-top", type=int, default=750, help="Pixels down from the top for the first seed (x is centered).")
    parser.add_argument("--up-from-bottom", type=int, default=20, help="Pixels up from the bottom for the second seed (x is centered).")
    parser.add_argument("--output-prefix", type=str, default="leaf_segmentation", help="Prefix for output files when a single image is provided.")
    parser.add_argument("--output-dir", type=Path, default=Path("demo2_leaves"), help="Directory to write outputs when processing a folder.")
    args = parser.parse_args()

    input_path = args.image
    if input_path.is_dir():
        success = False
        out_dir = args.output_dir
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        for img_path in sorted(input_path.iterdir()):
            if img_path.suffix.lower() not in extensions:
                continue
            out_file = out_dir / f"{img_path.stem}_leaf.png"
            if process_single(img_path, out_file):
                success = True
        if not success:
            raise SystemExit("No images were processed successfully.")
    else:
        out_file = Path(args.output_prefix).with_suffix(".leaf.png")
        if not process_single(input_path, out_file):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
