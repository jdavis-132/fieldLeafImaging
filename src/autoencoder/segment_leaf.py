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


def remove_upper_components(mask, max_area_ratio=0.15, min_rectangularity=0.6,
                           upper_region_height=1000):
    """
    Remove small rectangular components in the upper region (e.g., color reference cards).

    Args:
        mask: Binary mask
        max_area_ratio: Maximum area as ratio of total mask area to be considered removable
        min_rectangularity: Minimum area/bounding_box ratio to be considered rectangular
        upper_region_height: Height of upper region to check (pixels from top)

    Returns:
        Cleaned mask with upper rectangular components removed
    """
    cleaned_mask = mask.copy()
    height, width = mask.shape

    # Find all components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    if num_labels <= 1:
        return cleaned_mask

    # Find total mask area
    total_area = np.sum(mask > 0)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        # Check if component is in upper region
        if y > upper_region_height:
            continue

        # Check if component is small relative to total mask
        area_ratio = area / total_area if total_area > 0 else 0
        if area_ratio > max_area_ratio:
            continue

        # Check if component is rectangular
        bbox_area = w * h
        rectangularity = area / bbox_area if bbox_area > 0 else 0

        if rectangularity >= min_rectangularity:
            # Remove this component
            cleaned_mask[labels == label] = 0

    return cleaned_mask


def remove_color_card_by_detection(image, mask, card_width_range=(200, 600),
                                   card_height_range=(300, 800),
                                   upper_region_height=1500):
    """
    Detect and remove color reference card from mask using color-based detection.

    Args:
        image: Original BGR image
        mask: Binary mask
        card_width_range: Expected width range of the card in pixels
        card_height_range: Expected height range of the card in pixels
        upper_region_height: Height of upper region where card is expected

    Returns:
        Cleaned mask with color card removed
    """
    cleaned_mask = mask.copy()
    height, width = image.shape[:2]

    # Focus on upper region of the image
    upper_region = image[:upper_region_height, :]
    upper_mask = mask[:upper_region_height, :]

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Check if size matches expected card dimensions
        if (card_width_range[0] <= w <= card_width_range[1] and
            card_height_range[0] <= h <= card_height_range[1]):

            # Check rectangularity
            area = cv2.contourArea(contour)
            bbox_area = w * h
            if bbox_area == 0:
                continue
            rectangularity = area / bbox_area

            if rectangularity > 0.7:  # Highly rectangular
                # Remove this region from the mask
                cleaned_mask[y:y+h, x:x+w] = 0

    return cleaned_mask
  
def process_single(image_path, tolerance1=50, tolerance2=50, tolerance3=50,
                   down_from_top=750, up_from_bottom=20, trim_left=300, trim_right=100,
                   remove_upper_right=True, upper_right_x_offset=150, upper_right_y=400,
                   remove_upper_rectangles=True, max_area_ratio=0.15, min_rectangularity=0.6,
                   upper_region_height=1000,
                   card_region_height=1200, card_region_width=900):
    """Process a single image and return a binary mask."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image at {image_path}")
        return None

    height, width = image.shape[:2]
    seed1 = (width // 2, down_from_top)
    seed2 = (width // 2, height - up_from_bottom)
    # Seed in upper-right to remove color reference card and surrounding white area
    seed3 = (width - upper_right_x_offset, upper_right_y)

    working, _ = flood_remove(image, seed1, tolerance1)
    working, _ = flood_remove(working, seed2, tolerance2)

    # Remove upper-right region (color card + white background)
    if remove_upper_right:
        working, _ = flood_remove(working, seed3, tolerance3)

    foreground_mask = np.any(working != 0, axis=2).astype(np.uint8)

    # Remove color reference card BEFORE finding components
    # Strategy: Remove non-green (white/colored) pixels from upper region while preserving green leaf
    if remove_upper_rectangles:
        # Focus on upper region where card and white surface are located
        upper_region_top = 0
        upper_region_bottom = card_region_height

        # Extract the upper region from original image
        upper_region = image[upper_region_top:upper_region_bottom, :]
        upper_mask = foreground_mask[upper_region_top:upper_region_bottom, :]

        # Convert to HSV to identify green (leaf) vs non-green (card/white surface)
        hsv_upper = cv2.cvtColor(upper_region, cv2.COLOR_BGR2HSV)

        # Create mask for green leaf pixels
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        green_pixels = cv2.inRange(hsv_upper, lower_green, upper_green)

        # In the upper region: keep only pixels that are BOTH in foreground AND green
        # This removes white surface and colored card while preserving leaf
        filtered_upper_mask = cv2.bitwise_and(upper_mask, green_pixels)

        # Replace the upper region in the foreground mask with filtered version
        foreground_mask[upper_region_top:upper_region_bottom, :] = filtered_upper_mask

    _, leaf_mask = largest_component_touching_sides(foreground_mask)
    if leaf_mask is None:
        print(f"No component touches both borders after removal for {image_path}")
        return None

     # Trim noisy edges near the borders

    if trim_left + trim_right >= width:
      print(f"Image {image_path} is too narrow for trim_left={trim_left} + trim_right={trim_right}")
      return False

    leaf_mask[:, :trim_left] = False
    leaf_mask[:, width - trim_right :] = False

    # Remove rectangular components in upper region (e.g., color reference cards)
    if remove_upper_rectangles:
        leaf_mask = remove_upper_components(leaf_mask, max_area_ratio=max_area_ratio,
                                           min_rectangularity=min_rectangularity,
                                           upper_region_height=upper_region_height)
        # Also try color-based card detection
        leaf_mask = remove_color_card_by_detection(image, leaf_mask,
                                                    upper_region_height=upper_region_height)

        # Additional step: Filter out non-green regions in upper area
        # This removes white surfaces and color cards by keeping only green pixels
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Define green color range in HSV
        lower_green = np.array([35, 30, 30])  # Hue 35-85, with minimum saturation and value
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Apply green filter only to upper region
        upper_green_mask = np.ones_like(leaf_mask)
        upper_green_mask[:upper_region_height, :] = green_mask[:upper_region_height, :] > 0

        # Combine with existing mask (keep only pixels that are both in leaf_mask and green_mask in upper region)
        leaf_mask = leaf_mask & upper_green_mask

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
    parser = argparse.ArgumentParser(description="Segment leaf(s) by removing flood-filled regions and keeping the largest component touching both sides.")
    parser.add_argument("image", type=Path, help="Path to the input image or a directory of images.")
    parser.add_argument("--tolerance1", type=int, default=50, help="Color tolerance for the first flood fill (top-middle seed).")
    parser.add_argument("--tolerance2", type=int, default=50, help="Color tolerance for the second flood fill (bottom-middle seed).")
    parser.add_argument("--tolerance3", type=int, default=50, help="Color tolerance for the third flood fill (upper-right seed for color card removal).")
    parser.add_argument("--down-from-top", type=int, default=750, help="Pixels down from the top for the first seed (x is centered).")
    parser.add_argument("--up-from-bottom", type=int, default=20, help="Pixels up from the bottom for the second seed (x is centered).")
    parser.add_argument("--upper-right-x-offset", type=int, default=150, help="Pixels from right edge for upper-right seed (default: 150).")
    parser.add_argument("--upper-right-y", type=int, default=400, help="Pixels from top for upper-right seed (default: 400).")
    parser.add_argument("--remove-upper-right", action="store_true", default=True, help="Remove upper-right region (color card) via flood fill.")
    parser.add_argument("--no-remove-upper-right", action="store_false", dest="remove_upper_right", help="Disable upper-right region removal.")
    parser.add_argument("--remove-upper-rectangles", action="store_true", default=True, help="Remove rectangular components in upper region (color cards).")
    parser.add_argument("--no-remove-upper-rectangles", action="store_false", dest="remove_upper_rectangles", help="Disable upper rectangle removal.")
    parser.add_argument("--max-area-ratio", type=float, default=0.15, help="Max area ratio for removable upper components (default: 0.15).")
    parser.add_argument("--min-rectangularity", type=float, default=0.6, help="Min rectangularity for component removal (default: 0.6).")
    parser.add_argument("--upper-region-height", type=int, default=1000, help="Height of upper region to check for cards (default: 1000).")
    parser.add_argument("--card-region-height", type=int, default=1200, help="Height of card region to mask from top (default: 1200).")
    parser.add_argument("--card-region-width", type=int, default=900, help="Width of card region to mask from right edge (default: 900).")
    parser.add_argument("--output-prefix", type=str, default="leaf_segmentation", help="Prefix for output files when a single image is provided.")
    parser.add_argument("--output-dir", type=Path, default=Path("demo2_leaves"), help="Directory to write outputs when processing a folder.")
    parser.add_argument("--trim-left", type=int, default=300, help="Pixels to trim from left border (default: 300 for device 7).")
    parser.add_argument("--trim-right", type=int, default=100, help="Pixels to trim from right border (default: 100 for device 7).")
    args = parser.parse_args()

    input_path = args.image
    if input_path.is_dir():
        success = False
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        for img_path in sorted(input_path.iterdir()):
            if img_path.suffix.lower() not in extensions:
                continue
            out_file = out_dir / f"{img_path.stem}_leaf.png"
            binary_mask = process_single(img_path, args.tolerance1, args.tolerance2, args.tolerance3,
                                        args.down_from_top, args.up_from_bottom, args.trim_left, args.trim_right,
                                        args.remove_upper_right, args.upper_right_x_offset, args.upper_right_y,
                                        args.remove_upper_rectangles, args.max_area_ratio, args.min_rectangularity,
                                        args.upper_region_height, args.card_region_height, args.card_region_width)
            if binary_mask is not None:
                cv2.imwrite(str(out_file), binary_mask * 255)
                print(f"Wrote {out_file}")
                success = True
        if not success:
            raise SystemExit("No images were processed successfully.")
    else:
        out_file = Path(args.output_prefix).with_suffix(".leaf.png")
        binary_mask = process_single(input_path, args.tolerance1, args.tolerance2, args.tolerance3,
                                     args.down_from_top, args.up_from_bottom, args.trim_left, args.trim_right,
                                     args.remove_upper_right, args.upper_right_x_offset, args.upper_right_y,
                                     args.remove_upper_rectangles, args.max_area_ratio, args.min_rectangularity,
                                     args.upper_region_height, args.card_region_height, args.card_region_width)
        if binary_mask is not None:
            cv2.imwrite(str(out_file), binary_mask * 255)
            print(f"Wrote {out_file}")
        else:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
