"""
Example: Using the combine_masks function
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Example 1: Load and visualize saved combined mask
print("="*60)
print("Example 1: Load and visualize saved combined mask")
print("="*60)

# Load the binary mask
mask_path = "data/test_input/sam3_mask_2752_LeafPhotoA_2025-09-09 16_37_44.997-05_00.npy"
combined_mask = np.load(mask_path)

print(f"Mask shape: {combined_mask.shape}")
print(f"Unique values: {np.unique(combined_mask)}")
print(f"Mask coverage: {(combined_mask > 0).sum() / combined_mask.size * 100:.2f}%")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original mask
axes[0].imshow(combined_mask, cmap='gray')
axes[0].set_title('Binary Mask (0 or 1)')
axes[0].axis('off')

# Inverted (for better visibility)
axes[1].imshow(1 - combined_mask, cmap='gray')
axes[1].set_title('Inverted Mask')
axes[1].axis('off')

# With colormap
axes[2].imshow(combined_mask, cmap='viridis')
axes[2].set_title('Mask with Color')
axes[2].axis('off')

plt.tight_layout()
output_viz = "data/test_input/mask_visualization.png"
plt.savefig(output_viz, dpi=150, bbox_inches='tight')
print(f"Saved visualization to: {output_viz}")
plt.close()

# Example 2: Use mask to extract leaf regions
print("\n" + "="*60)
print("Example 2: Extract leaf regions using combined mask")
print("="*60)

# Load original image
image_path = "data/test_input/2752_LeafPhotoA_2025-09-09 16_37_44.997-05_00.jpg"
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

print(f"Image shape: {image_np.shape}")

# Apply mask (white background)
masked_white = image_np.copy()
masked_white[combined_mask == 0] = [255, 255, 255]

# Apply mask (black background)
masked_black = image_np.copy()
masked_black[combined_mask == 0] = [0, 0, 0]

# Apply mask (transparent background - save as PNG)
image_rgba = Image.open(image_path).convert("RGBA")
image_rgba_np = np.array(image_rgba)
image_rgba_np[combined_mask == 0, 3] = 0  # Set alpha to 0 for non-leaf areas

# Save results
output_white = Path("data/test_input/extracted_leaf_white_bg.jpg")
output_black = Path("data/test_input/extracted_leaf_black_bg.jpg")
output_transparent = Path("data/test_input/extracted_leaf_transparent.png")

Image.fromarray(masked_white).save(output_white)
Image.fromarray(masked_black).save(output_black)
Image.fromarray(image_rgba_np).save(output_transparent)

print(f"Saved white background extraction to: {output_white}")
print(f"Saved black background extraction to: {output_black}")
print(f"Saved transparent background extraction to: {output_transparent}")

# Example 3: Compute mask statistics
print("\n" + "="*60)
print("Example 3: Mask statistics")
print("="*60)

# Calculate statistics
total_pixels = combined_mask.size
leaf_pixels = (combined_mask > 0).sum()
background_pixels = total_pixels - leaf_pixels

print(f"Total pixels: {total_pixels:,}")
print(f"Leaf pixels: {leaf_pixels:,}")
print(f"Background pixels: {background_pixels:,}")
print(f"Leaf coverage: {leaf_pixels / total_pixels * 100:.2f}%")

# Find bounding box of combined mask
rows = np.any(combined_mask, axis=1)
cols = np.any(combined_mask, axis=0)
rmin, rmax = np.where(rows)[0][[0, -1]]
cmin, cmax = np.where(cols)[0][[0, -1]]

print(f"\nBounding box of all leaves:")
print(f"  Top-left: ({cmin}, {rmin})")
print(f"  Bottom-right: ({cmax}, {rmax})")
print(f"  Width: {cmax - cmin + 1} pixels")
print(f"  Height: {rmax - rmin + 1} pixels")

# Example 4: Create cropped version
print("\n" + "="*60)
print("Example 4: Create cropped version with padding")
print("="*60)

# Add padding
padding = 50
rmin_pad = max(0, rmin - padding)
rmax_pad = min(image_np.shape[0], rmax + padding)
cmin_pad = max(0, cmin - padding)
cmax_pad = min(image_np.shape[1], cmax + padding)

# Crop image and mask
cropped_image = image_np[rmin_pad:rmax_pad, cmin_pad:cmax_pad]
cropped_mask = combined_mask[rmin_pad:rmax_pad, cmin_pad:cmax_pad]

# Apply mask to cropped image
cropped_masked = cropped_image.copy()
cropped_masked[cropped_mask == 0] = [255, 255, 255]

output_cropped = Path("data/test_input/extracted_leaf_cropped.jpg")
Image.fromarray(cropped_masked).save(output_cropped)
print(f"Saved cropped extraction to: {output_cropped}")
print(f"Cropped size: {cropped_image.shape[1]} x {cropped_image.shape[0]} pixels")

print("\n" + "="*60)
print("All examples complete!")
print("="*60)
