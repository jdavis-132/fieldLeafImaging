"""
Test SAM3 model on leaf images
Dependencies: Download weights according to instructions at https://github.com/facebookresearch/sam3/tree/main
"""
import torch
from transformers import Sam3Processor, Sam3Model
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def combine_masks(masks, save_path=None):
    """
    Combine multiple binary masks into a single unified mask

    Args:
        masks: Tensor of masks [N, H, W] where N is number of masks
        save_path: Optional path to save the combined mask (as PNG or NPY)

    Returns:
        numpy array [H, W] with combined binary mask (values 0 or 1)
    """
    if len(masks) == 0:
        return None

    # Convert to numpy and combine using logical OR
    masks_np = masks.cpu().numpy()
    combined = np.zeros(masks_np.shape[1:], dtype=bool)

    for mask in masks_np:
        combined = np.logical_or(combined, mask > 0)

    combined = combined.astype(np.uint8)

    # Save if path is provided
    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix in ['.png', '.jpg']:
            # Save as image (0 -> black, 1 -> white)
            mask_img = Image.fromarray(combined * 255)
            mask_img.save(save_path)
            print(f"Saved combined mask image to: {save_path}")
        elif save_path.suffix in ['.npy', '.npz']:
            # Save as numpy array
            if save_path.suffix == '.npz':
                np.savez_compressed(save_path, mask=combined)
            else:
                np.save(save_path, combined)
            print(f"Saved combined mask array to: {save_path}")
        else:
            # Default to numpy
            np.save(save_path.with_suffix('.npy'), combined)
            print(f"Saved combined mask array to: {save_path.with_suffix('.npy')}")

    return combined


def overlay_masks(image, masks, alpha=0.5, color=(0, 255, 0), combine=False):
    """
    Overlay segmentation masks on the image using a single color

    Args:
        image: PIL Image
        masks: Tensor of masks
        alpha: Transparency (0-1)
        color: RGB tuple for mask color (default: green)
        combine: If True, combine all masks into one before overlaying

    Returns:
        PIL Image with mask overlay
    """
    image = image.convert("RGBA")

    if combine:
        # Combine all masks into a single binary mask
        combined_mask = combine_masks(masks)
        if combined_mask is None:
            return image.convert("RGB")

        masks_to_overlay = [combined_mask * 255]
    else:
        masks_to_overlay = 255 * masks.cpu().numpy().astype(np.uint8)

    # Apply mask overlay
    for mask in masks_to_overlay:
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha_channel = mask.point(lambda v: int(v * alpha))
        overlay.putalpha(alpha_channel)
        image = Image.alpha_composite(image, overlay)

    return image

def main():
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Load model and processor
    print("\nLoading SAM3 model...")
    model_path = "/home/schnable/Documents/fieldLeafImaging/src/sam3"
    model = Sam3Model.from_pretrained(model_path).to(device)
    processor = Sam3Processor.from_pretrained(model_path)
    print("Model loaded successfully!")

    # Load a test image
    test_image_dir = Path("/home/schnable/Documents/fieldLeafImaging/figures/supplemental/")
    image_files = list(test_image_dir.glob("*.jpg"))

    if not image_files:
        print("No test images found!")
        return

    test_image_path = image_files[0]
    print(f"\nProcessing image: {test_image_path.name}")
    image = Image.open(test_image_path).convert("RGB")
    print(f"Image size: {image.size}")

    # Test 1: Segment with text prompt "leaf"
    print("\n" + "="*60)
    print("Test 1: Text prompt segmentation - 'leaf'")
    print("="*60)

    text_prompt = "leaf"
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    print(f"Found {len(results['masks'])} objects with prompt '{text_prompt}'")
    if len(results['masks']) > 0:
        print(f"Scores: {results['scores'].tolist()}")
        print(f"Boxes shape: {results['boxes'].shape}")
        print(f"Masks shape: {results['masks'].shape}")

        # Save visualization with separate masks
        output_path = test_image_dir / f"sam3_output_text_{test_image_path.stem}.png"
        result_image = overlay_masks(image.copy(), results["masks"])
        result_image.save(output_path)
        print(f"Saved result to: {output_path}")

        # Save visualization with combined mask
        output_combined_path = test_image_dir / f"sam3_combined_{test_image_path.stem}.png"
        result_combined_image = overlay_masks(image.copy(), results["masks"], combine=True)
        result_combined_image.save(output_combined_path)
        print(f"Saved combined mask overlay to: {output_combined_path}")

        # Save the combined binary mask as image and numpy
        mask_img_path = test_image_dir / f"sam3_mask_{test_image_path.stem}.png"
        mask_npy_path = test_image_dir / f"sam3_mask_{test_image_path.stem}.npy"
        combined = combine_masks(results["masks"], save_path=mask_img_path)
        np.save(mask_npy_path, combined)
        print(f"Saved binary mask array to: {mask_npy_path}")

    # # Test 2: Different text prompts
    # print("\n" + "="*60)
    # print("Test 2: Segmentation with different prompts")
    # print("="*60)

    # # Try different prompts
    # prompts = ["plant", "green leaf", "maize leaf"]
    # 
    # for prompt in prompts:
    #     print(f"\nTrying prompt: '{prompt}'")
    #     inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    # 
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    # 
    #     results = processor.post_process_instance_segmentation(
    #         outputs,
    #         threshold=0.4,  # Lower threshold to catch more objects
    #         mask_threshold=0.4,
    #         target_sizes=inputs.get("original_sizes").tolist()
    #     )[0]
    # 
    #     print(f"  Found {len(results['masks'])} objects")
    #     if len(results['masks']) > 0:
    #         print(f"  Top 3 scores: {results['scores'][:3].tolist()}")
    # 
    # # Visualize semantic segmentation from the first test
    # print("\n" + "="*60)
    # print("Test 3: Semantic segmentation visualization")
    # print("="*60)

    # # Re-run with "leaf" prompt to get semantic segmentation
    # inputs = processor(images=image, text="leaf", return_tensors="pt").to(device)
    # with torch.no_grad():
    #     outputs = model(**inputs)
    # 
    # # Get semantic segmentation
    # if hasattr(outputs, 'semantic_seg') and outputs.semantic_seg is not None:
    #     semantic_seg = outputs.semantic_seg
    #     print(f"Semantic segmentation shape: {semantic_seg.shape}")
    # 
    #     # Save semantic segmentation visualization
    #     semantic_output_path = test_image_dir / f"sam3_semantic_{test_image_path.stem}.png"
    # 
    #     # Normalize and convert to image
    #     semantic_np = semantic_seg[0, 0].cpu().numpy()
    #     semantic_normalized = (semantic_np - semantic_np.min()) / (semantic_np.max() - semantic_np.min() + 1e-8)
    # 
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(image)
    #     plt.title("Original Image")
    #     plt.axis('off')
    # 
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(semantic_normalized, cmap='viridis')
    #     plt.title("Semantic Segmentation")
    #     plt.axis('off')
    # 
    #     plt.tight_layout()
    #     plt.savefig(semantic_output_path, dpi=150, bbox_inches='tight')
    #     plt.close()
    #     print(f"Saved semantic segmentation to: {semantic_output_path}")
    # else:
    #     print("Semantic segmentation not available in outputs")
    # 
    # print("\n" + "="*60)
    # print("Testing complete!")
    # print("="*60)

if __name__ == "__main__":
    main()
