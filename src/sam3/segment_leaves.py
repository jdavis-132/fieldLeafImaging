"""
Batch leaf segmentation using SAM3 model
Dependencies: Download weights according to instructions at https://github.com/facebookresearch/sam3/tree/main
"""
import os
import torch
from transformers import Sam3Processor, Sam3Model
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


class LeafSegmenter:
    """SAM3-based leaf segmentation"""

    def __init__(self, model_path="src/sam3", device=None):
        """
        Initialize the segmenter

        Args:
            model_path: Path to SAM3 model directory
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing LeafSegmenter on {self.device}")

        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        print("Loading model...")
        self.model = Sam3Model.from_pretrained(model_path).to(self.device)
        self.processor = Sam3Processor.from_pretrained(model_path)
        print("Model loaded successfully!")

    def segment_image(self, image_path, text_prompt="leaf", threshold=0.5, mask_threshold=0.5):
        """
        Segment a single image

        Args:
            image_path: Path to input image
            text_prompt: Text description of what to segment
            threshold: Confidence threshold for object detection
            mask_threshold: Threshold for mask binarization

        Returns:
            dict with 'masks', 'boxes', 'scores'
        """
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        return {
            'image': image,
            'masks': results['masks'],
            'boxes': results['boxes'],
            'scores': results['scores']
        }
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

    def save_visualization(self, result, output_path, alpha=0.5):
        """
        Save segmentation visualization

        Args:
            result: Output from segment_image()
            output_path: Where to save the visualization
            alpha: Transparency of mask overlay (0-1)
        """
        image = result['image'].convert("RGBA")
        masks = 255 * result['masks'].cpu().numpy().astype(np.uint8)

        n_masks = masks.shape[0]
        if n_masks == 0:
            print(f"No masks to visualize, saving original image")
            image.convert("RGB").save(output_path)
            return

        cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
        colors = [
            tuple(int(c * 255) for c in cmap(i)[:3])
            for i in range(n_masks)
        ]

        for mask, color in zip(masks, colors):
            mask_img = Image.fromarray(mask)
            overlay = Image.new("RGBA", image.size, color + (0,))
            alpha_channel = mask_img.point(lambda v: int(v * alpha))
            overlay.putalpha(alpha_channel)
            image = Image.alpha_composite(image, overlay)

        # Convert back to RGB for saving as JPEG
        image = image.convert("RGB")
        image.save(output_path)

    def process_directory(self, input_dir, output_dir, text_prompt="leaf",
                         threshold=0.5, pattern="*.jpg"):
        """
        Process all images in a directory

        Args:
            input_dir: Directory containing input images
            output_dir: Directory for output images
            text_prompt: Text description of what to segment
            threshold: Confidence threshold
            pattern: File pattern to match (e.g., "*.jpg", "*.png")
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        image_files = list(input_path.glob(pattern))
        print(f"\nFound {len(image_files)} images matching pattern '{pattern}'")

        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_file.name}")

            try:
                result = self.segment_image(
                    image_file,
                    text_prompt=text_prompt,
                    threshold=threshold
                )

                n_objects = len(result['masks'])
                print(f"  Found {n_objects} objects")

                if n_objects > 0:
                    print(f"  Scores: {result['scores'].tolist()}")

                    # Save visualization
                    output_file = output_path / f"segmented_{image_file.name}"
                    self.save_visualization(result, output_file)
                    print(f"  Saved to: {output_file}")

                    # Save masks as numpy arrays
                    mask_file = output_path / f"masks_{image_file.stem}.npz"
                    np.savez_compressed(
                        mask_file,
                        masks=result['masks'].cpu().numpy(),
                        boxes=result['boxes'].cpu().numpy(),
                        scores=result['scores'].cpu().numpy()
                    )
                    print(f"  Saved masks to: {mask_file}")
                else:
                    print(f"  No objects detected")

            except Exception as e:
                print(f"  Error processing {image_file.name}: {e}")
                continue

        print("\n" + "="*60)
        print("Batch processing complete!")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Segment leaves using SAM3")
    parser.add_argument("--input", "-i", required=True, help="Input directory or image file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--prompt", "-p", default="leaf", help="Text prompt (default: 'leaf')")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Detection threshold (default: 0.5)")
    parser.add_argument("--model", "-m", default="src/sam3",
                       help="Path to SAM3 model (default: src/sam3)")
    parser.add_argument("--pattern", default="*.jpg",
                       help="File pattern for batch processing (default: *.jpg)")

    args = parser.parse_args()

    # Initialize segmenter
    segmenter = LeafSegmenter(model_path=args.model)

    # Check if input is a file or directory
    input_path = Path(args.input)

    if input_path.is_file():
        # Process single file
        print(f"\nProcessing single image: {input_path}")
        result = segmenter.segment_image(
            input_path,
            text_prompt=args.prompt,
            threshold=args.threshold
        )

        print(f"Found {len(result['masks'])} objects")
        if len(result['masks']) > 0:
            print(f"Scores: {result['scores'].tolist()}")

        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True, parents=True)
        output_file = output_path / f"segmented_{input_path.name}"

        segmenter.save_visualization(result, output_file)
        print(f"Saved to: {output_file}")

    elif input_path.is_dir():
        # Process directory
        segmenter.process_directory(
            input_path,
            args.output,
            text_prompt=args.prompt,
            threshold=args.threshold,
            pattern=args.pattern
        )
    else:
        print(f"Error: {input_path} does not exist")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
