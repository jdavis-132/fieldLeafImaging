import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os

class SAM2TinyPredictor:
    def __init__(self, model_path="models/sam2.1_hiera_tiny.pt", device="auto"):
        """
        Initialize SAM 2.1 Tiny model

        Args:
            model_path (str): Path to the model checkpoint
            device (str): Device to run the model on ('auto', 'cuda', 'cpu')
        """
        # Auto-detect best available device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🚀 CUDA GPU detected: {gpu_name}")
            else:
                self.device = "cpu"
                print("⚠️  CUDA not available, using CPU")
        else:
            self.device = device

        self.model_path = model_path

        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Build the model
        print(f"Loading SAM 2.1 Tiny model on {self.device.upper()}...")
        try:
            self.sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", model_path, device=self.device)
            # Explicitly move model to device
            self.sam2_model = self.sam2_model.to(self.device)
            self.predictor = SAM2ImagePredictor(self.sam2_model)
            print(f"✅ Model loaded successfully on {self.device.upper()}!")

            # Print memory usage if on GPU
            if self.device == "cuda":
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                print(f"📊 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

        except Exception as e:
            if "CUDA" in str(e) and self.device == "cuda":
                print(f"⚠️  CUDA error: {e}")
                print("🔄 Falling back to CPU...")
                self.device = "cpu"
                self.sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", model_path, device=self.device)
                self.sam2_model = self.sam2_model.to(self.device)
                self.predictor = SAM2ImagePredictor(self.sam2_model)
                print("✅ Model loaded successfully on CPU!")
            else:
                raise e

    def load_image(self, image_path):
        """
        Load and preprocess image

        Args:
            image_path (str): Path to the image

        Returns:
            numpy.ndarray: RGB image array
        """
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path

        return image

    def predict_with_points(self, image_path, point_coords, point_labels):
        """
        Predict segmentation using point prompts

        Args:
            image_path (str): Path to the image
            point_coords (list): List of [x, y] coordinates
            point_labels (list): List of labels (1 for positive, 0 for negative)

        Returns:
            dict: Dictionary containing masks, scores, and logits
        """
        # Load and set image
        image = self.load_image(image_path)
        self.predictor.set_image(image)

        # Convert to numpy arrays
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

        # Predict
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        return {
            'masks': masks,
            'scores': scores,
            'logits': logits,
            'image': image
        }

    def predict_with_box(self, image_path, box_coords):
        """
        Predict segmentation using box prompt

        Args:
            image_path (str): Path to the image
            box_coords (list): [x1, y1, x2, y2] bounding box coordinates

        Returns:
            dict: Dictionary containing masks, scores, and logits
        """
        # Load and set image
        image = self.load_image(image_path)
        self.predictor.set_image(image)

        # Convert to numpy array
        box = np.array(box_coords)

        # Predict
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=False,
        )

        return {
            'masks': masks,
            'scores': scores,
            'logits': logits,
            'image': image
        }

    def predict_everything(self, image_path):
        """
        Segment everything in the image (using automatic mask generation)

        Args:
            image_path (str): Path to the image

        Returns:
            dict: Dictionary containing masks and image
        """
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        # Create mask generator
        mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)

        # Load image
        image = self.load_image(image_path)

        # Generate masks
        masks = mask_generator.generate(image)

        return {
            'masks': masks,
            'image': image
        }

    def visualize_prediction(self, result, save_path=None):
        """
        Visualize the prediction results

        Args:
            result (dict): Result from prediction methods
            save_path (str, optional): Path to save the visualization
        """
        image = result['image']

        if 'masks' in result and len(result['masks']) > 0:
            # For regular predictions
            masks = result['masks']

            # Create figure
            fig, axes = plt.subplots(1, len(masks) + 1, figsize=(15, 5))
            if len(masks) == 0:
                axes = [axes]

            # Show original image
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Show each mask
            for i, mask in enumerate(masks):
                if len(masks) > 1:
                    ax = axes[i + 1]
                    score = result.get('scores', [0])[i] if 'scores' in result else 0
                    ax.imshow(image)
                    ax.imshow(mask, alpha=0.5, cmap='jet')
                    ax.set_title(f'Mask {i+1} (Score: {score:.3f})')
                    ax.axis('off')
                else:
                    axes[1].imshow(image)
                    axes[1].imshow(mask, alpha=0.5, cmap='jet')
                    axes[1].set_title('Segmentation Mask')
                    axes[1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_everything_prediction(self, result, save_path=None):
        """
        Visualize the 'predict everything' results

        Args:
            result (dict): Result from predict_everything method
            save_path (str, optional): Path to save the visualization
        """
        image = result['image']
        masks = result['masks']

        # Create visualization with all masks overlaid
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Original image
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # All masks
        ax2.imshow(image)

        # Create a combined mask visualization
        combined_mask = np.zeros((*image.shape[:2], 3))
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            color = np.random.random(3)
            combined_mask[mask] = color

        ax2.imshow(combined_mask, alpha=0.6)
        ax2.set_title(f'All Segments ({len(masks)} masks)')
        ax2.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

def demo():
    """
    Demo function showing how to use SAM 2.1 Tiny
    """
    # Initialize predictor
    predictor = SAM2TinyPredictor()

    # Check if we have sample images in the data directory
    data_dir = "data"
    if os.path.exists(data_dir):
        # Look for image files in the extracted directories
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(root, file)
                    print(f"Found sample image: {image_path}")

                    # Demo with point prompts
                    print("Demo: Point-based segmentation")
                    # Use center of image as a positive point
                    img = cv2.imread(image_path)
                    h, w = img.shape[:2]
                    center_point = [[w//2, h//2]]  # Center of image
                    point_labels = [1]  # Positive point

                    result = predictor.predict_with_points(image_path, center_point, point_labels)
                    predictor.visualize_prediction(result, "point_segmentation_demo.png")

                    # Demo with automatic segmentation (first image only)
                    print("Demo: Automatic segmentation")
                    result_auto = predictor.predict_everything(image_path)
                    predictor.visualize_everything_prediction(result_auto, "auto_segmentation_demo.png")

                    break
            if 'image_path' in locals():
                break
    else:
        print("No data directory found. Please add some sample images to test the model.")

if __name__ == "__main__":
    demo()