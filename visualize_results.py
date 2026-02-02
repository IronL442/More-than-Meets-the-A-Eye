#!/usr/bin/env python
"""
Visualization script for pseudo labeled saliency maps.
View generated annotations and compare with original images.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def visualize_pseudo_labels(image_dir, label_dir, output_dir="./visualizations", num_samples=10):
    """
    Generate visualization comparisons of images and their pseudo labels.
    
    Args:
        image_dir: Path to original images
        label_dir: Path to generated pseudo labels
        output_dir: Where to save visualizations
        num_samples: Number of samples to visualize
    """
    
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Visualizing pseudo labels...")
    print(f"Images: {image_dir}")
    print(f"Labels: {label_dir}")
    print(f"Output: {output_dir}\n")
    
    # Get image files
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))[:num_samples]
    
    if not image_files:
        print("No images found!")
        return
    
    for idx, img_path in enumerate(image_files):
        try:
            # Load image
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            
            # Load corresponding label
            label_name = img_path.stem + ".npy"
            label_path = label_dir / label_name
            
            if not label_path.exists():
                print(f"Label not found for {img_path.name}")
                continue
            
            saliency_map = np.load(label_path)
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(img_array)
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            
            # Saliency map
            im = axes[1].imshow(saliency_map, cmap="hot")
            axes[1].set_title("Generated Saliency Map")
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1])
            
            # Overlay
            saliency_normalized = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
            saliency_resized = cv2.resize(saliency_normalized, (img_array.shape[1], img_array.shape[0]))
            overlay = (img_array * 0.7 + saliency_resized[:, :, np.newaxis] * 255 * 0.3).astype(np.uint8)
            axes[2].imshow(overlay)
            axes[2].set_title("Image + Saliency Overlay")
            axes[2].axis("off")
            
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / f"visualization_{idx:03d}_{img_path.stem}.png"
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close()
            
            print(f"[{idx+1}/{len(image_files)}] Saved: {output_path.name}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    # Configure paths
    IMAGE_DIR = "./data/BenchmarkIMAGES/SM"
    LABEL_DIR = "./pseudo_labels_filtered"
    OUTPUT_DIR = "./visualizations"
    NUM_SAMPLES = 10
    
    visualize_pseudo_labels(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR, NUM_SAMPLES)
