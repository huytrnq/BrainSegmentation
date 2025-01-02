import os
from PIL import Image
import numpy as np

def get_image_size_statistics(dataset_path):
    """
    Calculate the statistics of image dimensions (height and width) in a dataset.

    Args:
        dataset_path (str): Path to the dataset folder containing images in subfolders.

    Returns:
        None: Prints the statistics.
    """
    heights = []
    widths = []
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        heights.append(height)
                        widths.append(width)
                except Exception as e:
                    print(f"Could not process image {image_path}: {e}")
    
    if heights and widths:
        print("Height Statistics:")
        print(f"  Mean: {np.mean(heights):.2f}")
        print(f"  Median: {np.median(heights):.2f}")
        print(f"  Min: {np.min(heights)}")
        print(f"  Max: {np.max(heights)}")
        
        print("\nWidth Statistics:")
        print(f"  Mean: {np.mean(widths):.2f}")
        print(f"  Median: {np.median(widths):.2f}")
        print(f"  Min: {np.min(widths)}")
        print(f"  Max: {np.max(widths)}")
    else:
        print("No valid images found in the dataset.")

# Example usage:
dataset_path = "../datasets/Multiclass"  # Replace with the path to your dataset
get_image_size_statistics(dataset_path)