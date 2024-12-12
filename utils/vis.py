import torch
import matplotlib.pyplot as plt

def plot_mri(image, label, slice_idx=100):
    """
    Visualize a single slice of the 3D brain MRI and its segmentation label.
    
    Args:
        image (torch.Tensor or np.ndarray): 3D MRI image tensor (C, H, W, D) or (H, W, D).
        label (torch.Tensor or np.ndarray): 3D segmentation label tensor (H, W, D).
        slice_idx (int): Index of the slice to visualize along the axial (z) axis.
    """
    # Convert to numpy arrays if tensors
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    if isinstance(label, torch.Tensor):
        label = label.numpy()

    # Squeeze the channel dimension if present
    if image.ndim == 4:
        image = image.squeeze(0)

    # Extract the slice along the z-axis
    image_slice = image[:, :, slice_idx].T
    label_slice = label[:, :, slice_idx].T

    # Plot the slices
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f'MRI Slice {slice_idx}')
    plt.imshow(image_slice, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'Segmentation Slice {slice_idx}')
    plt.imshow(label_slice, cmap='viridis', alpha=0.7)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
def plot_images(images, titles, cols=1, figsize=(10, 5), wspace=0.1, hspace=0.1):
    """
    Visualize a list of images with their corresponding titles.
    
    Args:
        images (list): List of images to visualize.
        titles (list): List of titles for each image.
        cols (int): Number of columns in the grid.
        figsize (tuple): Figure size (width, height).
        wspace (float): Width space between subplots.
        hspace (float): Height space between subplots.
    """
    rows = len(images) // cols + (len(images) % cols > 0)
    plt.figure(figsize=figsize)
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(title, fontsize=10)
        plt.axis('off')
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()