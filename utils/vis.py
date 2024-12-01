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