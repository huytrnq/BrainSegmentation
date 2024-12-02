"""
This module contains a custom PyTorch dataset class for the skin lesion dataset.
It is used to load the images and their corresponding labels (if available) from the disk.
"""

import os
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset

class BrainMRIDataset(Dataset):
    """ PyTorch dataset for loading 3D brain MRI volumes and their segmentation labels. """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the root directory containing patient subfolders.
            transform (callable, optional): Optional transform to be applied to samples.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Collect all patient subfolders and their raw/segmentation paths
        self.data = []
        for patient_folder in sorted(os.listdir(root_dir)):
            patient_path = os.path.join(root_dir, patient_folder)
            if os.path.isdir(patient_path):
                # Initialize variables for raw and segmentation paths
                raw_image = None
                label_image = None
                # Iterate over files in the patient folder
                for file in os.listdir(patient_path):
                    if file.startswith('.'):  # Skip hidden files like .DS_Store
                        continue
                    if 'seg' in file.lower():  # Identify segmentation files
                        label_image = os.path.join(patient_path, file)
                    else:  # Assume remaining files are raw images
                        raw_image = os.path.join(patient_path, file)
                # Ensure both raw and label paths are found
                if raw_image and label_image:
                    self.data.append((raw_image, label_image))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_path, label_path = self.data[idx]

        # Load .nii.gz files using nibabel
        image = nib.load(raw_path).get_fdata().astype(np.float32)
        label = nib.load(label_path).get_fdata().astype(np.int64)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        # Convert to PyTorch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        return image, label


class BrainMRISliceDataset(Dataset):
    """Dataset for 2D MRI slices with Albumentations transformations."""
    def __init__(self, base_dir, slice_axis=2, transform=None, num_classes=3):
        """
        Args:
            base_dir (str): Path to the base directory containing patient subfolders.
            slice_axis (int): Axis to extract slices (0=axial, 1=coronal, 2=sagittal).
            transform (callable, optional): Transformations to apply to images and labels.
            num_classes (int): Number of classes in the segmentation mask.
        """
        self.base_dir = base_dir
        self.slice_axis = slice_axis
        self.transform = transform
        self.num_classes = 3  # Background, CSF, GM, WM

        # Collect paths for images and labels
        self.image_label_pairs = []
        for subfolder in os.listdir(base_dir):
            subfolder_path = os.path.join(base_dir, subfolder)
            if os.path.isdir(subfolder_path):
                image_path, label_path = None, None
                for file_name in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file_name)
                    if file_name.startswith("."):
                        continue
                    if "seg" in file_name.lower():
                        label_path = file_path
                    else:
                        image_path = file_path
                if image_path and label_path:
                    self.image_label_pairs.append((image_path, label_path))

        # Compute the total number of slices across all volumes
        self.slice_info = []  # (volume_idx, slice_idx) for each slice
        for volume_idx, (image_path, label_path) in enumerate(self.image_label_pairs):
            image = nib.load(image_path).get_fdata()
            num_slices = image.shape[self.slice_axis]
            self.slice_info.extend([(volume_idx, slice_idx) for slice_idx in range(num_slices)])

    def __len__(self):
        return len(self.slice_info)

    def mask_to_onehot(self, mask):
        """
        Convert a segmentation mask to one-hot encoded format.

        Args:
            mask (torch.Tensor): Segmentation mask of shape (H, W) with class indices.

        Returns:
            torch.Tensor: One-hot encoded tensor of shape (num_classes, H, W).
        """
        # Ensure mask is of dtype int64
        mask = mask.to(torch.int64)

        # Initialize one-hot tensor with zeros
        onehot = torch.zeros((self.num_classes + 1, *mask.shape), dtype=torch.float32, device=mask.device)

        # Scatter 1 at the corresponding class indices
        onehot.scatter_(0, mask.unsqueeze(0), 1)
        # Exclude the background class
        onehot = onehot[1:, ...]

        return onehot

    def __getitem__(self, idx):
        # Retrieve volume and slice index
        volume_idx, slice_idx = self.slice_info[idx]
        image_path, label_path = self.image_label_pairs[volume_idx]

        # Load the 3D image and label
        image = nib.load(image_path).get_fdata().astype(np.float32)
        label = nib.load(label_path).get_fdata().astype(np.int64)

        # Extract the slice along the specified axis
        if self.slice_axis == 0:
            image_slice = image[slice_idx, :, :]
            label_slice = label[slice_idx, :, :]
        elif self.slice_axis == 1:
            image_slice = image[:, slice_idx, :]
            label_slice = label[:, slice_idx, :]
        else:
            image_slice = image[:, :, slice_idx]
            label_slice = label[:, :, slice_idx]

        # Apply Albumentations transformations if needed
        if self.transform:
            augmented = self.transform(image=image_slice, mask=label_slice)
            image_slice = augmented['image']
            label_slice = augmented['mask']
        image_slice = image_slice.repeat(3, 1, 1)  # Convert to 3-channel image
        label_slice = self.mask_to_onehot(label_slice).squeeze(-1)
        label_slice = label_slice.float()  # Convert to floating point type

        return image_slice, label_slice