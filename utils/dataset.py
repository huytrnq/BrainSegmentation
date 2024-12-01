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
    """PyTorch dataset for loading 2D slices from subfolders of 3D MRI volumes."""
    def __init__(self, base_dir, slice_axis=2, transform=None):
        """
        Args:
            base_dir (str): Base directory containing patient subfolders.
            slice_axis (int): Axis to extract slices (0=axial, 1=coronal, 2=sagittal).
            transform (callable, optional): Optional transform to apply to slices.
        """
        self.base_dir = base_dir
        self.slice_axis = slice_axis
        self.transform = transform

        # Collect paths for images and labels
        self.image_label_pairs = []
        for subfolder in os.listdir(base_dir):
            subfolder_path = os.path.join(base_dir, subfolder)
            if os.path.isdir(subfolder_path):
                # Identify image and label files based on 'seg' in filename
                image_path = None
                label_path = None
                for file_name in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file_name)
                    if file_name.startswith("."):  # Skip hidden files like .DS_Store
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
            image = nib.load(image_path).get_fdata()  # Load to get shape
            num_slices = image.shape[self.slice_axis]
            self.slice_info.extend([(volume_idx, slice_idx) for slice_idx in range(num_slices)])

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        # Retrieve volume and slice index
        volume_idx, slice_idx = self.slice_info[idx]
        image_path, label_path = self.image_label_pairs[volume_idx]

        # Load the corresponding 3D image and label
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

        # Apply transformations if needed
        if self.transform:
            image_slice = self.transform(image_slice)

        return image_slice, label_slice
