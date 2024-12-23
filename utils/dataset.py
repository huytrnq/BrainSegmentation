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
    """Optimized Dataset for 2D MRI slices with Albumentations transformations."""
    def __init__(self, base_dir, slice_axis=2, transform=None, cache=False, ignore_background=False):
        """
        Args:
            base_dir (str): Path to the base directory containing patient subfolders.
            slice_axis (int): Axis to extract slices (0=axial, 1=coronal, 2=sagittal).
            transform (callable, optional): Transformations to apply to images and labels.
            num_classes (int): Number of classes in the segmentation mask.
            cache (bool): If True, caches volumes in memory after first load.
            ignore_background (bool): If True, ignore slices with only background class.
        """
        self.base_dir = base_dir
        self.slice_axis = slice_axis
        self.transform = transform
        self.cache = cache
        self.metadata = {} # Store metadata for each volume

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

        # Compute slice information (metadata only)
        self.slice_info = []  # (volume_idx, slice_idx)
        for volume_idx, (image_path, label_path) in enumerate(self.image_label_pairs):
            image = nib.load(image_path)
            label = nib.load(label_path)
            
            ## Extract metadata
            header = image.header
            affine = image.affine
            image_data = image.get_fdata()
            label_data = label.get_fdata().astype(np.int64)
            
            self.metadata[volume_idx] = {
                "shape": image_data.shape,
                "affine": affine,
                "header": header,
                "image_path": image_path,
            }
            
            # Determine number of slices along the specified axis
            num_slices = image_data.shape[self.slice_axis]
            for slice_idx in range(num_slices):
                slice_label = self.extract_slice(label_data, slice_idx)
                if ignore_background and self._is_background_only(slice_label):
                    continue
                self.slice_info.append((volume_idx, slice_idx))

        # Initialize cache if enabled
        self.volume_cache = {} if cache else None
        

    def _is_background_only(self, label_slice):
        """
        Check if a label slice contains only the background class.
        Args:
            label_slice (np.ndarray): A 2D slice of the label volume.
        Returns:
            bool: True if the slice is background-only, False otherwise.
        """
        return np.all(label_slice == 0)  # Assume background is class 0

    def __len__(self):
        return len(self.slice_info)

    def load_volume(self, path):
        """
        Load and return a volume. Cache if enabled.
        """
        if self.cache and path in self.volume_cache:
            return self.volume_cache[path]
        volume = nib.load(path).get_fdata()
        if self.cache:
            self.volume_cache[path] = volume
        return volume

    def extract_slice(self, volume, slice_idx):
        """
        Extract a single 2D slice from the 3D volume along the specified axis.
        """
        if self.slice_axis == 0:
            return volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            return volume[:, slice_idx, :]
        else:
            return volume[:, :, slice_idx]

    def __getitem__(self, idx):
        # Retrieve volume and slice index
        volume_idx, slice_idx = self.slice_info[idx]
        image_path, label_path = self.image_label_pairs[volume_idx]

        # Load volumes
        image = self.load_volume(image_path)
        label = self.load_volume(label_path)

        # Extract slices
        image_slice = self.extract_slice(image, slice_idx).astype(np.float32)
        label_slice = self.extract_slice(label, slice_idx).astype(np.int64)

        # Apply transformations if provided
        if self.transform:
            augmented = self.transform(image=image_slice, mask=label_slice)
            image_slice = augmented['image']
            label_slice = augmented['mask']

        # Convert to one-hot encoding for the label
        label_slice = label_slice.permute(2, 0, 1).float()

        # Add a channel dimension to the image
        if isinstance(image_slice, np.ndarray):  # If it's still a NumPy array
            image_slice = torch.from_numpy(image_slice).float()
        else:  # If it's already a tensor
            image_slice = image_slice.clone().detach().to(torch.float32)

        return image_slice, label_slice, volume_idx, slice_idx
