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