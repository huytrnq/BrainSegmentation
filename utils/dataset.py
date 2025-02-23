"""
This module contains a custom PyTorch dataset class for the skin lesion dataset.
It is used to load the images and their corresponding labels (if available) from the disk.
"""

import os
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset


class BrainMRIDataset(SubjectsDataset):
    """
    TorchIO-based dataset for 3D Brain MRI volumes and segmentation labels.
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the directory containing patient subfolders.
                            Each subfolder should contain an MRI volume and a segmentation label.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.subjects = self._get_subjects()
        super().__init__(self.subjects, transform=transform)

    def _get_subjects(self):
        """
        Collect all patient subfolders and create TorchIO subjects.

        Returns:
            list: A list of TorchIO Subject objects.
        """
        subjects = []
        for patient_folder in sorted(os.listdir(self.root_dir)):
            patient_path = os.path.join(self.root_dir, patient_folder)
            if os.path.isdir(patient_path):
                # Find image and label files
                image_path, label_path = None, None
                for file_name in os.listdir(patient_path):
                    if file_name.startswith('.'):
                        continue  # Ignore hidden files
                    file_path = os.path.join(patient_path, file_name)
                    if 'seg' in file_name.lower():
                        label_path = file_path
                    else:
                        image_path = file_path
                
                if image_path and label_path:
                    # Create a TorchIO Subject
                    subject = Subject(
                        image=ScalarImage(image_path),
                        mask=LabelMap(label_path)
                    )
                    subjects.append(subject)
                elif image_path:
                    # Create a TorchIO Subject with only the image
                    subject = Subject(
                        image=ScalarImage(image_path)
                    )
                    subjects.append(subject)
        return subjects

    def calculate_class_weights(self, num_classes):
        """
        Calculate class weights based on voxel distribution in segmentation labels.

        Args:
            num_classes (int): Number of segmentation classes.

        Returns:
            torch.Tensor: Class weights inversely proportional to class frequencies.
        """
        voxel_counts = torch.zeros(num_classes, dtype=torch.float32)

        # Iterate over all subjects to accumulate voxel counts for each class
        for subject in self.subjects:
            label_map = subject['mask'].data  # Load the segmentation label
            for class_id in range(num_classes):
                voxel_counts[class_id] += (label_map == class_id).sum().item()

        # Compute class weights (inverse frequency)
        total_voxels = voxel_counts.sum().item()
        class_weights = total_voxels / (voxel_counts + 1e-6)  # Avoid division by zero

        return class_weights / class_weights.sum()  # Normalize weights
    
    def calculate_class_weights_log(self, num_classes):
        """
        Calculate class weights using a logarithmic adjustment.

        Args:
            num_classes (int): Number of segmentation classes.

        Returns:
            torch.Tensor: Log-adjusted class weights.
        """
        voxel_counts = torch.zeros(num_classes, dtype=torch.float32)

        # Count voxels for each class
        for subject in self.subjects:
            label_map = subject['mask'].data  # Load the segmentation label
            for class_id in range(num_classes):
                voxel_counts[class_id] += (label_map == class_id).sum().item()

        # Logarithmic adjustment
        class_weights = 1.0 / (torch.log(voxel_counts + 1) + 1)
        return class_weights / class_weights.sum()  # Normalize weights


class WeightedLabelSampler(Sampler):
    """
    Custom sampler to sample slices based on weighted label probabilities.
    """
    def __init__(self, dataset, label_probabilities, num_samples=None):
        """
        Args:
            dataset (BrainMRISliceDataset): The dataset instance.
            label_probabilities (dict): A dictionary containing probabilities for each label.
                Example: {0: 0, 1: 2, 2: 1, 3: 1}
            num_samples (int, optional): Total number of samples to draw. Defaults to the dataset length.
        """
        self.dataset = dataset
        self.label_probabilities = label_probabilities
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.slice_weights = self._compute_weights()

    def _compute_weights(self):
        """
        Compute weights for each slice based on label probabilities.
        Returns:
            np.ndarray: Weights for all slices in the dataset.
        """
        slice_weights = []
        for volume_idx, slice_idx in self.dataset.slice_info:
            _, label_path = self.dataset.image_label_pairs[volume_idx]
            label_volume = self.dataset.load_volume(label_path)
            label_slice = self.dataset.extract_slice(label_volume, slice_idx)
            
            # Count the occurrences of each class in the slice
            class_counts = {k: np.sum(label_slice == k) for k in self.label_probabilities.keys()}
            
            # Compute the slice weight as the sum of the weighted probabilities
            slice_weight = sum(class_counts[k] * self.label_probabilities[k] for k in class_counts)
            slice_weights.append(slice_weight)
        
        # Normalize weights
        slice_weights = np.array(slice_weights)
        return slice_weights / np.sum(slice_weights)

    def __iter__(self):
        """
        Create an iterator that samples indices based on computed weights.
        Returns:
            Iterator[int]: Indices for sampling.
        """
        return iter(np.random.choice(len(self.dataset), size=self.num_samples, p=self.slice_weights))

    def __len__(self):
        """
        Return the total number of samples to draw.
        Returns:
            int: Total number of samples.
        """
        return self.num_samples


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
    
    def _get_class_weights(self, num_classes=4):
        class_counts = np.zeros(num_classes)
        for _, label_path in self.image_label_pairs:
            label = nib.load(label_path).get_fdata().astype(np.int64)
            for class_idx in range(num_classes):
                class_counts[class_idx] += np.sum(label == class_idx)
        total_voxels = np.sum(class_counts)
        class_weights = 1 + np.log(total_voxels / class_counts)  # Logarithmic normalization
        return torch.tensor(class_weights, dtype=torch.float32)

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

