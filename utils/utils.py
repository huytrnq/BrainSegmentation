"""Utility functions for training and testing the model."""

import os
import torch
import numpy as np
import nibabel as nib
from utils.metric import accuracy, dice_coefficient


def train(model, train_loader, criterion, optimizer, device, monitor):
    """
    Train the model for one epoch and track metrics.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for the training dataset.
        criterion: Loss function.
        optimizer: Optimizer for updating model weights.
        device: Device to run the training on (CPU or GPU).
        monitor: MetricsMonitor object for tracking metrics.

    Returns:
        dict: Dictionary with average metrics for the epoch.
    """
    model.train()
    monitor.reset()

    for batch_idx, (images, labels, volume_idx, slice_idx) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Squeeze the channel dimension of labels
        labels = labels.squeeze(1).long()  # Shape becomes [batch_size, height, width]

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate batch accuracy
        dice_score = dice_coefficient(outputs, labels)

        # Update monitor with loss and accuracy
        monitor.update("loss", loss.item(), count=len(images))
        # monitor.update("accuracy", batch_accuracy, count=len(images))
        monitor.update("dice_score", np.mean(dice_score), count=len(images))

        # Print iteration metrics
        monitor.print_iteration(batch_idx + 1, len(train_loader), phase="Train")

    # Print final metrics for the epoch
    monitor.print_final(phase="Train")
    return {metric: monitor.compute_average(metric) for metric in monitor.metrics}


def validate(model, valid_loader, criterion, device, monitor):
    """
    Validate the model for one epoch and track metrics.

    Args:
        model: PyTorch model to evaluate.
        valid_loader: DataLoader for the validation dataset.
        criterion: Loss function.
        device: Device to run the evaluation on (CPU or GPU).
        monitor: MetricsMonitor object for tracking metrics.

    Returns:
        dict: Dictionary with average metrics for the validation epoch.
    """
    model.eval()
    monitor.reset()

    with torch.no_grad():
        for batch_idx, (images, labels, volume_idx, slice_idx) in enumerate(valid_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Squeeze the channel dimension of labels
            labels = labels.squeeze(1).long()  # Shape becomes [batch_size, height, width]

            # Forward pass
            outputs = model(images)
            # outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, labels)

            # Calculate batch accuracy
            dice_score = dice_coefficient(outputs, labels)

            # Update monitor with loss and accuracy
            monitor.update("loss", loss.item(), count=len(images))
            # monitor.update("accuracy", batch_accuracy, count=len(images))
            monitor.update("dice_score", np.mean(dice_score), count=len(images))

            # Print iteration metrics
            monitor.print_iteration(batch_idx + 1, len(valid_loader), phase="Validation")

    # Print final metrics for the validation epoch
    monitor.print_final(phase="Validation")
    return {metric: monitor.compute_average(metric) for metric in monitor.metrics}



def get_data_paths(data_dir):
    """
    Load image and label paths from a directory.

    Args:
        data_dir (str): Path to the data directory.

    Returns:
        list, list: List of image paths and list of label paths.
    """
    image_paths, label_paths = [], []
    for patient_folder in sorted(os.listdir(data_dir)):
        patient_path = os.path.join(data_dir, patient_folder)
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
            if image_path is not None and label_path is not None:
                image_paths.append(image_path)
                label_paths.append(label_path)
    return image_paths, label_paths


def merge_slices_to_nifti(segmented_slices, reference_nifti_path, output_nifti_path):
    """
    Merge 2D segmented slices into a 3D NIfTI file.
    
    Args:
        segmented_slices (list or ndarray): List or array of 2D segmented slices (shape: [depth, height, width]).
        reference_nifti_path (str): Path to the reference NIfTI file to get affine and header information.
        output_nifti_path (str): Path to save the merged 3D NIfTI file.
    """
    # Convert the list of 2D slices into a 3D NumPy array if needed
    if isinstance(segmented_slices, list):
        segmented_3d = np.stack(segmented_slices, axis=0)  # Shape: [depth, height, width]
    else:
        segmented_3d = segmented_slices  # Already in 3D format
    
    # Ensure the array has the correct shape: [Z, Y, X]
    if segmented_3d.ndim != 3:
        raise ValueError(f"Expected 3D array, but got shape {segmented_3d.shape}")
    
    # Load reference NIfTI file to get affine and header information
    reference_nifti = nib.load(reference_nifti_path)
    affine = reference_nifti.affine
    header = reference_nifti.header
    
    # Create a new NIfTI image
    segmented_nifti = nib.Nifti1Image(segmented_3d.astype(np.uint8), affine, header)
    
    # Save the new NIfTI file
    nib.save(segmented_nifti, output_nifti_path)
    print(f"Saved 3D segmented NIfTI file to: {output_nifti_path}")

# Example usage
# Assume segmented_slices is a list of 2D arrays [slice1, slice2, ...]
# segmented_slices = [slice1, slice2, slice3, ...]
# reference_nifti_path = "path_to_reference_file.nii"
# output_nifti_path = "segmented_output.nii"
# merge_slices_to_nifti(segmented_slices, reference_nifti_path, output_nifti_path)