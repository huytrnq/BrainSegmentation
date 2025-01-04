"""Utility functions for training and testing the model."""

import os
from tqdm import tqdm
import torch
import numpy as np
import nibabel as nib
from utils.metric import dice_coefficient, dice_score_3d


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


def train_3d(model, train_loader, criterion, optimizer, device, epoch, EPOCHS, NUM_CLASSES):
    """Train the model for one epoch - 3D version.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for the training dataset.
        criterion: Loss function.
        optimizer: Optimizer for updating model weights.
        device: Device to run the training on (CPU or GPU).
        epoch (int): Current epoch number.
        EPOCHS (int): Total number of EPOCHS.
        NUM_CLASSES (int): Number of classes in the segmentation mask.
    """
    model.train()
    train_loss = 0
    avg_dice = []
    csf_dice = []
    gm_dice = []
    wm_dice = []
    progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Training Epoch {epoch + 1}/{EPOCHS}")
    for batch in progress_bar:
        images, masks = batch["image"]["data"].to(device), batch["mask"]["data"].long().to(device)  # Adjust keys if necessary

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, masks)
        train_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Dice score
        pred = torch.argmax(outputs, dim=1) 
        masks = masks.squeeze(1)
        dice = dice_score_3d(pred, masks, NUM_CLASSES)
        avg_dice.append(np.mean(list(dice.values())))
        csf_dice.append(dice[1])
        gm_dice.append(dice[2])
        wm_dice.append(dice[3])
        
        #update the progress bar
        progress_bar.set_postfix({"Loss": loss.item() / len(batch), "Avg Dice": np.mean(list(dice.values())), "CSF Dice": dice[1], "GM Dice": dice[2], "WM Dice": dice[3]})

    print(f"Epoch {epoch + 1}, Loss: {train_loss/len(train_loader):.4f}")
    print(f"Epoch {epoch + 1}, Dice: {np.mean(avg_dice):.4f}", f"CSF Dice: {np.mean(csf_dice):.4f}", f"GM Dice: {np.mean(gm_dice):.4f}", f"WM Dice: {np.mean(wm_dice):.4f}")
    return train_loss/len(train_loader), np.mean(avg_dice), np.mean(csf_dice), np.mean(gm_dice), np.mean(wm_dice)

def validate_3d(model, val_loader, criterion, device, epoch, EPOCHS, NUM_CLASSES):
    """Validate the model for one epoch - 3D version.

    Args:
        model: PyTorch model to evaluate.
        val_loader: DataLoader for the validation dataset.
        criterion: Loss function.
        device: Device to run the evaluation on (CPU or GPU).
        epoch (int): Current epoch number.
        EPOCHS (int): Total number of epochs.
        NUM_CLASSES (int): Number of classes in the segmentation mask.
    """
    model.eval() # Set the model to evaluation mode

    # Validation loop
    val_loss = 0
    avg_dice = []
    csf_dice = []
    gm_dice = []
    wm_dice = []
    progress_bar = tqdm(val_loader, total=len(val_loader), desc=f"Validation Epoch {epoch + 1}/{EPOCHS}")
    with torch.no_grad():
        for batch in progress_bar:
            images, masks = batch["image"]["data"].to(device), batch["mask"]["data"].long().to(device)  # Adjust keys if necessary

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            # Dice score
            pred = torch.argmax(outputs, dim=1) 
            masks = masks.squeeze(1)
            dice = dice_score_3d(pred, masks, NUM_CLASSES)
            avg_dice.append(np.mean(list(dice.values())))
            csf_dice.append(dice[1])
            gm_dice.append(dice[2])
            wm_dice.append(dice[3])
            
            #update the progress bar
            progress_bar.set_postfix({"Loss": loss.item() / len(batch), "Avg Dice": np.mean(list(dice.values())), "CSF Dice": dice[1], "GM Dice": dice[2], "WM Dice": dice[3]})

        print(f"Epoch {epoch + 1}, Loss: {val_loss/len(val_loader):.4f}")
        print(f"Epoch {epoch + 1}, Dice: {np.mean(avg_dice):.4f}", f"CSF Dice: {np.mean(csf_dice):.4f}", f"GM Dice: {np.mean(gm_dice):.4f}", f"WM Dice: {np.mean(wm_dice):.4f}\n")
    return val_loss/len(val_loader), np.mean(avg_dice), np.mean(csf_dice), np.mean(gm_dice), np.mean(wm_dice)

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