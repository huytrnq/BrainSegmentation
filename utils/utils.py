"""Utility functions for training and testing the model."""

import os
import json
from tqdm import tqdm
import numpy as np
import nibabel as nib
import torch
import torchio as tio
from utils.metric import dice_coefficient, dice_score_3d, hausdorff_distance, average_volumetric_difference


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


def validate_3d_patch(model, val_dataset, criterion, epoch, EPOCHS, DEVICE, NUM_CLASSES, PATCH_SIZE, BATCH_SIZE, NUM_WORKERS):
    losses = []
    predicts = []  # To store segmented images in their original size
    masks = []
    avg_dice = []
    csf_dice = []
    gm_dice = []
    wm_dice = []
    for subject in tqdm(val_dataset):
        # Get the mask as numpy
        mask = subject['mask'][tio.DATA]
        
        # Initialize GridSampler and GridAggregator
        grid_sampler = tio.inference.GridSampler(
            subject=subject,
            patch_size=PATCH_SIZE,
            patch_overlap=0,  # Adjust overlap as needed
        )
        aggregator = tio.inference.GridAggregator(grid_sampler)

        # Create DataLoader for patch-based inference
        val_loader = torch.utils.data.DataLoader(
            grid_sampler,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

        # Perform inference on patches
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['image'][tio.DATA].to(DEVICE)
                patch_mask = batch['mask'][tio.DATA].long().to(DEVICE)
                locations = batch[tio.LOCATION]  # Patch locations
                predictions = model(inputs)  # Replace with your model inference
                loss = criterion(predictions, patch_mask)
                losses.append(loss.item())
                aggregator.add_batch(predictions, locations)

                # Aggregate predictions into the original image size
                aggregated_prediction = aggregator.get_output_tensor().cpu()

        # Store the segmented image and mask
        predicts.append(aggregated_prediction)
        masks.append(mask)
    
    # Convert to tensors
    predicts = torch.stack(predicts, dim=0)
    masks = torch.stack(masks, dim=0)
    predicts = torch.argmax(predicts, dim=1)
    masks = masks.squeeze(1)
    # Calculate Dice scores
    dice = dice_score_3d(predicts, masks, NUM_CLASSES)
    avg_dice.append(np.mean(list(dice.values())))
    csf_dice.append(dice[1])
    gm_dice.append(dice[2])
    wm_dice.append(dice[3])
    
    # Print results
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {np.mean(losses):.4f}, Avg Dice: {np.mean(avg_dice):.4f}, CSF Dice: {np.mean(csf_dice):.4f}, GM Dice: {np.mean(gm_dice):.4f}, WM Dice: {np.mean(wm_dice):.4f}')
    
    return np.mean(losses), np.mean(avg_dice), np.mean(csf_dice), np.mean(gm_dice), np.mean(wm_dice)


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

def get_model_hyperparameters(model):
    """
    Extracts hyperparameters and top-level attributes from a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        dict: A dictionary containing model hyperparameters and parameter shapes.
    """
    config = {}

    # Retrieve attributes if available (e.g., from UNet or custom models)
    if hasattr(model, "__dict__"):
        config.update({key: value for key, value in vars(model).items() if not key.startswith("_")})

    # Include parameter shapes for learnable parameters
    for name, param in model.named_parameters():
        config[f"param_{name}"] = list(param.shape)

    return config


def save_model_config_to_file(model, file_path="model_config.json"):
    """
    Save the model configuration to a file in JSON format.

    Args:
        model (torch.nn.Module): The PyTorch model.
        file_path (str): Path to save the configuration file.
    """
    # Extract model hyperparameters
    model_config = get_model_hyperparameters(model)
    
    # Save as JSON
    with open(file_path, "w") as f:
        json.dump(model_config, f, indent=4)
        
def export_to_nii(array, file_path, spacing, affine=None):
    """
    Export a NumPy array to a NIfTI file with specified spacing and affine.

    Args:
        array (numpy.ndarray): The array to export.
        file_path (str): Path to save the NIfTI file.
        spacing (tuple): Spacing of the image in each dimension.
        affine (numpy.ndarray, optional): Affine transformation matrix.
    """
    # Cast the array to a supported data type
    array = array.astype(np.int32)  # Or np.float32, depending on your needs

    # Create a default affine if none is provided
    if affine is None:
        affine = np.diag(spacing + (1.0,))

    # Create a NIfTI image
    nifti_image = nib.Nifti1Image(array, affine)

    # Save the image to the specified file path
    nib.save(nifti_image, file_path)
    print(f"Saved NIfTI file to {file_path}")
    

def evaluate_segmentation(predictions, ground_truth, num_classes, include_background=True):
    """
    Evaluate segmentation performance using Dice score, Hausdorff Distance, and Average Volumetric Difference.

    Args:
        predictions (torch.Tensor): Predicted labels with shape [N, D, H, W].
        ground_truth (torch.Tensor): Ground truth labels with shape [N, D, H, W].
        num_classes (int): Number of segmentation classes.
        include_background (bool): Whether to include the background class in the evaluation.

    Returns:
        dict: A dictionary containing Dice scores, Hausdorff Distances, and Average Volumetric Differences.
    """
    # Compute Dice scores
    dice = dice_score_3d(predictions, ground_truth, num_classes)
    mean_dice = np.mean(list(dice.values())[1:])

    # Compute Hausdorff Distances
    hd = hausdorff_distance(predictions, ground_truth, num_classes, include_background=include_background)
    mean_hd = np.mean(list(hd.values())[1:])

    # Compute Average Volumetric Differences
    avd = average_volumetric_difference(predictions, ground_truth, num_classes)
    mean_avd = np.mean(list(avd.values())[1:])

    # Print results
    print(f"Dice scores: {dice}")
    print(f"Mean Dice score: {mean_dice}")
    print(f"Hausdorff Distances: {hd}")
    print(f"Mean Hausdorff Distance: {mean_hd}")
    print(f"Average Volumetric Differences: {avd}")
    print(f"Mean Average Volumetric Difference: {mean_avd}")

    # Return metrics
    return {
        "Dice": dice,
        "Mean Dice": mean_dice,
        "Hausdorff Distances": hd,
        "Mean Hausdorff Distance": mean_hd,
        "Average Volumetric Differences": avd,
        "Mean Average Volumetric Difference": mean_avd,
    }