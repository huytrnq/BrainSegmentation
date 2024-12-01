"""Utility functions for training and testing the model."""

import torch


def calculate_batch_accuracy(predictions, labels, include_background=True):
    """
    Calculate batch accuracy with an option to include or exclude background class.

    Args:
        predictions (torch.Tensor): Predicted class labels (B, H, W, D).
        labels (torch.Tensor): Ground truth labels (B, H, W, D).
        include_background (bool): Whether to include background class in accuracy calculation.

    Returns:
        float: Batch accuracy as a value between 0 and 1.
    """
    predictions = torch.argmax(predictions, dim=1)  # Get predicted class
    if not include_background:
        mask = labels > 0  # Exclude background (assume class 0 is background)
        predictions = predictions[mask]
        labels = labels[mask]

    if len(labels) == 0:  # If no foreground voxels exist, return 0 accuracy
        return 0.0

    accuracy = (predictions == labels).float().mean().item()
    return accuracy


def train(model, train_loader, criterion, optimizer, device, monitor, include_background=True):
    """
    Train the model for one epoch and track metrics.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for the training dataset.
        criterion: Loss function.
        optimizer: Optimizer for updating model weights.
        device: Device to run the training on (CPU or GPU).
        monitor: MetricsMonitor object for tracking metrics.
        include_background (bool): Whether to include background in accuracy calculation.

    Returns:
        dict: Dictionary with average metrics for the epoch.
    """
    model.train()
    monitor.reset()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate batch accuracy
        batch_accuracy = calculate_batch_accuracy(outputs, labels, include_background)

        # Update monitor with loss and accuracy
        monitor.update("loss", loss.item(), count=len(images))
        monitor.update("accuracy", batch_accuracy, count=len(images))

        # Print iteration metrics
        monitor.print_iteration(batch_idx + 1, len(train_loader), phase="Train")

    # Print final metrics for the epoch
    monitor.print_final(phase="Train")
    return {metric: monitor.compute_average(metric) for metric in monitor.metrics}


def validate(model, valid_loader, criterion, device, monitor, include_background=True):
    """
    Validate the model for one epoch and track metrics.

    Args:
        model: PyTorch model to evaluate.
        valid_loader: DataLoader for the validation dataset.
        criterion: Loss function.
        device: Device to run the evaluation on (CPU or GPU).
        monitor: MetricsMonitor object for tracking metrics.
        include_background (bool): Whether to include background in accuracy calculation.

    Returns:
        dict: Dictionary with average metrics for the validation epoch.
    """
    model.eval()
    monitor.reset()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate batch accuracy
            batch_accuracy = calculate_batch_accuracy(outputs, labels, include_background)

            # Update monitor with loss and accuracy
            monitor.update("loss", loss.item(), count=len(images))
            monitor.update("accuracy", batch_accuracy, count=len(images))

            # Print iteration metrics
            monitor.print_iteration(batch_idx + 1, len(valid_loader), phase="Validation")

    # Print final metrics for the validation epoch
    monitor.print_final(phase="Validation")
    return {metric: monitor.compute_average(metric) for metric in monitor.metrics}