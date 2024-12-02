"""Utility functions for training and testing the model."""

import torch
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
        batch_accuracy = accuracy(outputs, labels)
        dice_score = dice_coefficient(labels, outputs)

        # Update monitor with loss and accuracy
        monitor.update("loss", loss.item(), count=len(images))
        monitor.update("accuracy", batch_accuracy, count=len(images))
        monitor.update("dice_score", dice_score, count=len(images))

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
        for batch_idx, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate batch accuracy
            batch_accuracy = accuracy(outputs, labels)
            dice_score = dice_coefficient(labels, outputs)

            # Update monitor with loss and accuracy
            monitor.update("loss", loss.item(), count=len(images))
            monitor.update("accuracy", batch_accuracy, count=len(images))
            monitor.update("dice_score", dice_score, count=len(images))

            # Print iteration metrics
            monitor.print_iteration(batch_idx + 1, len(valid_loader), phase="Validation")

    # Print final metrics for the validation epoch
    monitor.print_final(phase="Validation")
    return {metric: monitor.compute_average(metric) for metric in monitor.metrics}