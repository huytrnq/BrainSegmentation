"""
This module contains a class for tracking metrics and implementing early stopping.
It is used to monitor the performance of the model during training and validation.
"""

import numpy as np
import torch


class MetricsMonitor:
    """Monitor for tracking metrics and implementing early stopping."""

    def __init__(self, metrics=None, patience=5, delta=0.0001, mode="min", export_path="best_model.pth"):
        """
        Combines metric tracking and early stopping with real-time updates.

        Args:
            metrics (list): List of metric names to track (e.g., ['loss', 'accuracy']).
            patience (int): Patience for early stopping.
            delta (float): Minimum change to qualify as an improvement for early stopping.
            mode (str): 'min' for loss (lower is better) or 'max' for accuracy (higher is better).
            export_path (str): File path to save the best model.
        """
        self.metrics = metrics if metrics else ["loss", "accuracy"]
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.export_path = export_path
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.reset()

    def reset(self):
        """
        Resets metrics and early stopping variables for a new epoch or phase.
        """
        self.metric_totals = {metric: 0.0 for metric in self.metrics}
        self.metric_counts = {metric: 0 for metric in self.metrics}

    def update(self, metric_name, value, count=1):
        """
        Updates a specific metric with a new value.

        Args:
            metric_name (str): Name of the metric to update.
            value (float): Value to add to the metric.
            count (int): Number of samples contributing to the metric (default is 1).
        """
        if metric_name in self.metric_totals:
            self.metric_totals[metric_name] += value * count
            self.metric_counts[metric_name] += count
        else:
            raise ValueError(f"Metric '{metric_name}' is not being tracked.")

    def compute_average(self, metric_name):
        """
        Computes the average value for a specific metric.

        Args:
            metric_name (str): Name of the metric to compute.

        Returns:
            float: The average value of the metric.
        """
        if self.metric_counts[metric_name] > 0:
            return self.metric_totals[metric_name] / self.metric_counts[metric_name]
        return 0.0

    def print_iteration(self, iteration, total_iterations, phase="Train"):
        """
        Prints real-time metrics for each iteration.

        Args:
            iteration (int): Current iteration index.
            total_iterations (int): Total number of iterations in the epoch.
            phase (str): Name of the current phase (e.g., 'Train', 'Validation').
        """
        metrics_str = ", ".join(
            f"{metric}: {self.compute_average(metric):.4f}" for metric in self.metrics
        )
        print(
            f"\r[{phase}] Iteration {iteration}/{total_iterations} - {metrics_str}",
            end="",
            flush=True,
        )

    def print_final(self, phase="Train"):
        """
        Prints the final average values of all tracked metrics.

        Args:
            phase (str): Name of the current phase (e.g., 'Train', 'Validation').
        """
        metrics_str = ", ".join(
            f"{metric}: {self.compute_average(metric):.4f}" for metric in self.metrics
        )
        print(f"\n{phase} Metrics - {metrics_str}")

    def early_stopping_check(self, metric, model):
        """
        Implements early stopping based on a monitored metric.

        Args:
            metric (float): The validation metric to monitor.
            model: The model to save if performance improves.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        score = -metric if self.mode == "max" else metric
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            print(f"Early stopping patience counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Save the best model to the specified file."""
        torch.save(model.state_dict(), self.export_path)
        print(f"\nModel improved and saved to {self.export_path}!")


def dice_coefficient(predictions, labels, num_classes=4, smooth=1e-6):
    """
    Calculate Dice Score for each class.
    Args:
        predictions (torch.Tensor): Model predictions (logits), shape [batch_size, num_classes, height, width].
        labels (torch.Tensor): Ground truth labels, shape [batch_size, height, width].
        num_classes (int): Number of classes including background.
        smooth (float): Smoothing factor to avoid division by zero.
    Returns:
        list: Dice score for each class.
    """
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
        
    # Convert logits to probabilities
    predictions = torch.softmax(predictions.float(), dim=1)  # Shape [batch_size, num_classes, height, width]

    # Convert labels to one-hot encoding
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes).permute(0, 3, 1, 2).float()
    # Shape [batch_size, num_classes, height, width]

    dice_scores = []
    for c in range(num_classes):
        # Extract probabilities and labels for the current class
        pred = predictions[:, c]  # Shape [batch_size, height, width]
        true = one_hot_labels[:, c]  # Shape [batch_size, height, width]

        # Calculate intersection and union
        intersection = (pred * true).sum(dim=(1, 2))  # Sum over height and width
        union = pred.sum(dim=(1, 2)) + true.sum(dim=(1, 2))  # Sum over height and width

        # Compute Dice score
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.mean().item())  # Average over the batch

    return dice_scores


def accuracy(predictions, labels, background_class=-1):
    """
    Calculate the accuracy for segmentation.

    Args:
        predictions (torch.Tensor): Predicted class indices of shape (N, H, W) or (N, H, W, D).
        labels (torch.Tensor): Ground truth class indices of shape (N, H, W) or (N, H, W, D).
        background_class (int): Index of the background class.

    Returns:
        float: Accuracy score.
    """
    # Convert probabilities to class predictions
    num_classes = predictions.size(1)  # Number of classes
    predictions = torch.argmax(predictions, dim=1)
    # Step 2: Convert class predictions to one-hot encoding
    one_hot_pred = torch.nn.functional.one_hot(predictions, num_classes=num_classes)
    one_hot_pred = one_hot_pred.permute(0, 3, 1, 2)

    
    # Ensure predictions and labels have the same shape
    if one_hot_pred.shape != labels.shape:
        raise ValueError("Shape of predictions and labels must match.")

    # Create a mask to ignore the background class
    mask = labels != background_class

    # Mask predictions and labels
    masked_predictions = one_hot_pred[mask]
    masked_labels = labels[mask]

    # Compute accuracy
    correct = (masked_predictions == masked_labels).sum().item()
    total = mask.sum().item()

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def dice_score_3d(prediction, ground_truth, num_classes, smooth=1e-6):
    """
    Compute the Dice Score for multi-class 3D volumes using PyTorch.

    Args:
        prediction (torch.Tensor): Predicted segmentation (shape: [Z, Y, X]).
        ground_truth (torch.Tensor): Ground truth segmentation (shape: [Z, Y, X]).
        num_classes (int): Number of classes.
        smooth (float): Small smoothing factor to avoid division by zero.

    Returns:
        dict: Dice Score for each class.
    """
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction)
    if isinstance(ground_truth, np.ndarray):
        ground_truth = torch.from_numpy(ground_truth)
        
    dice_scores = {}
    for class_id in range(num_classes):
        # Create binary masks for the current class
        pred_class = (prediction == class_id).float()
        gt_class = (ground_truth == class_id).float()

        # Compute Dice Score
        intersection = torch.sum(pred_class * gt_class)
        union = torch.sum(pred_class) + torch.sum(gt_class)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores[class_id] = dice.item()

    return dice_scores