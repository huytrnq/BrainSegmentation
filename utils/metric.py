"""
This module contains a class for tracking metrics and implementing early stopping.
It is used to monitor the performance of the model during training and validation.
"""

import torch


class MetricsMonitor:
    """Monitor for tracking metrics and implementing early stopping."""

    def __init__(self, metrics=None, patience=5, delta=0.0001, mode="min", save_path="best_model.pth"):
        """
        Combines metric tracking and early stopping with real-time updates.

        Args:
            metrics (list): List of metric names to track (e.g., ['loss', 'accuracy']).
            patience (int): Patience for early stopping.
            delta (float): Minimum change to qualify as an improvement for early stopping.
            mode (str): 'min' for loss (lower is better) or 'max' for accuracy (higher is better).
            save_path (str): File path to save the best model.
        """
        self.metrics = metrics if metrics else ["loss", "accuracy"]
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.save_path = save_path
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
        torch.save(model.state_dict(), self.save_path)
        print(f"\nModel improved and saved to {self.save_path}!")


def dice_coefficient(labels, predictions, eps=1e-7):
    """
    Compute the Dice coefficient for a batch of ground truth masks and predicted masks.

    Args:
        labels (torch.Tensor): Ground truth masks (shape: [N, *]).
        predictions (torch.Tensor): Predicted masks (shape: [N, *]).
        eps (float): Smoothing value to avoid division by zero.

    Returns:
        torch.Tensor: Dice coefficient (mean over the batch).
    """
    intersection = torch.sum(labels * predictions)
    union = torch.sum(labels) + torch.sum(predictions)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()

def accuracy(predictions, labels):
    """
    Calculate batch accuracy with an option to include or exclude background class.

    Args:
        predictions (torch.Tensor): Predicted class labels (B, H, W, D).
        labels (torch.Tensor): Ground truth labels (B, H, W, D).

    Returns:
        float: Batch accuracy as a value between 0 and 1.
    """

    if len(labels) == 0:  # If no foreground voxels exist, return 0 accuracy
        return 0.0
    # Convert one-hot predictions and labels to class indices
    predictions = predictions.argmax(dim=1)  # Shape: (batch_size, H, W)
    labels = labels.argmax(dim=1)  # Shape: (batch_size, H, W)
    
    # exclude background class
    

    accuracy = (predictions == labels).float().mean().item()
    return accuracy