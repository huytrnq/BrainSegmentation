"""This module contains the DiceLoss class, which is used to compute the Dice loss."""

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    """

    def __init__(self, smooth=1e-6):
        """
        Initialize DiceLoss.

        Args:
            smooth (float): Smoothing factor to prevent division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Compute Dice Loss.

        Args:
            predictions (torch.Tensor): Predicted probabilities of shape (N, C, H, W) or (N, C, H, W, D).
            targets (torch.Tensor): One-hot encoded ground truth of the same shape as predictions.

        Returns:
            torch.Tensor: Dice loss (scalar).
        """
        # Convert predictions to probabilities using softmax
        predictions = torch.softmax(predictions, dim=1)
        
        # Flatten the tensors to compute Dice score
        predictions = predictions.contiguous().view(predictions.shape[0], predictions.shape[1], -1)
        targets = targets.contiguous().view(targets.shape[0], targets.shape[1], -1)

        # Calculate intersection and union
        intersection = torch.sum(predictions * targets, dim=2)
        union = torch.sum(predictions, dim=2) + torch.sum(targets, dim=2)

        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice loss is 1 - Dice coefficient
        loss = 1 - dice.mean()

        return loss
    
    
class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, smooth=1e-6):
        """
        Combined Dice and Cross-Entropy Loss for multi-class segmentation.

        Args:
            weight (torch.Tensor, optional): Class weights for CrossEntropyLoss.
            ignore_index (int, optional): Index to ignore in CrossEntropyLoss.
            dice_weight (float): Weight for the Dice Loss component.
            ce_weight (float): Weight for the Cross-Entropy Loss component.
            smooth (float): Smoothing factor for Dice Loss to avoid division by zero.
        """
        super(DiceCrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth

    def forward(self, logits, labels):
        """
        Compute the combined loss.

        Args:
            logits (torch.Tensor): Model predictions (logits) of shape [batch_size, num_classes, height, width].
            labels (torch.Tensor): Ground truth labels of shape [batch_size, height, width].

        Returns:
            torch.Tensor: Combined loss value.
        """
        # Cross-Entropy Loss
        ce_loss = self.ce_loss(logits, labels)

        # Dice Loss
        num_classes = logits.size(1)
        probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes).permute(0, 3, 1, 2).float()
        dice_loss = self._dice_loss(probs, one_hot_labels)

        # Combined Loss
        combined_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        return combined_loss

    def _dice_loss(self, probs, one_hot_labels):
        """
        Compute Dice Loss for multi-class predictions.

        Args:
            probs (torch.Tensor): Softmax probabilities of shape [batch_size, num_classes, height, width].
            one_hot_labels (torch.Tensor): One-hot encoded ground truth of shape [batch_size, num_classes, height, width].

        Returns:
            torch.Tensor: Dice loss value.
        """
        intersection = (probs * one_hot_labels).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + one_hot_labels.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()