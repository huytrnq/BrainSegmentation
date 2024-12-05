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
    
    
class DiceBCELoss(nn.Module):
    """
    Dice Loss combined with Binary Cross-Entropy Loss.
    """

    def __init__(self, dice_weight=1, bce_weight=1, smooth=1e-6):
        """
        Initialize DiceBCELoss.

        Args:
            dice_weight (float): Weight of Dice Loss.
            bce_weight (float): Weight of Binary Cross-Entropy Loss.
            smooth (float): Smoothing factor to prevent division by zero.
        """
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        """
        Compute DiceBCELoss.

        Args:
            predictions (torch.Tensor): Predicted logits of shape (N, C, H, W) or (N, C, H, W, D).
            targets (torch.Tensor): One-hot encoded ground truth of the same shape as predictions.

        Returns:
            torch.Tensor: DiceBCE loss (scalar).
        """
        # Compute Dice Loss
        dice_loss = self.dice_loss(predictions, targets)

        # Compute Binary Cross-Entropy Loss
        bce_loss = self.bce_loss(predictions, targets)

        # Combine Dice Loss and BCE Loss
        loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss

        return loss