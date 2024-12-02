"""This module contains the DiceLoss class, which is used to compute the Dice loss."""

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        """
        Initializes the Dice loss.

        Args:
            smooth (float): A small constant to avoid division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """
        Computes the Dice loss.

        Args:
            y_pred (torch.Tensor): Predictions from the model (between 0 and 1).
            y_true (torch.Tensor): Ground-truth labels.

        Returns:
            torch.Tensor: The Dice loss.
        """
        intersection = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred)
        dice = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        return dice