"""This module contains the DiceLoss class, which is used to compute the Dice loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    """
    __name__ = 'dice_loss'
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Computes Dice Loss.
        Args:
            logits (torch.Tensor): Raw predictions (shape: [B, C, H, W]).
            targets (torch.Tensor): Ground truth (shape: [B, H, W]).
        Returns:
            torch.Tensor: Dice Loss value.
        """
        # Ensure logits and targets are compatible
        if len(targets.shape) == 3:  # Not one-hot encoded
            num_classes = logits.size(1)
            targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)

        # Compute intersection and union
        intersection = (probs * targets).sum(dim=(2, 3))  # Sum over spatial dimensions
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss
        return 1 - dice.mean()
    
    
class DiceCrossEntropyLoss(nn.Module):
    __name__ = 'dice_cross_entropy_loss'
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


class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, smooth=1e-6, lambda_dice=0.5, lambda_focal=0.5):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal

    def dice_loss(self, logits, targets):
        """
        Computes Dice Loss.
        Args:
            logits (torch.Tensor): Raw predictions (shape: [B, C, H, W]).
            targets (torch.Tensor): Ground truth (shape: [B, H, W] with class indices).
        Returns:
            torch.Tensor: Dice Loss value.
        """
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).permute(0, 3, 1, 2).float()  # [B, C, H, W]

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))  # Sum over spatial dimensions
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def focal_loss(self, logits, targets):
        """
        Computes Focal Loss.
        Args:
            logits (torch.Tensor): Raw predictions (shape: [B, C, H, W]).
            targets (torch.Tensor): Ground truth (shape: [B, H, W] with class indices).
        Returns:
            torch.Tensor: Focal Loss value.
        """
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # [B, H, W]
        pt = torch.exp(-ce_loss)  # Probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

    def forward(self, logits, targets):
        """
        Combines Dice Loss and Focal Loss.
        Args:
            logits (torch.Tensor): Raw predictions (shape: [B, C, H, W]).
            targets (torch.Tensor): Ground truth (shape: [B, H, W] with class indices).
        Returns:
            torch.Tensor: Combined loss value.
        """
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        return self.lambda_dice * dice + self.lambda_focal * focal