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

    def __init__(self, dice_weight=0.5, ce_weight=0.5, smooth=1e-6, is_3d=False, class_weights=None):
        """
        Combined Dice and Cross-Entropy Loss for multi-class segmentation.

        Args:
            dice_weight (float): Weight for the Dice Loss component.
            ce_weight (float): Weight for the Cross-Entropy Loss component.
            smooth (float): Smoothing factor for Dice Loss to avoid division by zero.
            is_3d (bool): Whether to use 3D segmentation (default: False for 2D segmentation).
        """
        super(DiceCrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.is_3d = is_3d  # Flag to toggle between 2D and 3D modes

    def forward(self, logits, labels):
        """
        Compute the combined loss.

        Args:
            logits (torch.Tensor): Model predictions.
                - Shape for 2D: [batch_size, num_classes, height, width].
                - Shape for 3D: [batch_size, num_classes, depth, height, width].
            labels (torch.Tensor): Ground truth labels.
                - Shape for 2D: [batch_size, height, width].
                - Shape for 3D: [batch_size, depth, height, width].

        Returns:
            torch.Tensor: Combined loss value.
        """
        # Squeeze any singleton dimensions in labels
        labels = labels.squeeze(1) if labels.dim() > logits.dim() - 1 else labels
        
        # Cross-Entropy Loss
        if self.ce_weight > 0:
            ce_loss = self.ce_loss(logits, labels)
        else:
            ce_loss = 0.0

        # Dice Loss
        num_classes = logits.size(1)
        probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities        
        
        if self.is_3d:
            one_hot_labels = F.one_hot(labels, num_classes).permute(0, 4, 1, 2, 3).float()
            dice_loss = self._dice_loss_3d(probs, one_hot_labels)
        else:
            one_hot_labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2).float()
            dice_loss = self._dice_loss_2d(probs, one_hot_labels)

        # Combined Loss
        combined_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        return combined_loss

    def _dice_loss_2d(self, probs, one_hot_labels):
        """
        Compute Dice Loss for 2D segmentation.

        Args:
            probs (torch.Tensor): Predicted probabilities, shape [batch_size, num_classes, height, width].
            one_hot_labels (torch.Tensor): One-hot encoded ground truth, shape [batch_size, num_classes, height, width].

        Returns:
            torch.Tensor: Dice loss value.
        """
        intersection = torch.sum(probs * one_hot_labels, dim=(2, 3))  # Sum over height and width
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(one_hot_labels, dim=(2, 3))  # Sum over height and width
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # Dice coefficient for each class
        dice_loss = 1 - dice.mean()  # Average over batch and classes
        return dice_loss

    def _dice_loss_3d(self, probs, one_hot_labels):
        """
        Compute Dice Loss for 3D segmentation.

        Args:
            probs (torch.Tensor): Predicted probabilities, shape [batch_size, num_classes, depth, height, width].
            one_hot_labels (torch.Tensor): One-hot encoded ground truth, shape [batch_size, num_classes, depth, height, width].

        Returns:
            torch.Tensor: Dice loss value.
        """
        intersection = torch.sum(probs * one_hot_labels, dim=(2, 3, 4))  # Sum over depth, height, and width
        union = torch.sum(probs, dim=(2, 3, 4)) + torch.sum(one_hot_labels, dim=(2, 3, 4))  # Sum over depth, height, and width
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # Dice coefficient for each class
        dice_loss = 1 - dice.mean()  # Average over batch and classes
        return dice_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceFocalLoss(nn.Module):
    __name__ = "dice_focal_loss"

    def __init__(self, dice_weight=0.5, focal_weight=0.5, smooth=1e-6, gamma=2, alpha=None, is_3d=False, ignore_background=False):
        """
        Combined Dice and Focal Loss for multi-class segmentation with option to ignore background.

        Args:
            dice_weight (float): Weight for the Dice Loss component.
            focal_weight (float): Weight for the Focal Loss component.
            smooth (float): Smoothing factor for Dice Loss to avoid division by zero.
            gamma (float): Focusing parameter for Focal Loss.
            alpha (float or list): Balancing factor for Focal Loss. Can be a scalar or a list for class weights.
            is_3d (bool): Whether to use 3D segmentation (default: False for 2D segmentation).
            ignore_background (bool): Whether to ignore background class in the loss calculation.
        """
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
        self.gamma = gamma
        self.alpha = alpha
        self.is_3d = is_3d
        self.ignore_background = ignore_background

    def forward(self, logits, labels):
        """
        Compute the combined Dice-Focal loss.

        Args:
            logits (torch.Tensor): Model predictions.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Combined loss value.
        """
        # Ensure labels have the correct dimensions
        labels = labels.squeeze(1) if labels.dim() > logits.dim() - 1 else labels

        # Compute probabilities from logits
        probs = torch.softmax(logits, dim=1)

        # Compute Focal Loss
        focal_loss = self._focal_loss(probs, labels)

        # Compute Dice Loss
        num_classes = logits.size(1)

        if self.is_3d:
            # Handle 3D case
            one_hot_labels = F.one_hot(labels, num_classes).permute(0, 4, 1, 2, 3).float()
            dice_loss = self._dice_loss_3d(probs, one_hot_labels)
        else:
            # Handle 2D case
            one_hot_labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2).float()
            dice_loss = self._dice_loss_2d(probs, one_hot_labels)

        # Combine Dice and Focal Loss
        combined_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        return combined_loss

    def _focal_loss(self, probs, labels):
        """
        Compute Focal Loss.

        Args:
            probs (torch.Tensor): Predicted probabilities.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Focal loss value.
        """
        num_classes = probs.size(1)
        one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
        one_hot_labels = one_hot_labels.permute(0, -1, *range(1, len(probs.shape) - 1))

        # Optionally ignore background class
        if self.ignore_background:
            probs = probs[:, 1:]
            one_hot_labels = one_hot_labels[:, 1:]

        pt = (probs * one_hot_labels).sum(dim=1)
        log_pt = torch.log(pt + self.smooth)

        # Compute the focal loss
        focal_loss = -(1 - pt) ** self.gamma * log_pt

        # Apply alpha if specified
        if self.alpha is not None:
            alpha = torch.tensor(self.alpha, device=probs.device)
            alpha_t = (one_hot_labels * alpha.view(1, -1, *[1] * (probs.dim() - 2))).sum(dim=1)
            focal_loss *= alpha_t

        return focal_loss.mean()

    def _dice_loss_2d(self, probs, one_hot_labels):
        """
        Compute Dice Loss for 2D segmentation.

        Args:
            probs (torch.Tensor): Predicted probabilities.
            one_hot_labels (torch.Tensor): One-hot encoded ground truth.

        Returns:
            torch.Tensor: Dice loss value.
        """
        if self.ignore_background:
            probs = probs[:, 1:]
            one_hot_labels = one_hot_labels[:, 1:]

        intersection = torch.sum(probs * one_hot_labels, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(one_hot_labels, dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def _dice_loss_3d(self, probs, one_hot_labels):
        """
        Compute Dice Loss for 3D segmentation.

        Args:
            probs (torch.Tensor): Predicted probabilities.
            one_hot_labels (torch.Tensor): One-hot encoded ground truth.

        Returns:
            torch.Tensor: Dice loss value.
        """
        if self.ignore_background:
            probs = probs[:, 1:]
            one_hot_labels = one_hot_labels[:, 1:]

        intersection = torch.sum(probs * one_hot_labels, dim=(2, 3, 4))
        union = torch.sum(probs, dim=(2, 3, 4)) + torch.sum(one_hot_labels, dim=(2, 3, 4))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()