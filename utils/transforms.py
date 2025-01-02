import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

class RobustZNormalization(ImageOnlyTransform):
    """
    Custom Albumentations transform for robust z-normalization using the 25th and 75th quantiles.
    """
    def __init__(self, always_apply=True, p=1.0):
        super(RobustZNormalization, self).__init__(always_apply, p)

    def apply(self, img, **params):
        """
        Normalize the image using robust z-normalization.

        Args:
            img (np.ndarray): Input image to normalize.

        Returns:
            np.ndarray: Normalized image.
        """
        q25 = np.percentile(img, 25)
        q75 = np.percentile(img, 75)
        mean = (q25 + q75) / 2
        std = (q75 - q25) / 2  # Interquartile range as robust std

        if std > 0:
            img = (img - mean) / std
        return img