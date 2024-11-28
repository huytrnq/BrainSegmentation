"""
This module contains a custom PyTorch dataset class for the skin lesion dataset.
It is used to load the images and their corresponding labels (if available) from the disk.
"""

from collections import Counter
from torch.utils.data import Dataset
from PIL import Image
