"""
CIFAR-100 dataset
version: 0.0.1
update: 2024-02-08
"""
from .cifar10 import CIFAR10


class CIFAR100(CIFAR10):
    """CIFAR-100 dataset.

    Args:
        root (str): root directory of the dataset.
        split (str): split of the dataset (train/val/test).
        transform (callable): a function/transform to preprocess the image.
        download (bool): if True, download the dataset from the internet and put it in root directory.
    """

    def __init__(
        self,
        root,
        split: str = 'train',
        transform = None,
        download: bool = False,
    ):
        super().__init__(root, split, transform, download)
