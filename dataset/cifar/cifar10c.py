"""
CIFAR-10-C dataset.
version: 0.0.1
update: 2024-02-08
"""
import os
import numpy as np
from PIL import Image

from torchvision.datasets import CIFAR10 as _CIFAR10
from dataset.base import BasicDataset, Datum


LEVELS = [1, 2, 3, 4, 5]
CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
               'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


class CIFAR10C(BasicDataset, _CIFAR10):
    """CIFAR-10-C dataset.

    Args:
        root (str): root directory of the dataset.
        corruption (str): corruption type.
        level (int): corruption level.
        transform (callable): a function/transform to preprocess the image.
    """
    sub_dir = "CIFAR-10-C"

    def __init__(
        self,
        root,
        corruption: str,
        level: int,
        transform=None,
    ):
        assert corruption in CORRUPTIONS, f"Invalid corruption type: {corruption}"
        assert level in LEVELS, f"Invalid corruption level: {level}"

        _CIFAR10.__init__(self, root, train=False, transform=transform, download=True)

        self.corruption = corruption
        self.level = level

        # Parse classnames
        self._classnames = self.classes
        self._num_classes = len(self._classnames)

        # Parse samples
        self.samples = self.read_data(
            npy_file=os.path.join(self.root, self.sub_dir, f"{corruption}.npy"),
            level=level
        )

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample: Datum = self.samples[index]

        img = Image.fromarray(sample.image)
        if self.transform is not None:
            img = self.transform(img)

        return {
            'img': img,
            'label': sample.label,
            'classname': sample.classname,
        }
    
    def read_data(self, npy_file: str, level: int):
        """Return a list of Datum objects.
        Args:
            npy_file (str): path to the npy file.
            level (int): corruption level.
        """
        raw_data = np.load(npy_file)
        raw_data = raw_data[(level - 1)*10000: level*10000]

        samples = []
        for img, label in zip(raw_data, self.targets):
            item = Datum(image=img, label=label, classname=self.classnames[label])
            samples.append(item)

        return samples
