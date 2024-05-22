"""
CIFAR-10 datasets.
version: 0.0.1
update: 2024-02-08
"""
from PIL import Image

from torchvision.datasets import CIFAR10 as _CIFAR10
from dataset.base import BasicDataset, Datum


class CIFAR10(BasicDataset, _CIFAR10):
    """CIFAR-10 dataset.

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
        _CIFAR10.__init__(self, root, train=(split=='train'), transform=transform, download=download)

        # Parse classnames
        self._classnames = self.classes
        self._num_classes = len(self._classnames)

        # Parse samples
        self.samples = self.read_data(self.data, self.targets, self.classnames)

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
        
    def read_data(self, images, targets, classnames):
        """Return a list of Datum objects.
        Args:
            images: list of images.
            targets: list of labels.
            classnames: list of classnames.
        """
        samples = []
        for img, label in zip(images, targets):
            item = Datum(image=img, label=label, classname=classnames[label])
            samples.append(item)
        return samples
