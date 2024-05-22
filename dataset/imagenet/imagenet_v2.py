"""
ImageNet-V2 dataset.
version: 0.0.2
update: 2023-12-14
"""
import os

from dataset.base import Datum
from .imagenet_a import ImageNet_A


class ImageNet_V2(ImageNet_A):
    """ImageNet-V2 dataset.

    Args:
        root (str): root directory of the dataset.
        transform (callable): a function/transform to preprocess the image.
    """

    subdir = 'imagenetv2-matched-frequency-format-val'

    def __init__(self, **kwargs):
        super(ImageNet_V2, self).__init__(**kwargs)

    def read_data(self, classnames):
        """Return a list of Datum objects.
        Args:
            classnames (dict): key-value pairs of <folder name>: <class name>.
        Returns:
            items (list): a list of Datum objects.
            used_class (list): a list of used class names.
        """
        folders = list(classnames.keys())
        used_class = []
        items = []

        for label in range(1000):
            img_folder = os.path.join(self.image_dir, str(label))
            imnames = [f for f in os.listdir(img_folder) if not f.startswith('.')]
            classname = classnames[folders[label]]
            used_class.append(classname)

            for imname in imnames:
                impath = os.path.join(img_folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items, used_class
