"""
ImageNet-A dataset.
version: 0.0.2
update: 2023-12-14
"""
import os

from dataset.base import BasicDataset, Datum
from .imagenet import ImageNet


class ImageNet_A(BasicDataset):
    """ImageNet-A dataset.
    
    Args:
        root (str): root directory of the dataset.
        transform (callable): a function/transform to preprocess the image.
    """

    subdir = 'imagenet-a'

    def __init__(
        self,
        root,
        transform=None,
        **kwargs
    ):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.image_dir = os.path.join(self.root, self.subdir)

        # Parse classnames
        text_file = os.path.join(self.root, 'classnames.txt')
        classnames = ImageNet.read_classnames(text_file)

        # Load samples
        self.samples, self._classnames = self.read_data(classnames)
        self._num_classes = len(self._classnames)
        
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample: Datum = self.samples[index]
        impath = sample.impath
    
        img = self.load_image(impath)
        if self.transform is not None:
            img = self.transform(img)

        return {
            'img': img,
            'label': sample.label,
            'classname': sample.classname,
        }
    
    def read_data(self, classnames):
        """Return a list of Datum objects.
        Args:
            classnames (dict): key-value pairs of <folder name>: <class name>.
        Returns:
            items (list): a list of Datum objects.
            used_class (list): a list of used class names.
        """
        folders = sorted(f.name for f in os.scandir(self.image_dir) if f.is_dir())
        used_class = []
        items = []

        for label, folder in enumerate(folders):
            img_folder = os.path.join(self.image_dir, folder)
            imnames = [f for f in os.listdir(img_folder) if not f.startswith('.')]
            classname = classnames[folder]
            used_class.append(classname)

            for imname in imnames:
                impath = os.path.join(img_folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items, used_class
