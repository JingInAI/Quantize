"""
ImageNet-C dataset
version: 0.0.1
update: 2024-02-08
"""
import os

from dataset.base import BasicDataset, Datum
from .imagenet import ImageNet


LEVELS = [1, 2, 3, 4, 5]
CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
               'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


class ImageNet_C(BasicDataset):
    """ImageNet-C dataset.

    Args:
        root (str): root directory of the dataset.
        corruption (str): corruption type.
        level (int): corruption level.
        transform (callable): a function/transform to preprocess the image.
    """

    def __init__(
        self,
        root,
        corruption: str,
        level: int,
        transform=None,
    ):
        assert corruption in CORRUPTIONS, f"Invalid corruption type: {corruption}"
        assert level in LEVELS, f"Invalid corruption level: {level}"

        self.root = os.path.abspath(os.path.expanduser(root))
        self.image_dir = os.path.join(self.root, corruption, str(level))

        # Parse classnames
        text_file = os.path.join(self.root, 'classnames.txt')
        classnames = ImageNet.read_classnames(text_file)
        self._classnames = list(classnames.values())
        self._num_classes = len(classnames)

        # Load samples
        self.samples = self.read_data(classnames, self.image_dir)
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

    def read_data(self, classnames, image_dir) -> list:
        """Return a list of Datum objects.
        Args:
            classnames (dict): key-value pairs of <folder name>: <class name>.
            image_dir (str): image directory (e.g., /path/to/imagenet-c/gaussian_noise/1/)
        """
        folders = sorted(f.name for f in os.scandir(image_dir) if f.is_dir())
        samples = []

        for label, folder in enumerate(folders):
            image_folder = os.path.join(image_dir, folder)
            imnames = [f for f in os.listdir(image_folder) if not f.startswith('.')]
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                samples.append(item)

        return samples
