"""
Base dataset class.
version: 0.0.3
update: 2024-02-12
"""
import os
import random
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset

from utils import get_logger


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class Datum():
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath="", image=None, label=0, domain=0, classname=""):
        assert isinstance(impath, str)
        if impath:
            assert os.path.isfile(impath), f"No file found at {impath}"

        self._impath = impath
        self._image = image
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath
    
    @property
    def image(self):
        return self._image

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class BasicDataset(Dataset):
    """Basic dataset class."""

    def __init__(self):
        self._num_classes = 0
        self._classnames = []

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def classnames(self):
        return self._classnames

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=False
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        get_logger().info(f"Creating a {num_shots}-shot dataset")

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def load_image(self, impath):
        """Load an image from disk.

        Args:
            impath (str): image path.
        """
        return pil_loader(impath)
