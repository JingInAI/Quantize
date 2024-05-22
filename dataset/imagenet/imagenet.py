"""
ImageNet dataset.
version: 0.0.2
update: 2023-12-14
"""
import os
import math
import pickle
from collections import OrderedDict

from utils import get_logger, get_cfg
from dataset.base import BasicDataset, Datum


class ImageNet(BasicDataset):
    """ImageNet dataset.

    Args:
        root (str): root directory of the dataset.
        split (str): split of the dataset (train/val/test).
        subclass (str): subclass of the dataset (all/base/new), 
            base means the first 500 classes, new means the last 500 classes.
        num_shots (int): number of shots. If <=0, use the original dataset.
        transform (callable): a function/transform to preprocess the image.
    """

    def __init__(
        self,
        root,
        split='train',
        subclass='all',
        num_shots=-1,
        transform=None,
    ):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.image_dir = self.root
        self.preprocessed = os.path.join(self.root, "imagenet.pkl")
        self.split_fewshot_dir = os.path.join(self.root, "split_fewshot")
        os.makedirs(self.split_fewshot_dir, exist_ok=True)
        logger = get_logger()

        # Parse classnames
        text_file = os.path.join(self.root, 'classnames.txt')
        classnames = self.read_classnames(text_file)
        self._classnames = list(classnames.values())
        self._num_classes = len(classnames)

        # Load preprocessed whole dataset
        if os.path.exists(self.preprocessed):
            logger.info(f"Loading preprocessed whole data from {self.preprocessed}")
            with open(self.preprocessed, 'rb') as f:
                preprocessed = pickle.load(f)
                train = preprocessed['train']
                test = preprocessed['test']
        else:
            train = self.read_data(classnames, 'train')
            test = self.read_data(classnames, 'val')

            logger.info(f"Saving preprocessed whole data to {self.preprocessed}")
            preprocessed = {'train': train, 'test': test}
            with open(self.preprocessed, 'wb') as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Generate few-shot subset
        if num_shots >= 1 and split == 'train':
            seed = get_cfg().seed
            preprocessed = os.path.join(self.split_fewshot_dir, f"{num_shots}shot_seed{seed}.pkl")

            if os.path.exists(preprocessed) and seed >= 0:
                logger.info(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, 'rb') as f:
                    data = pickle.load(f)
                    train = data['train']
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)

                if seed >= 0:
                    logger.info(f"Saving preprocessed few-shot data to {preprocessed}")
                    data = {'train': train}
                    with open(preprocessed, 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.samples = train if split == 'train' else test
        self.samples = self.select_subclass(self.samples, subclass)
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

    @staticmethod
    def read_classnames(text_file) -> OrderedDict:
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir) -> list:
        """Return a list of Datum objects.
        Args:
            classnames (dict): key-value pairs of <folder name>: <class name>.
            split_dir (str): split directory (train/val).
        """
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            split_folder = os.path.join(split_dir, folder)
            imnames = [f for f in os.listdir(split_folder) if not f.startswith('.')]
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
    
    def select_subclass(self, samples: list, subclass='all'):
        """Return a list of samples of the given subclass.
        Args:
            subclass (str): subclass of the dataset (all/base/new), 
                base means the first 500 classes, new means the last 500 classes.
            samples (list): a list of samples.
        Returns:
            a list of samples of the given subclass.
        """
        assert subclass in ['all', 'base', 'new']

        if subclass == 'all':
            return samples

        get_logger().info(f"Selecting {subclass} classes")
        labels = set()
        for item in samples:
            labels.add(item.label)
        labels = sorted(list(labels))

        m = math.ceil(len(labels) / 2)
        if subclass == 'base':
            selected = labels[:m]
        else:
            selected = labels[m:]
        relabeler = {label: i for i, label in enumerate(selected)}

        output = []
        subclassnames = OrderedDict()
        for item in samples:
            if item.label in selected:
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname)
                output.append(item_new)
                subclassnames[item_new.label] = item_new.classname

        self._classnames = list(subclassnames.values())
        self._num_classes = len(subclassnames)
        
        return output
