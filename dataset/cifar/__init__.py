from .cifar10 import CIFAR10
from .cifar100 import CIFAR100
from .cifar10c import CIFAR10C

from dataset.loader import DATASETS

DATASETS.register({
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'cifar10c': CIFAR10C,
})
