from .imagenet import ImageNet
from .imagenet_a import ImageNet_A
from .imagenet_r import ImageNet_R
from .imagenet_sketch import ImageNet_Sketch
from .imagenet_v2 import ImageNet_V2
from .imagenet_c import ImageNet_C

from dataset.loader import DATASETS

DATASETS.register({
    'imagenet': ImageNet,
    'imagenet_a': ImageNet_A,
    'imagenet_r': ImageNet_R,
    'imagenet_sketch': ImageNet_Sketch,
    'imagenet_v2': ImageNet_V2,
    'imagenet_c': ImageNet_C,
})
