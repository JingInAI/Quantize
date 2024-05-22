"""
ImageNet-R dataset.
version: 0.0.2
update: 2023-12-14
"""
from .imagenet_a import ImageNet_A


class ImageNet_R(ImageNet_A):
    """ImageNet-R dataset.
    
    Args:
        root (str): root directory of the dataset.
        transform (callable): a function/transform to preprocess the image.
    """

    subdir = 'imagenet-r'

    def __init__(self, **kwargs):
        super(ImageNet_R, self).__init__(**kwargs)
