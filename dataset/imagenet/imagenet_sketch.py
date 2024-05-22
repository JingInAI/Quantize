"""
ImageNet-Sketch dataset.
version: 0.0.2
update: 2023-12-14
"""
from .imagenet_a import ImageNet_A


class ImageNet_Sketch(ImageNet_A):
    """ImageNet-Sketch dataset.
    
    Args:
        root (str): root directory of the dataset.
        transform (callable): a function/transform to preprocess the image.
    """

    subdir = 'sketch'

    def __init__(self, **kwargs):
        super(ImageNet_Sketch, self).__init__(**kwargs)
