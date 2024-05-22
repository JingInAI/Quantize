"""
AugMix
version: 0.0.2
update: 2023-12-14
"""
import torch
import numpy as np
from PIL import ImageOps, Image

from .transforms import TRANSFORMS

__all__ = ['AugMix']


@TRANSFORMS.register
class AugMix(torch.nn.Module):
    """An unofficial implementation of AugMix.

    AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty
    https://arxiv.org/abs/1912.02781

    Args:
        preaugment (torchvision.transforms.Compose): pre-augmentation before applying AugMix.
        preprocess (torchvision.transforms.Compose): pre-processing after applying AugMix.
        baseaugment (torchvision.transforms.Compose, optional): 
            augmentation to generate an original view without AugMix,
            if None, original view is not included in the output.
        apply_augmix (bool, optional): whether to apply AugMix for output views.
        n_views (int, optional): number of output views.
        severity (int, optional): severity of AugMix.
    
    Returns:
        torch.Tensor: AugMixed images, the size is (n_views, C, H, W),
            if baseaugment is not None, the first view is the original image.

    Examples:
        >>> from torchvision import transforms
        >>> from dataset.augmix import AugMix
        >>> mean = [0.485, 0.456, 0.406]
        >>> std = [0.229, 0.224, 0.225]
        >>> preprocess = transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     transforms.Normalize(mean, std)
        >>> ])
        >>> preaugment = transforms.Compose([
        >>>     transforms.RandomResizedCrop(224),
        >>>     transforms.RandomHorizontalFlip()
        >>> ])
        >>> augmix = AugMix(
        >>>     preaugment,
        >>>     preprocess,
        >>>     baseaugment=None,
        >>>     apply_augmix=True,
        >>>     n_views=2,
        >>>     severity=1
        >>> )
        >>> image = Image.open('path/to/image.jpg')
        >>> augmix(image)
        >>> # output: torch.Tensor (2, 3, 224, 224)
    """

    def __init__(
        self,
        preaugment,
        preprocess,
        baseaugment=None,
        apply_augmix=True,
        n_views=2,
        severity=1,
    ):
        super().__init__()
        self.preaugment = preaugment
        self.preprocess = preprocess
        self.baseaugment = baseaugment
        self.apply_augmix = apply_augmix

        if apply_augmix:
            self.aug_list = [
                autocontrast,
                equalize,
                rotate,
                solarize,
                shear_x,
                shear_y,
                translate_x,
                translate_y,
                posterize,
            ]
        else:
            self.aug_list = []

        self.n_views = n_views
        self.severity = severity

    def augmix(self, image, aug_list=[], severity=1):
        """Apply AugMix to an image.
        Args:
            image (PIL.Image): input image.
            aug_list (list): list of augmentation functions.
            severity (int): severity of augmentation.
        Returns:
            torch.Tensor: AugMixed image.
        """
        x_orig = self.preaugment(image)
        x_processed = self.preprocess(x_orig)
        if len(aug_list) == 0:
            return x_processed

        w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
        m = np.float32(np.random.beta(1.0, 1.0))

        mix = torch.zeros_like(x_processed)
        for i in range(3):
            x_aug = x_orig.copy()
            for _ in range(np.random.randint(1, 4)):
                x_aug = np.random.choice(aug_list)(x_aug, severity)
            mix += w[i] * self.preprocess(x_aug)
        mix = m * x_processed + (1 - m) * mix
        return mix

    def forward(self, img):
        """Apply AugMix to an image.
        Args:
            img (PIL.Image): input images.
        Returns:
            torch.Tensor: AugMixed images.
        """
        if self.baseaugment:
            base_img = [self.preprocess(self.baseaugment(img))]
        else:
            base_img = []
        
        views = [
            self.augmix(img, self.aug_list, self.severity) 
            for _ in range(self.n_views - len(base_img))]

        return torch.stack(base_img + views)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        format_string += f'preaugment={self.preaugment}, '
        format_string += f'preprocess={self.preprocess}, '
        format_string += f'baseaugment={self.baseaugment}, '
        format_string += f'apply_augmix={self.apply_augmix}, '
        format_string += f'n_views={self.n_views}, '
        format_string += f'severity={self.severity})'
        return format_string

def autocontrast(pil_img, level=None):
    return ImageOps.autocontrast(pil_img)

def equalize(pil_img, level=None):
    return ImageOps.equalize(pil_img)

def rotate(pil_img, level):
    degrees = int_parameter(rand_lvl(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR, fillcolor=128)

def solarize(pil_img, level):
    level = int_parameter(rand_lvl(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)

def shear_x(pil_img, level):
    level = float_parameter(rand_lvl(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR, fillcolor=128)

def shear_y(pil_img, level):
    level = float_parameter(rand_lvl(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR, fillcolor=128)

def translate_x(pil_img, level):
    level = int_parameter(rand_lvl(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR, fillcolor=128)

def translate_y(pil_img, level):
    level = int_parameter(rand_lvl(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR, fillcolor=128)

def posterize(pil_img, level):
    level = int_parameter(rand_lvl(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)

def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)

def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.

def rand_lvl(n):
    return np.random.uniform(low=0.1, high=n)
