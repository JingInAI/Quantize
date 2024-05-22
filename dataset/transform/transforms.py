"""
Transforms
version: 0.0.4
update: 2024-04-19
"""
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from utils import Configs, Register, most_similar
from .custom_funcs import CUSTOMFUNCS

TRANSFORMS = Register({
    'random_resized_crop': transforms.RandomResizedCrop,
    'random_horizontal_flip': transforms.RandomHorizontalFlip,
    'random_vertical_flip': transforms.RandomVerticalFlip,
    'random_rotation': transforms.RandomRotation,
    'random_affine': transforms.RandomAffine,
    'color_jitter': transforms.ColorJitter,
    'to_tensor': transforms.ToTensor,
    'normalize': transforms.Normalize,
    'resize': transforms.Resize,
    'center_crop': transforms.CenterCrop,
    'pad': transforms.Pad,
    'lambda': transforms.Lambda,
    'random_apply': transforms.RandomApply,
    'random_choice': transforms.RandomChoice,
    'random_crop': transforms.RandomCrop,
    'random_order': transforms.RandomOrder,
    'random_grayscale': transforms.RandomGrayscale,
    'random_perspective': transforms.RandomPerspective,
    'random_erasing': transforms.RandomErasing,
    'five_crop': transforms.FiveCrop,
    'ten_crop': transforms.TenCrop,
    'linear_transformation': transforms.LinearTransformation,
    'grayscale': transforms.Grayscale,
    'gaussian_blur': transforms.GaussianBlur,
})


def build_transform(trans):
    """Build transform.
    Args:
        trans (Configs): transform config
    Returns:
        torchvision.transforms.Compose: transform
    """
    
    transform = []

    for t in trans.cfg.keys():
        if t not in TRANSFORMS:
            raise ValueError(
                f'Unknown transform {t}. ' + 
                f'Do you mean "{most_similar(t, TRANSFORMS.keys())}"?')

        transform.append(
            TRANSFORMS[t](
                **parse_parameters(trans.cfg[t])
        ))

    return transforms.Compose(transform)


def parse_parameters(params) -> dict:
    """ Parse parameters for transforms.
    Args:
        params : parameters for transforms
    Returns:
        dict: parsed parameters
    """
    def check_augments(params: dict, names: list):
        for name in names:
            if name in params:
                params[name] = build_transform(Configs(params[name]))

    if not isinstance(params, dict):
        return {}
    else:
        if 'interpolation' in params:
            params['interpolation'] = _interpolation_modes_from_str(params['interpolation'])

        check_augments(params, ['baseaugment', 'preaugment', 'preprocess'])

        if 'custom_funcs' in params:
            custom_funcs = params['custom_funcs']
            if isinstance(custom_funcs, str):
                custom_funcs = [custom_funcs]
            params['custom_funcs'] = [CUSTOMFUNCS[func] for func in custom_funcs]
            
        return params


def _interpolation_modes_from_str(mode: str) -> InterpolationMode:
    inverse_modes_mapping = {
        'nearest': InterpolationMode.NEAREST,
        'bilinear': InterpolationMode.BILINEAR,
        'bicubic': InterpolationMode.BICUBIC,
        'box': InterpolationMode.BOX,
        'hamming': InterpolationMode.HAMMING,
        'lanczos': InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[mode]
