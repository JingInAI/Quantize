"""
ResNet
version: 0.0.1
update: 2023-12-14
"""
import torchvision.models as models

from modelzoo.load import MODELS

MODELS.register({
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'resnext50_32x4d': models.resnext50_32x4d,
    'resnext101_32x8d': models.resnext101_32x8d,
    'resnext101_64x4d': models.resnext101_64x4d,
    'wide_resnet50_2': models.wide_resnet50_2,
    'wide_resnet101_2': models.wide_resnet101_2,
})
