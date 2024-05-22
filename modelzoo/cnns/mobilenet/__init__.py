"""
MobileNet V1/V2/V3
version: 0.0.2
update: 2023-12-28
"""
import torchvision.models as models
from .mobilenetv1 import mobilenet_v1

from modelzoo.load import MODELS

MODELS.register({
    'mobilenetv1': mobilenet_v1,
    'mobilenetv2': models.mobilenet_v2,
    'mobilenetv3_large': models.mobilenet_v3_large,
    'mobilenetv3_small': models.mobilenet_v3_small,
})
