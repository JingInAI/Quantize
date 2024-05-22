"""
Vision Transformer (ViT)
version: 0.0.1
update: 2023-12-28
"""
import torchvision.models as models

from modelzoo.load import MODELS

MODELS.register({
    'vit_b_16': models.vit_b_16,
    'vit_b_32': models.vit_b_32,
    'vit_l_16': models.vit_l_16,
    'vit_l_32': models.vit_l_32,
    'vit_h_14': models.vit_h_14,
})
