"""
CLIP Model
version: 0.0.1
update: 2024-03-21
"""
import torch
import torch.nn as nn
from tqdm import tqdm

from .model import *
from .clip import *
from .coop import *
from .cocoop import *


class CLIPModel(nn.Module):
    def __init__(self, backbone, classnames, prompts=None, precision='fp32', *args, **kwargs):
        super().__init__()
        if backbone not in clip._MODELS.keys():
            raise ValueError(f"clip backbone {backbone} not found in available models")
        
        self.model = clip.load(backbone, device='cpu')[0]
        if precision == 'fp32':
            self.model.float()
        elif precision == 'fp16':
            convert_weights(self.model)

        self.classnames = classnames
        assert len(self.classnames) > 0, "classnames should not be empty."

        if prompts is None or len(prompts) == 0:
            self.prompts = ['a photo of a {}.']
        else:
            self.prompts = prompts
        
        self.zeroshot_classifier(self.classnames, self.prompts)
    
    def zeroshot_classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []

            for classname in tqdm(classnames, dynamic_ncols=True, desc='Creating zeroshot classifier'):
                texts = [template.format(classname) for template in templates]  # format with class
                texts = tokenize(texts).cpu()  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)

        self.zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cpu()
    
    def forward(self, images):
        image_features = self.model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ self.zeroshot_weights.to(image_features.device)

        return logits


def clip_rn50(*args, **kwargs):
    return CLIPModel('RN50', *args, **kwargs)

def clip_rn101(*args, **kwargs):
    return CLIPModel('RN101', *args, **kwargs)

def clip_rn50x4(*args, **kwargs):
    return CLIPModel('RN50x4', *args, **kwargs)

def clip_rn50x16(*args, **kwargs):
    return CLIPModel('RN50x16', *args, **kwargs)

def clip_rn50x64(*args, **kwargs):
    return CLIPModel('RN50x64', *args, **kwargs)

def clip_vitb32(*args, **kwargs):
    return CLIPModel('ViT-B/32', *args, **kwargs)

def clip_vitb16(*args, **kwargs):
    return CLIPModel('ViT-B/16', *args, **kwargs)

def clip_vitl14(*args, **kwargs):
    return CLIPModel('ViT-L/14', *args, **kwargs)

def clip_vitl14_336px(*args, **kwargs):
    return CLIPModel('ViT-L/14@336px', *args, **kwargs)


from modelzoo.load import MODELS

MODELS.register({
    'clip_rn50': clip_rn50,
    'clip_rn101': clip_rn101,
    'clip_rn50x4': clip_rn50x4,
    'clip_rn50x16': clip_rn50x16,
    'clip_rn50x64': clip_rn50x64,
    'clip_vit-b32': clip_vitb32,
    'clip_vit-b16': clip_vitb16,
    'clip_vit-l14': clip_vitl14,
    'clip_vit-l14@336px': clip_vitl14_336px,
})
