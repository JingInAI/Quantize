# Model Zoo

This directory contains a collection of pre-trained models for various tasks. These models are compatible with [PyTorch](https://pytorch.org/) framework.

## Convolutional Neural Networks (CNNs)

### ResNet ([Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385))

| Name | Description |
|---|---|
| `resnet18` | `torchvision.models.resnet18` |
| `resnet34` | `torchvision.models.resnet34` |
| `resnet50` | `torchvision.models.resnet50` |
| `resnet101` | `torchvision.models.resnet101` |
| `resnet152` | `torchvision.models.resnet152` |
| `resnext50_32x4d` | `torchvision.models.resnext50_32x4d` |
| `resnext101_32x8d` | `torchvision.models.resnext101_32x8d` |
| `resnext101_64x4d` | `torchvision.models.resnext101_64x4d` |


### Wide ResNet ([Wide Residual Networks](https://arxiv.org/abs/1605.07146))

| Name | Description |
|---|---|
| `wide_resnet50_2` | `torchvision.models.wide_resnet50_2` |
| `wide_resnet101_2` | `torchvision.models.wide_resnet101_2` |
| `wrn_40_2` | `Quantize.modelzoo.cnns.wideresnet.wrn_40_2` |


### MobileNet ([MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861))

| Name | Description |
|---|---|
| `mobilenetv1` | `Quantize.modelzoo.cnns.mobilenet.mobilenetv1.mobilenet_v1` |


### MobileNetV2 ([MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381))

| Name | Description |
|---|---|
| `mobilenetv2` | `torchvision.models.mobilenet_v2` |


### MobileNetV3 ([Searching for MobileNetV3](https://arxiv.org/abs/1905.02244v2))

| Name | Description |
|---|---|
| `mobilenetv3_large` | `torchvision.models.mobilenet_v3_large` |
| `mobilenetv3_small` | `torchvision.models.mobilenet_v3_small` |



## Transformers

### Vision Transformer (ViT) ([An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929))

| Name | Description |
|---|---|
| `vit_b_16` | `torchvision.models.vit_b_16` |
| `vit_b_32` | `torchvision.models.vit_b_32` |
| `vit_l_16` | `torchvision.models.vit_l_16` |
| `vit_l_32` | `torchvision.models.vit_l_32` |
| `vit_h_14` | `torchvision.models.vit_h_14` |


## Language-Vision Models

### CLIP ([Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020))

| Name | Description |
|---|---|
| `clip_rn50` | `Quantize.modelzoo.clip.clip_rn50` |
| `clip_rn101` | `Quantize.modelzoo.clip.clip_rn101` |
| `clip_rn50x4` | `Quantize.modelzoo.clip.clip_rn50x4` |
| `clip_rn50x16` | `Quantize.modelzoo.clip.clip_rn50x16` |
| `clip_rn50x64` | `Quantize.modelzoo.clip.clip_rn50x64` |
| `clip_vit-b32` | `Quantize.modelzoo.clip.clip_vitb32` |
| `clip_vit-b16` | `Quantize.modelzoo.clip.clip_vitb16` |
| `clip_vit-l14` | `Quantize.modelzoo.clip.clip_vitl14` |
| `clip_vit-l14@336px` | `Quantize.modelzoo.clip.clip_vitl14_336px` |



# Modules

This directory contains a collection of modules that can be used to build quantized neural networks. These modules are compatible with [PyTorch](https://pytorch.org/) framework.

### Quantized Modules

| Name | Description |
|---|---|
| `quantizer` | `Quantize.modelzoo.modules.Quantizer` |
| `quantlinear` | `Quantize.modelzoo.modules.QuantLinear` |
| `quantconv2d` | `Quantize.modelzoo.modules.QuantConv2d` |
| `quantrelu` | `Quantize.modelzoo.modules.QuantReLU` |
| `quantmaxpool2d` | `Quantize.modelzoo.modules.QuantMaxPool2d` |
| `quantadaptiveavgpool2d` | `Quantize.modelzoo.modules.QuantAdaptiveAvgPool2d` |
| `quantmultiheadattention` | `Quantize.modelzoo.modules.QuantMultiheadAttention` |


### Quantization Tools

| Name | Description |
|---|---|
| `minmax` | `Quantize.modelzoo.modules.range.MinMax` |
| `maminmax` | `Quantize.modelzoo.modules.range.MAMinMax` |
| `mse` | `Quantize.modelzoo.modules.range.MSE` |
| `cross_entropy` | `Quantize.modelzoo.modules.range.CrossEntropy` |
| `bias_correct` | `Quantize.modelzoo.modules.range.BiasCorrect` |
| `aciq` | `Quantize.modelzoo.modules.range.ACIQ` |
| `adaround` | `Quantize.modelzoo.modules.range.AdaRound` |
| `awq` | `Quantize.modelzoo.modules.range.AWQ` |


### Custom Operators

| Name | Description |
|---|---|
| `quantlinear_forward` | `Quantize.modelzoo.modules.operator.quantlinear_forward` |
| `quantconv2d_forward` | `Quantize.modelzoo.modules.operator.quantconv2d_forward` |
