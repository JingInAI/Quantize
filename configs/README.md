# Configs and Experiments

This directory contains the configuration files for the quantization experiments. 
The configuration files are written in YAML format and are used to define the quantization settings for the experiments.
We provide experimental results for corresponding configurations as examples.


## Pre-trained on ImageNet-1K

Pre-trained models on ImageNet-1K train split are available in the [Model Zoo](../modelzoo).
We report the Top-1 accuracy of quantized models on ImageNet-1K eval split.
Note that `WmAn` denotes the weight bit-width `m` and activation bit-width `n`.

### ResNet-18

| Method | W32A32 | W8A32 | W7A32 | W6A32 | W5A32 | W4A32 | W3A32 | W2A32 |
|--------|--------|-------|-------|-------|-------|-------|-------|-------|
| PTQ/minmax_layer <br> ([config](runners/ptq/weight_quantize/minmax_layer.yaml)) | 69.76 | 69.55 | 68.32 | 63.91 | 19.36 | 0.12 | 0.10 | 0.10 |
| PTQ/minmax_channel <br> ([config](runners/ptq/weight_quantize/minmax_channel.yaml)) | 69.76 | 69.63 | 69.63 | 68.87 | 65.02 | 44.52 | 0.38 | 0.10 |
| PTQ/mse_layer <br> ([config](runners/ptq/weight_quantize/mse_layer.yaml)) | 69.76 | 69.47 | 68.18 | 66.04 | 53.11 | 24.24 | 0.85 | 0.13 |
| PTQ/mse_channel <br> ([config](runners/ptq/weight_quantize/mse_channel.yaml)) | 69.76 | 69.70 | 69.57 | 69.28 | 65.16 | 55.03 | 5.42 | 0.12 |
| PTQ/awq <br> ([config](runners/ptq/awq/base.yaml)) | 69.76 | 69.70 | 69.61 | 69.26 | 65.26 | 54.98 | 5.27 | 0.11 |
| PTQ/minmax_layer+ <br> bias_correct ([config](runners/ptq/bias_correct/minmax_layer.yaml)) | 69.76 | 69.58 | 68.96 | 65.80 | 40.04 | 0.38 | 0.10 | 0.10 |
| PTQ/minmax_channel+ <br> bias_correct ([config](runners/ptq/bias_correct/minmax_channel.yaml)) | 69.76 | 69.68 | 69.69 | 69.52 | 67.81 | 58.57 | 24.13 | 0.10 |
| PTQ/mse_layer+ <br> bias_correct ([config](runners/ptq/bias_correct/mse_layer.yaml)) | 69.76 | 69.65 | 68.88 | 68.03 | 61.68 | 39.23 | 4.17 | 0.12 |
| PTQ/mse_channel+ <br> bias_correct ([config](runners/ptq/bias_correct/mse_channel.yaml)) | 69.76 | 69.72 | 69.66 | 69.47 | 68.48 | 64.64 | 30.39 | 0.12 |
| PTQ/awq+ <br> bias_correct ([config](runners/ptq/bias_correct/awq.yaml)) | 69.76 | 69.74 | 69.64 | 69.47 | 68.56 | 64.60 | 28.71 | 0.10 |
| AdaRound/minmax_layer <br> ([config](runners/adaround/weight_quantize/minmax_layer.yaml)) | 69.76 | 69.62 | 69.50 | 69.01 | 67.08 | 50.15 | 0.10 | 0.10 |
| AdaRound/minmax_channel <br> ([config](runners/adaround/weight_quantize/minmax_channel.yaml)) | 69.76 | 69.68 | 69.66 | 69.63 | 69.36 | 68.14 | 60.26 | 0.14 |
| AdaRound/mse_layer <br> ([config](runners/adaround/weight_quantize/mse_layer.yaml)) | 69.76 | 69.67 | 69.63 | 69.53 | 68.87 | 67.73 | 63.36 | 0.34 |
| AdaRound/mse_channel <br> ([config](runners/adaround/weight_quantize/mse_channel.yaml)) | 69.76 | 69.68 | 69.64 | 69.63 | 69.44 | 68.85 | 66.04 | 47.62 |
| AdaRound/awq <br> ([config](runners/adaround/awq/base.yaml)) | 69.76 | 69.70 | 69.63 | 69.63 | 69.30 | 68.57 | 64.57 | 16.24 |
| AdaRound/minmax_layer+ <br> bias_correct ([config](runners/adaround/bias_correct/minmax_layer.yaml)) | 69.76 | 69.70 | 69.59 | 69.20 | 67.69 | 53.68 | 0.10 | 0.10 |
| AdaRound/minmax_channel+ <br> bias_correct ([config](runners/adaround/bias_correct/minmax_channel.yaml)) | 69.76 | 69.74 | 69.68 | 69.65 | 69.56 | 68.58 | 62.48 | 0.11 |
| AdaRound/mse_layer+ <br> bias_correct ([config](runners/adaround/bias_correct/mse_layer.yaml)) | 69.76 | 69.70 | 69.68 | 69.64 | 69.16 | 68.14 | 64.45 | 0.32 |
| AdaRound/mse_channel+ <br> bias_correct ([config](runners/adaround/bias_correct/mse_channel.yaml)) | 69.76 | 69.72 | 69.70 | 69.66 | 69.48 | 68.97 | 66.71 | 51.85 |
| AdaRound/awq+ <br> bias_correct ([config](runners/adaround/bias_correct/awq.yaml)) | 69.76 | 69.71 | 69.73 | 69.66 | 69.42 | 68.69 | 66.01 | 23.48 |



| Method | W32A32 | W32A8 | W32A7 | W32A6 | W32A5 | W32A4 | W32A3 | W32A2 |
|--------|--------|-------|-------|-------|-------|-------|-------|-------|
| PTQ/minmax_layer <br> ([config](runners/ptq/activation_quantize/minmax_layer.yaml)) | 69.76 | 69.66 | 69.48 | 68.28 | 61.56 | 25.99 | 0.65 | 0.10 |
| PTQ/minmax_channel <br> ([config](runners/ptq/activation_quantize/minmax_channel.yaml)) | 69.76 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| PTQ/mse_layer <br> ([config](runners/ptq/activation_quantize/mse_layer.yaml)) | 69.76 | 69.71 | 69.65 | 69.38 | 68.48 | 65.16 | 52.54 | 10.62 |
| PTQ/mse_channel <br> ([config](runners/ptq/activation_quantize/mse_channel.yaml)) | 69.76 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| PTQ/aciq_layer <br> ([config](runners/ptq/activation_quantize/aciq_layer.yaml)) | 69.76 | 67.71 | 66.90 | 65.28 | 62.38 | 53.94 | 30.51 | 5.10 |
| PTQ/aciq_channel <br> ([config](runners/ptq/activation_quantize/aciq_channel.yaml)) | 69.76 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |






### ViT-B/32

| Method | W32A32 | W8A32 | W7A32 | W6A32 | W5A32 | W4A32 | W3A32 | W2A32 |
|--------|--------|-------|-------|-------|-------|-------|-------|-------|
| PTQ/minmax_layer <br> ([config](runners/ptq/weight_quantize/minmax_layer.yaml)) | 75.92 | 75.81 | 75.48 | 74.20 | 52.09 | 0.27 | 0.12 | 0.11 |
| PTQ/minmax_channel <br> ([config](runners/ptq/weight_quantize/minmax_channel.yaml)) | 75.92 | 75.91 | 75.87 | 75.83 | 75.51 | 72.58 | 3.23 | 0.11 |
| PTQ/mse_layer <br> ([config](runners/ptq/weight_quantize/mse_layer.yaml)) | 75.92 | 75.89 | 75.75 | 75.36 | 73.19 | 53.41 | 2.16 | 0.11 |
| PTQ/mse_channel <br> ([config](runners/ptq/weight_quantize/mse_channel.yaml)) | 75.92 | 75.88 | 75.87 | 75.90 | 75.51 | 74.21 | 60.47 | 0.10 |
| PTQ/awq <br> ([config](runners/ptq/awq/base.yaml)) | 75.92 | 75.91 | 75.89 | 75.85 | 75.75 | 74.73 | 58.64  | 0.10 |
| PTQ/minmax_layer+ <br> bias_correct ([config](runners/ptq/bias_correct/minmax_layer.yaml)) | 75.92 | 75.80 | 75.53 | 74.25 | 60.93 | 0.41 | 0.14 | 0.09 |
| PTQ/minmax_channel+ <br> bias_correct ([config](runners/ptq/bias_correct/minmax_channel.yaml)) | 75.92 | 75.89 | 75.91 | 75.82 | 75.54 | 73.68 | 35.32 | 0.18 |
| PTQ/mse_layer+ <br> bias_correct ([config](runners/ptq/bias_correct/mse_layer.yaml)) | 75.92 | 75.84 | 75.68 | 75.32 | 73.98 | 63.38 | 4.54 | 0.07 |
| PTQ/mse_channel+ <br> bias_correct ([config](runners/ptq/bias_correct/mse_channel.yaml)) | 75.92 | 75.90 | 75.83 | 75.85 | 75.65 | 74.55 | 63.81 | 0.13 |
| PTQ/awq+ <br> bias_correct ([config](runners/ptq/bias_correct/awq.yaml)) | 75.92 | 75.91 | 75.89 | 75.85 | 75.73 | 75.11 | 69.08 | 0.11 |
| AdaRound/minmax_layer <br> ([config](runners/adaround/weight_quantize/minmax_layer.yaml)) | 75.92 | 75.88 | 75.77 | 75.59 | 74.58 | 66.08 | 1.34 | 0.10 |
| AdaRound/minmax_channel <br> ([config](runners/adaround/weight_quantize/minmax_channel.yaml)) | 75.92 | 75.90 | 75.91 | 75.86 | 75.73 | 75.49 | 73.37 | 1.32 |
| AdaRound/mse_layer <br> ([config](runners/adaround/weight_quantize/mse_layer.yaml)) | 75.92 | 75.88 | 75.83 | 75.85 | 75.71 | 75.56 | 73.90 | 35.06 |
| AdaRound/mse_channel <br> ([config](runners/adaround/weight_quantize/mse_channel.yaml)) | 75.92 | 75.92 | 75.86 | 75.84 | 75.81 | 75.62 | 74.93 | 71.12 |
| AdaRound/awq <br> ([config](runners/adaround/awq/base.yaml)) | 75.92 | 75.87 | 75.89 | 75.84 | 75.74 | 75.53 | 74.27 | 21.31 |
| AdaRound/minmax_layer+ <br> bias_correct ([config](runners/adaround/bias_correct/minmax_layer.yaml)) | 75.92 | 75.88 | 75.71 | 75.59 | 74.47 | 66.49 | 2.60 | 0.13 |
| AdaRound/minmax_channel+ <br> bias_correct ([config](runners/adaround/bias_correct/minmax_channel.yaml)) | 75.92 | 75.90 | 75.94 | 75.88 | 75.75 | 75.48 | 73.60 | 2.61 |
| AdaRound/mse_layer+ <br> bias_correct ([config](runners/adaround/bias_correct/mse_layer.yaml)) | 75.92 | 75.89 | 75.82 | 75.86 | 75.71 | 75.56 | 73.98 | 43.98 |
| AdaRound/mse_channel+ <br> bias_correct ([config](runners/adaround/bias_correct/mse_channel.yaml)) | 75.92 | 75.91 | - | - | - | - | - | - |
| AdaRound/awq+ <br> bias_correct ([config](runners/adaround/bias_correct/awq.yaml)) | 75.92 | 75.89 | 75.90 | 75.84 | 75.83 | 75.61 | 74.34 | 38.26 |



| Method | W32A32 | W32A8 | W32A7 | W32A6 | W32A5 | W32A4 | W32A3 | W32A2 |
|--------|--------|-------|-------|-------|-------|-------|-------|-------|
| PTQ/minmax_layer <br> ([config](runners/ptq/activation_quantize/minmax_layer.yaml)) | 75.92 | 75.16 | 68.69 | 36.75 | 0.37 | 0.10 | 0.10 | 0.10 |
| PTQ/minmax_channel <br> ([config](runners/ptq/activation_quantize/minmax_channel.yaml)) | 75.92 | 75.82 | 75.04 | 70.57 | 14.79 | 0.14 | 0.09 | 0.09 |
| PTQ/mse_layer <br> ([config](runners/ptq/activation_quantize/mse_layer.yaml)) | 75.92 | 75.24 | 70.83 | 44.04 | 1.48 | 0.26 | 0.22 | 0.12 |
| PTQ/mse_channel <br> ([config](runners/ptq/activation_quantize/mse_channel.yaml)) | 75.92 | 75.88 | 75.23 | 72.48 | 56.03 | 1.64 | 0.33 | 0.17 |
| PTQ/aciq_layer <br> ([config](runners/ptq/activation_quantize/aciq_layer.yaml)) | 75.92 | 0.10 | 0.10 | 0.11 | 0.13 | 0.13 | 0.14 | 0.13 |
| PTQ/aciq_channel <br> ([config](runners/ptq/activation_quantize/aciq_channel.yaml)) | 75.92 | 0.10 | 0.12 | 0.11 | 0.12 | 0.11 | 0.13 | 0.08 |


