# Runners

Runners are frameworks used to implement quantization methods on a neural network. These runners are compatible with [PyTorch](https://pytorch.org/) framework.



## Post-Training Quantization (PTQ) ([A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295))


| Method | Description | Config | Script |
| --- | --- | --- | --- |
| `minmax_layer` | Per-tensor quantization with Min-Max scaling | [config](../configs/runners/ptq/weight_quantize/minmax_layer.yaml) | [script](../scripts/ptq/weight_quantize.sh) |
| `minmax_channel` | Per-channel quantization with Min-Max scaling | [config](../configs/runners/ptq/weight_quantize/minmax_channel.yaml) | [script](../scripts/ptq/weight_quantize.sh) |
| `mse_layer` | Per-tensor quantization with MSE scaling | [config](../configs/runners/ptq/weight_quantize/mse_layer.yaml) | [script](../scripts/ptq/weight_quantize.sh) |
| `mse_channel` | Per-channel quantization with MSE scaling | [config](../configs/runners/ptq/weight_quantize/mse_channel.yaml) | [script](../scripts/ptq/weight_quantize.sh) |
| `bias_correct` | Bias correction for quantization | [config](../configs/runners/ptq/bias_correct) | [script](../scripts/ptq/bias_correct.sh) |
| `awq` | [Activation-aware weight quantization](https://arxiv.org/abs/2306.00978) | [config](../configs/runners/ptq/awq/base.yaml) | [script](../scripts/ptq/awq.sh) |



## AdaRound ([Up or Down? Adaptive Rounding for Post-Training Quantization](https://arxiv.org/abs/2004.10568))

| Method | Description | Config | Script |
| --- | --- | --- | --- |
| `adaround` | Adaptive rounding for PTQ | [config](../configs/runners/adaround) | [script](../scripts/adaround) |



## Quantization-Aware Training (QAT) ([A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295))

| Method | Description | Config | Script |
| --- | --- | --- | --- |
| `qat` | Quantization-aware training | [config](../configs/runners/qat) | [script](../scripts/qat) |



## Activation Quantization ([A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295))

| Method | Description | Config | Script |
| --- | --- | --- | --- |
| `minmax_layer` | Per-tensor quantization with Min-Max scaling | [config](../configs/runners/ptq/activation_quantize/minmax_layer.yaml) | [script](../scripts/ptq/activation_quantize.sh) |
| `minmax_channel` | Per-channel quantization with Min-Max scaling | [config](../configs/runners/ptq/activation_quantize/minmax_channel.yaml) | [script](../scripts/ptq/activation_quantize.sh) |
| `mse_layer` | Per-tensor quantization with MSE scaling | [config](../configs/runners/ptq/activation_quantize/mse_layer.yaml) | [script](../scripts/ptq/activation_quantize.sh) |
| `mse_channel` | Per-channel quantization with MSE scaling | [config](../configs/runners/ptq/activation_quantize/mse_channel.yaml) | [script](../scripts/ptq/activation_quantize.sh) |
| `aciq_layer` | Per-tensor quantization with [ACIQ](https://arxiv.org/abs/1810.05723) | [config](../configs/runners/ptq/activation_quantize/aciq_layer.yaml) | [script](../scripts/ptq/activation_quantize.sh) |
| `aciq_channel` | Per-channel quantization with [ACIQ](https://arxiv.org/abs/1810.05723) | [config](../configs/runners/ptq/activation_quantize/aciq_channel.yaml) | [script](../scripts/ptq/activation_quantize.sh) |
