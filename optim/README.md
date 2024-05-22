# Optimizer

This directory contains a collection of optimizers used to train models. These optimizers are compatible with [PyTorch](https://pytorch.org/) framework.

| Name | Description |
|---|---|
| `sgd` | `torch.optim.SGD` |
| `rmsprop` | `torch.optim.RMSprop` |
| `adam` | `torch.optim.Adam` |
| `adamw` | `torch.optim.AdamW` |


# LR Scheduler

This directory contains a collection of learning rate schedulers used to train models. These learning rate schedulers are compatible with [PyTorch](https://pytorch.org/) framework.

| Name | Description |
|---|---|
| `step` | `torch.optim.lr_scheduler.StepLR` |
| `multistep` | `torch.optim.lr_scheduler.MultiStepLR` |
| `exponential` | `torch.optim.lr_scheduler.ExponentialLR` |
| `cosine` | `torch.optim.lr_scheduler.CosineAnnealingLR` |
| `cosine_warmup` | `Quantize.optim.lr_scheduler.CosineWarmupLR` |
| `linear_warmup` | `Quantize.optim.lr_scheduler.LinearWarmupLR` |
| `constant` | `Quantize.optim.lr_scheduler.ConstantLR` |
