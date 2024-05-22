# Dataset Library

This library contains implementations of various datasets,
including the following:

| Name | Description |
|---|---|
| `imagenet` | `Quantize.dataset.imagenet.ImageNet` |
| `imagenet_a` | `Quantize.dataset.imagenet.ImageNet_A` |
| `imagenet_r` | `Quantize.dataset.imagenet.ImageNet_R` |
| `imagenet_sketch` | `Quantize.dataset.imagenet.ImageNet_Sketch` |
| `imagenet_v2` | `Quantize.dataset.imagenet.ImageNet_V2` |
| `imagenet_c` | `Quantize.dataset.imagenet.ImageNet_C` |
| `cifar10` | `Quantize.dataset.cifar.CIFAR10` |
| `cifar10c` | `Quantize.dataset.cifar.CIFAR10C` |
| `cifar100` | `Quantize.dataset.cifar.CIFAR100` |


# Transform Library

This library contains implementations of various data transformations,
including the following:

### Official Transforms

| Name | Description |
|---|---|
| `random_resized_crop` | `torchvision.transforms.RandomResizedCrop` |
| `random_horizontal_flip` | `torchvision.transforms.RandomHorizontalFlip` |
| `random_vertical_flip` | `torchvision.transforms.RandomVerticalFlip` |
| `random_rotation` | `torchvision.transforms.RandomRotation` |
| `random_affine` | `torchvision.transforms.RandomAffine` |
| `color_jitter` | `torchvision.transforms.ColorJitter` |
| `to_tensor` | `torchvision.transforms.ToTensor` |
| `normalize` | `torchvision.transforms.Normalize` |
| `resize` | `torchvision.transforms.Resize` |
| `center_crop` | `torchvision.transforms.CenterCrop` |
| `pad` | `torchvision.transforms.Pad` |
| `lambda` | `torchvision.transforms.Lambda` |
| `random_apply` | `torchvision.transforms.RandomApply` |
| `random_choice` | `torchvision.transforms.RandomChoice` |
| `random_crop` | `torchvision.transforms.RandomCrop` |
| `random_order` | `torchvision.transforms.RandomOrder` |
| `random_grayscale` | `torchvision.transforms.RandomGrayscale` |
| `random_perspective` | `torchvision.transforms.RandomPerspective` |
| `random_erasing` | `torchvision.transforms.RandomErasing` |
| `five_crop` | `torchvision.transforms.FiveCrop` |
| `ten_crop` | `torchvision.transforms.TenCrop` |
| `linear_transformation` | `torchvision.transforms.LinearTransformation` |
| `grayscale` | `torchvision.transforms.Grayscale` |
| `gaussian_blur` | `torchvision.transforms.Gaussian` |


### Custom Transforms

| Name | Description|
|---|---|
| `augmix` | `Quantize.dataset.transform.AugMix` |
| `augexpand` | `Quantize.dataset.transform.AugExpand` |
| `random_rotate` | `Quantize.dataset.transform.custom_funcs.random_rotate` |
| `rotate_with_labels` | `Quantize.dataset.transform.custom_funcs.rotate_with_labels` |
