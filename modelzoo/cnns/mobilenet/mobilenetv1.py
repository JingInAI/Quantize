"""
MobileNetV1
version: 0.0.1
update: 2023-12-27
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """ BasicBlock for MobileNetV1
    
    Args:
        inplanes (int): input channels
        planes (int): output channels
        stride (int): stride of the first conv layer. Default: 1
    """
    def __init__(
        self, 
        inplanes: int,
        planes: int,
        stride: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=stride, padding=1, groups=inplanes, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class MobileNetV1(nn.Module):
    """ MobileNetV1
    
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861

    Args:
        num_classes (int): number of classes. Default: 1000
        in_channels (int): input channels. Default: 3

    Examples:
    
        >>> from modelzoo.cnns.mobilenet import mobilenetv1
        >>> model = mobilenetv1()
        >>> print(model)
    
    """
    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(32, 64, 1, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 6, 2)
        self.layer5 = self._make_layer(512, 1024, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    

def mobilenet_v1(pretrained=False, **kwargs):
    model = MobileNetV1(**kwargs)
    if pretrained:
        raise NotImplementedError("Pretrained MobileNetV1 are not supported yet.")
    return model
