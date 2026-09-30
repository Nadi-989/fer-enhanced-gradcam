"""Backbones and the layers each CAM method reads from.

For every architecture we expose:
  deep   - last convolutional block (standard Grad-CAM target layer)
  fusion - three blocks of decreasing resolution used by MSF-Grad-CAM
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Small 4-block CNN baseline trained from scratch."""

    def __init__(self, num_classes: int):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
            )

        self.block1, self.block2 = block(3, 32), block(32, 64)
        self.block3, self.block4 = block(64, 128), block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4),
                                  nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.block4(x)
        return self.head(x)


def _disable_inplace(model: nn.Module) -> None:
    # In-place activations break gradient hooks on the tensors CAM methods read.
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False


def _weights(enum, pretrained):
    return enum.DEFAULT if pretrained else None


def build_model(arch: str, num_classes: int, pretrained: bool = True):
    arch = arch.lower()
    if arch in ("resnet18", "resnet50"):
        ctor, wenum = ((models.resnet18, models.ResNet18_Weights) if arch == "resnet18"
                       else (models.resnet50, models.ResNet50_Weights))
        m = ctor(weights=_weights(wenum, pretrained))
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
        layers = {"deep": m.layer4, "fusion": [m.layer2, m.layer3, m.layer4]}
    elif arch == "vgg16":
        m = models.vgg16_bn(weights=_weights(models.VGG16_BN_Weights, pretrained))
        m.avgpool = nn.AdaptiveAvgPool2d(1)
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, num_classes))
        f = m.features  # ReLU at the end of blocks 3, 4, 5
        layers = {"deep": f[42], "fusion": [f[22], f[32], f[42]]}
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=_weights(models.EfficientNet_B0_Weights, pretrained))
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, num_classes))
        f = m.features
        layers = {"deep": f[8], "fusion": [f[3], f[5], f[8]]}
    elif arch == "simplecnn":
        m = SimpleCNN(num_classes)
        layers = {"deep": m.block4, "fusion": [m.block2, m.block3, m.block4]}
    else:
        raise ValueError(f"Unknown arch '{arch}'. Use resnet18, resnet50, vgg16, efficientnet_b0, simplecnn.")
    _disable_inplace(m)
    return m, layers
