from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    shufflenet_v2_x0_5,
    ShuffleNet_V2_X0_5_Weights,
)


def adapt_stem(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Rebuild a conv with new input channels, preserving spatial params."""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=getattr(conv, "dilation", 1),
        groups=getattr(conv, "groups", 1),
        bias=conv.bias is not None,
    )


def build_mobilenetv3_small(
    num_classes: int = 7, in_channels: int = 1, pretrained: bool = True
) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    m = mobilenet_v3_small(weights=weights)
    old = m.features[0][0]
    m.features[0][0] = adapt_stem(old, in_channels)
    in_feat = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_feat, num_classes)
    return m


def build_shufflenetv2_0_5x(
    num_classes: int = 7, in_channels: int = 1, pretrained: bool = True
) -> nn.Module:
    weights = ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
    m = shufflenet_v2_x0_5(weights=weights)
    old = m.conv1[0]
    m.conv1[0] = adapt_stem(old, in_channels)
    in_feat = m.fc.in_features
    m.fc = nn.Linear(in_feat, num_classes)
    return m
