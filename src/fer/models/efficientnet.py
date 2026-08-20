from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    efficientnet_b2,
    EfficientNet_B2_Weights,
)


class ECA(nn.Module):
    """Efficient Channel Attention (ECA) — 1D conv over GAP, 3 params."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean(dim=(2, 3))  # (B, C)
        y = y.unsqueeze(1)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = self.sigmoid(y)  # (B, 1, C)
        y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


class EfficientNetB0(nn.Module):
    """1-channel EfficientNet-B0 adapted from legacy project (model.py).

    Replaces the ImageNet RGB stem with a 1-channel conv (same kernel/stride/
    padding), preserving the pre-trained backbone feature extraction.
    Set eca=True to add a lightweight ECA gate after features (exp. 1280-ch).
    """

    def __init__(self, num_classes: int = 7, in_channels: int = 1, pretrained: bool = True, eca: bool = False):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)

        old_conv = self.model.features[0][0]  # nn.Conv2d
        self.model.features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        self.eca: nn.Module | None = ECA(k=3) if eca else None

        in_feat = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_feat, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.features(x)
        if self.eca is not None:
            x = self.eca(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x


class EfficientNetB2(nn.Module):
    """1-channel EfficientNet-B2 capacity-matched baseline (~0.043 GFLOPs, 7.71 M params).

    Replaces the ImageNet RGB stem with a 1-channel conv (same kernel/stride/
    padding), preserving the pre-trained backbone feature extraction.
    """

    def __init__(self, num_classes: int = 7, in_channels: int = 1, pretrained: bool = True, eca: bool = False):
        super().__init__()
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b2(weights=weights)

        old_conv = self.model.features[0][0]  # nn.Conv2d
        self.model.features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        self.eca: nn.Module | None = ECA(k=3) if eca else None

        in_feat = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_feat, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.features(x)
        if self.eca is not None:
            x = self.eca(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x

