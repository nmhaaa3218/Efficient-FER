from __future__ import annotations

import torch
from torch import nn

from fer.models.efficientnet import EfficientNetB0, EfficientNetB2
from fer.models.lightweight import build_mobilenetv3_small, build_shufflenetv2_0_5x

_REGISTRY = {
    "efficientnet_b0": EfficientNetB0,
    "efficientnet_b0_eca": lambda **kw: EfficientNetB0(eca=True, **kw),
    "efficientnet_b2": EfficientNetB2,
    "mobilenetv3_small": build_mobilenetv3_small,
    "shufflenetv2_0_5x": build_shufflenetv2_0_5x,
}


def get_model(name: str, num_classes: int = 7, in_channels: int = 1, pretrained: bool = True, eca: bool = False) -> nn.Module:
    if name == "efficientnet_b0_eca":
        return EfficientNetB0(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained, eca=True)
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    builder = _REGISTRY[name]
    if isinstance(builder, type):
        if builder in (EfficientNetB0, EfficientNetB2):
            return builder(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained, eca=eca)
        return builder(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
    return builder(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)


def forward_input(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Wrapper kept for symmetry; models accept raw grayscale tensors."""
    return model(x)


__all__ = ["get_model", "_REGISTRY"]
