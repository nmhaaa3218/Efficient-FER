from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLabelKLDivLoss(nn.Module):
    """KL-divergence loss for FERPlus soft (probability) labels."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        log_softmax = F.log_softmax(logits / self.temperature, dim=1)
        target_probs = F.normalize(target_probs, p=1, dim=1)
        return F.kl_div(log_softmax, target_probs, reduction="batchmean") * (self.temperature**2)


class FocalLoss(nn.Module):
    """Focal Loss gamma=2 for class imbalance, optionally with class weights and label smoothing."""

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_criterion(label_mode: str, label_smoothing: float = 0.0, num_classes: int = 7, class_weights: torch.Tensor | None = None, loss_type: str = "ce") -> nn.Module:
    if label_mode == "soft":
        return SoftLabelKLDivLoss()
    if loss_type == "focal":
        return FocalLoss(gamma=2.0, label_smoothing=label_smoothing, weight=class_weights)
    if class_weights is not None:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weights)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
