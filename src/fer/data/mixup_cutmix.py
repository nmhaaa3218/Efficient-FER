from __future__ import annotations

import numpy as np
import torch


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return x, y, y
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index]


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return x, y, y
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    _, _, h, w = x.shape
    cx, cy = int(w * torch.rand(1)), int(h * torch.rand(1))
    r_w = int(w * np.sqrt(1 - lam))
    r_h = int(h * np.sqrt(1 - lam))
    x1 = max(0, cx - r_w // 2)
    y1 = max(0, cy - r_h // 2)
    x2 = min(w, cx + r_w // 2)
    y2 = min(h, cy + r_h // 2)
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
    return mixed_x, y, y[index]


def mixup_criterion(
    criterion: torch.nn.Module, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float
) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
