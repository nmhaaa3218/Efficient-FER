from __future__ import annotations

from typing import Any

import torch
from torch import nn


def compute_flops(model: nn.Module, image_size: int = 48, in_channels: int = 1) -> dict[str, Any]:
    """Parameter count + FLOPs via fvcore."""
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    model.eval()
    dummy = torch.zeros(1, in_channels, image_size, image_size)
    flops = FlopCountAnalysis(model, dummy).total()
    params = sum(p.numel() for p in model.parameters())
    return {"params": params, "params_m": params / 1e6, "gflops": flops / 1e9, "flops": flops}
