from __future__ import annotations

import time

import torch
from torch import nn


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    device,
    image_size: int = 48,
    in_channels: int = 1,
    iterations: int = 1000,
    warmup: int = 100,
    batch_size: int = 1,
) -> dict:
    model = model.eval().to(device)
    x = torch.zeros(batch_size, in_channels, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)

    ms = torch.tensor(timings, dtype=torch.float32).numpy() * 1000.0
    return {
        "mean_ms": float(ms.mean()),
        "std_ms": float(ms.std()),
        "p50_ms": float(np_quantile(ms, 0.5)),
        "p95_ms": float(np_quantile(ms, 0.95)),
        "fps": float(1000.0 / ms.mean()),
        "iterations": iterations,
    }


def np_quantile(arr, q):
    import numpy as np

    return float(np.quantile(arr, q))


def export_onnx(model: nn.Module, path: str, image_size: int = 48, in_channels: int = 1) -> None:
    model.eval()
    dummy = torch.zeros(1, in_channels, image_size, image_size)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["output"], opset_version=13)
