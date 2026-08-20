from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_probs(models: list[nn.Module], loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities per model [M, N, C], labels [N])."""
    probs_list, all_labels = [], []
    for model in models:
        model.eval()
        model_probs = []
        for images, batch_labels in loader:
            images = images.to(device)
            probs = F.softmax(model(images), dim=1).cpu().numpy()
            model_probs.append(probs)
            if model is models[0]:
                all_labels.extend(batch_labels.cpu().numpy().reshape(-1))
        probs_list.append(np.concatenate(model_probs))
    return np.stack(probs_list), np.asarray(all_labels)


def weighted_average(probs: np.ndarray, weights: list[float] | None = None) -> np.ndarray:
    """Weighted average of softmax probabilities. probs: [M, N, C]."""
    weights = np.asarray(weights if weights is not None else [1.0 / len(probs)] * len(probs))
    weights = weights / weights.sum()
    return np.tensordot(weights, probs, axes=1)


def reciprocal_rank_fusion(probs: np.ndarray, k: int = 60, weights: list[float] | None = None) -> np.ndarray:
    """RRF over per-model softmax scores. probs: [M, N, C]."""
    m, n, c = probs.shape
    if weights is None:
        weights = np.ones(m)
    weights = np.asarray(weights) / np.sum(weights)
    fused = np.zeros((n, c))
    for i in range(m):
        order = np.argsort(-probs[i], axis=1)
        ranks = np.empty_like(order)
        for j in range(n):
            ranks[j, order[j]] = np.arange(1, c + 1)
        fused += weights[i] * 1.0 / (k + ranks)
    return fused / fused.sum(axis=1, keepdims=True)


def predict_from_probs(fused: np.ndarray) -> np.ndarray:
    return fused.argmax(axis=1)


def fuse(models: list[nn.Module], loader: DataLoader, device, method: str = "weighted_avg", weights=None, k: int = 60) -> np.ndarray:
    probs, labels = collect_probs(models, loader, device)
    if method == "rrf":
        return reciprocal_rank_fusion(probs, k=k, weights=weights)
    return weighted_average(probs, weights=weights)
