from __future__ import annotations

import json

import numpy as np

from fer.eval.ensemble import collect_probs, weighted_average
from fer.training.metrics import compute_metrics


def weight_sweep(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    metric: str = "accuracy",
    step: float = 0.1,
) -> dict:
    """Grid sweep weight_a in {0.0..1.0} step, w_b = 1 - w_a. Returns results dict."""
    results = []
    w = np.arange(0.0, 1.0 + 1e-9, step)
    for wa in w:
        fused = weighted_average(probs, weights=[wa, 1.0 - wa])
        preds = fused.argmax(axis=1)
        m = compute_metrics(labels, preds, num_classes=num_classes)
        results.append({"w_a": float(wa), "w_b": float(1.0 - wa), **{k: v for k, v in m.items() if k != "confusion_matrix"}})
    best = max(results, key=lambda r: r[metric])
    return {"results": results, "best": best}


def save_sweep(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
