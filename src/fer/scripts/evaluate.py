from __future__ import annotations

import argparse
import json

import torch

from fer.config import Config
from fer.data.loaders import build_loaders
from fer.eval.ensemble import collect_probs, weighted_average
from fer.models.factory import get_model
from fer.training.metrics import compute_metrics
from fer.utils.device import get_device, to_device


def main():
    p = argparse.ArgumentParser(description="Evaluate model(s) on test split")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", nargs="+", required=True, help="Checkpoint path(s)")
    p.add_argument("--weights", nargs="+", type=float, default=None, help="Ensemble weights")
    p.add_argument("--split", default="test")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    device = get_device(cfg.train.device)
    loader = build_loaders(cfg, device, args.split)

    models = [to_device(get_model(cfg.model.name, cfg.model.num_classes, cfg.model.in_channels, False, eca=cfg.model.eca), device) for _ in args.ckpt]
    for m, ckpt in zip(models, args.ckpt):
        m.load_state_dict(torch.load(ckpt, map_location=device))

    probs, labels = collect_probs(models, loader, device)
    if len(models) > 1:
        fused = weighted_average(probs, weights=args.weights)
    else:
        fused = probs[0]
    preds = fused.argmax(axis=1)
    metrics = compute_metrics(labels, preds, cfg.model.num_classes)
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
