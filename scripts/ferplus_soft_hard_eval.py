#!/usr/bin/env python
"""Compute FERPlus soft vs hard macro-F1 on FERPlus test split.

Reproduces paper §5 Table 4 values for Spatial and Spatial-Occlusion:
  Spatial hard 71.05 / soft 71.65
  S-Oc    hard 70.17 / soft 71.53

Uses folder-derived crowd-max-vote labels (default FERDataset hard mode),
matching the labels used in results/efficientnet_b0_v{2,4}_ferplus*.json.

Usage:
  python scripts/ferplus_soft_hard_eval.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from fer.data.datasets import FERDataset
from fer.models.factory import get_model
from fer.training.metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
TFM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def eval_ferplus(dirname: str, seeds=(42, 43, 44)):
    """Return per-seed acc and macro-F1 on FERPlus test (folder labels)."""
    accs, f1s, bas = [], [], []
    for s in seeds:
        ckpt = ROOT / "runs" / dirname / f"seed_{s}.pth"
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd["model"] if "model" in sd else sd
        m = get_model("efficientnet_b0", 7, 1, False, eca=False).to(device).eval()
        m.load_state_dict(sd)
        ds = FERDataset(ROOT / "data" / "ferplus", "test",
                        transform=TFM, dataset="ferplus")
        dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
        preds, labs = [], []
        with torch.no_grad():
            for x, y in dl:
                preds.append(m(x.to(device)).argmax(1).cpu().numpy())
                labs.append(y.numpy())
        pred = np.concatenate(preds)
        lab = np.concatenate(labs)
        m_metrics = compute_metrics(lab.tolist(), pred.tolist(), 7)
        accs.append(m_metrics["accuracy"])
        f1s.append(m_metrics["macro_f1"])
        bas.append(m_metrics["balanced_accuracy"])
    return accs, f1s, bas


CONFIGS = {
    "v2_hard": "efficientnet_b0_v2_ferplus",
    "v2_soft": "efficientnet_b0_v2_ferplus_soft",
    "v4_hard": "efficientnet_b0_v4_ferplus",
    "v4_soft": "efficientnet_b0_v4_ferplus_soft",
}

results = {"note": "FERPlus test split, folder-derived crowd-max-vote labels "
                   "(matches results/efficientnet_b0_v{2,4}_ferplus*.json setup). "
                   "Reproduces paper §5 Table 4 macro-F1 values."}

for tag, dirname in CONFIGS.items():
    a, f, b = eval_ferplus(dirname)
    results[tag] = {
        "per_seed_acc": a,
        "mean_acc": float(np.mean(a)),
        "std_acc": float(np.std(a, ddof=1)),
        "per_seed_macro_f1": f,
        "mean_macro_f1": float(np.mean(f)),
        "per_seed_balanced_acc": b,
    }

ROOT.joinpath("results").mkdir(exist_ok=True)
(ROOT / "results" / "ferplus_soft_hard_eval.json").write_text(
    json.dumps(results, indent=2)
)
print("wrote results/ferplus_soft_hard_eval.json")
for tag in CONFIGS:
    r = results[tag]
    print(f"  {tag}: acc={r['mean_acc']*100:.2f}%, macro_f1={r['mean_macro_f1']*100:.2f}%")
print()
print("Paper §5 Table 4 claim:")
print("  Spatial hard macro_f1: 71.05  | soft: 71.65")
print("  S-Oc    hard macro_f1: 70.17  | soft: 71.53")
