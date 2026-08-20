"""Validate 50ep per-seed PrivateTest accuracies and save to results/*.json.

Usage:
    python scripts/validate_50ep.py          # evaluates v2/v3/v4 50ep on PrivateTest
    python scripts/validate_50ep.py --variant v2  # single variant
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from fer.config import Config
from fer.data.datasets import FERDataset
from fer.data.transforms import eval_transform
from fer.models.factory import get_model
from fer.utils.device import get_device, to_device
from fer.eval.ensemble import collect_probs

ROOT = Path(__file__).resolve().parents[1]
device = get_device("auto")

def eval_variant(variant: str):
    run_dir = ROOT / f"runs/efficientnet_b0_{variant}_fer2013_50ep_ls01"
    cfg = Config.from_yaml(run_dir / "config.yaml")
    ds = FERDataset(ROOT / "data/fer2013", "test", transform=eval_transform(48), dataset="fer2013")
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    accs=[]
    for seed in [42,43,44]:
        m = get_model(cfg.model.name, 7, 1, pretrained=False, eca=cfg.model.eca)
        m.load_state_dict(torch.load(run_dir / f"seed_{seed}.pth", map_location="cpu", weights_only=True))
        m = to_device(m, device)
        probs, labels = collect_probs([m], loader, device)
        acc = (probs[0].argmax(1) == labels).mean()
        accs.append(float(acc))
        print(f"  {variant} seed {seed}: {acc*100:.2f}%")
    out = {
        "model": f"efficientnet_b0_{variant}_fer2013_50ep_ls01",
        "config": f"configs/train/efficientnet_b0_{variant}_fer2013_50ep_ls01.yaml",
        "checkpoints": [f"runs/efficientnet_b0_{variant}_fer2013_50ep_ls01/seed_{s}.pth" for s in [42,43,44]],
        "test": {"accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs, ddof=1)), "values": accs}},
        "evaluated": "2026-08-14",
        "device": str(device),
        "validated": True
    }
    out_path = ROOT / f"results/efficientnet_b0_{variant}_fer2013_50ep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  -> {out_path}: {np.mean(accs)*100:.2f}±{np.std(accs, ddof=1)*100:.2f}%")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--variant", choices=["v2","v3","v4"], default=None)
    args=p.parse_args()
    variants=[args.variant] if args.variant else ["v2","v3","v4"]
    for v in variants:
        print(f"{v}:")
        eval_variant(v)

if __name__=="__main__":
    main()
