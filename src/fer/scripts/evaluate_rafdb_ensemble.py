"""RAF-DB cross-dataset evaluation — Phase C (#72).

Evaluates 9 RAF-DB ckpts (v1/v2/v4 × 3 seeds) on RAF-DB test split.
Computes:
- Per-seed single-model test acc + macro F1 + balanced acc
- Per-seed Spatial+Occlusion ensemble (v2+v4, α=0.5) + mean±SD across seeds
- Variant-mean ensemble (avg 3 seeds per variant, then weighted avg α=0.5)

Reads:  runs/efficientnet_b0_{v1,v2,v4}_rafdb_50ep_ls01/{config.yaml, seed_{42,43,44}.pth}
Writes: results/ensemble_rafdb_v2v4_test.json

Usage:
    python -m fer.scripts.evaluate_rafdb_ensemble
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fer.config import Config
from fer.data.datasets import FERDataset
from fer.data.transforms import eval_transform
from fer.eval.ensemble import collect_probs, weighted_average
from fer.models.factory import get_model
from fer.training.metrics import compute_metrics
from fer.utils.device import get_device, to_device


VARIANTS = ["v1", "v2", "v4"]
SEEDS = [42, 43, 44]
RUNS_DIR = Path("runs")
RESULTS_DIR = Path("results")


def load_model(variant: str, seed: int, device):
    """Load RAF-DB ckpt + return (model_on_device, cfg)."""
    run_dir = RUNS_DIR / f"efficientnet_b0_{variant}_rafdb_50ep_ls01"
    cfg = Config.from_yaml(run_dir / "config.yaml")
    model = get_model(cfg.model.name, cfg.model.num_classes, cfg.model.in_channels, pretrained=False, eca=cfg.model.eca)
    sd = torch.load(run_dir / f"seed_{seed}.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    return to_device(model, device), cfg


def main():
    device = get_device("auto")
    print(f"Device: {device}")

    # All 3 RAF-DB configs share identical data block — use v1 as reference
    cfg_ref = Config.from_yaml(RUNS_DIR / "efficientnet_b0_v1_rafdb_50ep_ls01" / "config.yaml")
    ds = FERDataset(
        root=Path(cfg_ref.data.root) / cfg_ref.data.name,
        split="test",
        label_mode=cfg_ref.data.label_mode,
        transform=eval_transform(cfg_ref.data.image_size),
        dataset=cfg_ref.data.name,
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    print(f"Test set: {len(ds)} samples, classes=0..6")

    # Load all models + collect probs (one model at a time — collect_probs is for ensembles)
    all_probs: dict[str, np.ndarray] = {}
    test_labels: np.ndarray | None = None
    for v in VARIANTS:
        for s in SEEDS:
            model, _ = load_model(v, s, device)
            probs, labels = collect_probs([model], loader, device)
            all_probs[f"{v}_{s}"] = probs[0]
            if test_labels is None:
                test_labels = labels
            print(f"  loaded {v} seed{s}: probs {probs[0].shape}")

    # Per-seed single-model metrics
    per_seed = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        for s in SEEDS:
            preds = all_probs[f"{v}_{s}"].argmax(axis=1)
            m = compute_metrics(test_labels, preds, 7)
            per_seed[v][f"seed_{s}"] = {
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "balanced_accuracy": m["balanced_accuracy"],
            }
            print(f"  {v} seed{s}: acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")

    # Per-seed Spatial+Occlusion ensemble (v2+v4, α=0.5)
    ensemble_per_seed = {}
    pair_accs: list[float] = []
    for s in SEEDS:
        fused = weighted_average(
            np.stack([all_probs[f"v2_{s}"], all_probs[f"v4_{s}"]]),
            weights=[0.5, 0.5],
        )
        preds = fused.argmax(axis=1)
        m = compute_metrics(test_labels, preds, 7)
        ensemble_per_seed[f"seed_{s}"] = {
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "balanced_accuracy": m["balanced_accuracy"],
        }
        pair_accs.append(m["accuracy"])
        print(f"  v2+v4 seed{s} α=0.5: acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")

    # Variant-mean ensemble (avg 3 seeds per variant, then weighted avg α=0.5)
    p_v2_mean = np.mean(np.stack([all_probs[f"v2_{s}"] for s in SEEDS]), axis=0)
    p_v4_mean = np.mean(np.stack([all_probs[f"v4_{s}"] for s in SEEDS]), axis=0)
    fused_mean = weighted_average(
        np.stack([p_v2_mean, p_v4_mean]),
        weights=[0.5, 0.5],
    )
    m_mean = compute_metrics(test_labels, fused_mean.argmax(axis=1), 7)

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "note": "RAF-DB cross-dataset eval (#72). 9 ckpts trained 50ep ls=0.1 ×3 seeds on RAF-DB 48x48 1ch (no tuning).",
        "test_size": int(len(ds)),
        "per_seed_single": per_seed,
        "per_seed_summary": {
            v: {
                "accuracy_mean": float(np.mean([per_seed[v][f"seed_{s}"]["accuracy"] for s in SEEDS])),
                "accuracy_std": float(np.std([per_seed[v][f"seed_{s}"]["accuracy"] for s in SEEDS])),
                "macro_f1_mean": float(np.mean([per_seed[v][f"seed_{s}"]["macro_f1"] for s in SEEDS])),
            }
            for v in VARIANTS
        },
        "ensemble_v2_v4_alpha05_per_seed": ensemble_per_seed,
        "ensemble_v2_v4_alpha05_summary": {
            "accuracy_mean": float(np.mean(pair_accs)),
            "accuracy_std": float(np.std(pair_accs)),
            "macro_f1_mean": float(np.mean([ensemble_per_seed[f"seed_{s}"]["macro_f1"] for s in SEEDS])),
        },
        "ensemble_v2_v4_alpha05_variant_mean": {
            "method": "avg 3 seeds per variant (softmax), then weighted avg α=0.5",
            "accuracy": m_mean["accuracy"],
            "macro_f1": m_mean["macro_f1"],
            "balanced_accuracy": m_mean["balanced_accuracy"],
        },
    }
    out_path = RESULTS_DIR / "ensemble_rafdb_v2v4_test.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()