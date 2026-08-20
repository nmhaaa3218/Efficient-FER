#!/usr/bin/env python
"""RAF-DB per-class precision/recall/F1 by regime + 6-net variant-mean v2+v4.

Reproducible from existing checkpoints at
  runs/efficientnet_b0_{v1,v2,v4}_rafdb_50ep_ls01/seed_{42,43,44}.pth

Writes:
  results/rafdb_per_class.json   (per-class P/R/F1 by regime, 3-seed mean ± std;
                                  6-net variant-mean v2+v4 per-class point estimate)
  results/rafdb_per_class.md     (compact Markdown table for paper §6)

Uses fer.training.metrics.compute_metrics which already returns
per_class_precision/recall/f1 (per_class_* keys).

Usage:
  python scripts/rafdb_per_class_eval.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fer.data.datasets import FERDataset
from fer.data.transforms import eval_transform
from fer.eval.ensemble import collect_probs
from fer.models.factory import get_model
from fer.training.metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

VARIANTS = ["v1", "v2", "v4"]
SEEDS = (42, 43, 44)
EMOTION = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def load_logits_probs(variant: str, seed: int, device, loader):
    """Return softmax probabilities for a (variant, seed) on RAF-DB test."""
    sd = torch.load(RUNS / f"efficientnet_b0_{variant}_rafdb_50ep_ls01/seed_{seed}.pth",
                     map_location="cpu", weights_only=False)
    sd = sd["model"] if "model" in sd else sd
    m = get_model("efficientnet_b0", 7, 1, False, eca=False).to(device).eval()
    m.load_state_dict(sd)
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(torch.softmax(m(x.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out)


def per_class_summary(per_seed_pc: list) -> dict:
    """Aggregate per-seed per-class metrics into mean ± std per class."""
    out = {}
    for cls in EMOTION:
        cls_i = EMOTION.index(cls)
        p = np.array([d["per_class_precision"][cls_i] for d in per_seed_pc])
        r = np.array([d["per_class_recall"][cls_i]    for d in per_seed_pc])
        f = np.array([d["per_class_f1"][cls_i]        for d in per_seed_pc])
        # per_class support = confusion matrix diagonal of true class
        per_class_n = [d["confusion_matrix"][cls_i][cls_i] for d in per_seed_pc]
        out[cls] = {
            "precision_mean": float(p.mean()),
            "precision_std":  float(p.std(ddof=1)),
            "recall_mean":    float(r.mean()),
            "recall_std":     float(r.std(ddof=1)),
            "f1_mean":        float(f.mean()),
            "f1_std":         float(f.std(ddof=1)),
            "support_per_seed": per_class_n,
            "support_mean":   float(np.mean(per_class_n)),
        }
    return out


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available() else "cpu")
    ds = FERDataset(ROOT / "data" / "rafdb", "test",
                    transform=eval_transform(48), dataset="rafdb")
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    labels = np.array([ds[i][1] for i in range(len(ds))])
    print(f"RAF-DB test: {len(labels)} samples, 7 classes")

    # Cache softmax probs once per (variant, seed)
    print("Loading softmax probabilities...")
    probs = {(v, s): load_logits_probs(v, s, device, loader)
             for v in VARIANTS for s in SEEDS}

    # Per-class metrics per variant per seed
    out = {
        "note": "RAF-DB per-class P/R/F1 by regime. 3-seed mean ± std; "
                "6-net variant-mean v2+v4 is a single point estimate "
                "(3-seed avg softmax probs per variant, then α=0.5 weighted "
                "average, then argmax). 7 classes correspond to the FER-2013 "
                "label space; contempt in RAF-DB is mapped to class 6 (Neutral) "
                "by src/fer/data/rafdb.py per the paper protocol.",
        "split": "test",
        "n_samples": int(len(labels)),
        "device": str(device),
        "per_class_by_regime": {},
        "variant_mean_v2_v4_6net_per_class": None,
        "variant_mean_v2_v4_6net_overall": None,
    }

    for v in VARIANTS:
        per_seed_pc = [compute_metrics(labels, probs[(v, s)].argmax(1), 7) for s in SEEDS]
        out["per_class_by_regime"][v] = {
            "per_seed_overall": [
                {
                    "accuracy": float(compute_metrics(labels, probs[(v, s)].argmax(1), 7)["accuracy"]),
                    "macro_f1": float(compute_metrics(labels, probs[(v, s)].argmax(1), 7)["macro_f1"]),
                } for s in SEEDS
            ],
            "per_class_aggregate": per_class_summary(per_seed_pc),
        }

    # 6-net variant-mean v2+v4: average each variant's softmax across 3 seeds,
    # then α=0.5 weighted average, then per-class metrics.
    vm_v2 = np.stack([probs[("v2", s)] for s in SEEDS]).mean(axis=0)
    vm_v4 = np.stack([probs[("v4", s)] for s in SEEDS]).mean(axis=0)
    ens = 0.5 * vm_v2 + 0.5 * vm_v4
    ens_pred = ens.argmax(1)
    overall = compute_metrics(labels, ens_pred, 7)
    out["variant_mean_v2_v4_6net_overall"] = {
        "accuracy": overall["accuracy"],
        "macro_f1": overall["macro_f1"],
        "balanced_accuracy": overall["balanced_accuracy"],
    }
    pc = {}
    for cls_i, cls in enumerate(EMOTION):
        # support = true positives of this class
        n = int(overall["confusion_matrix"][cls_i][cls_i])
        # precision/recall/f1 are scalars across the whole prediction
        p = float(overall["per_class_precision"][cls_i])
        r = float(overall["per_class_recall"][cls_i])
        f = float(overall["per_class_f1"][cls_i])
        pc[cls] = {"precision": p, "recall": r, "f1": f, "support": n}
    out["variant_mean_v2_v4_6net_per_class"] = pc

    RESULTS.joinpath("rafdb_per_class.json").write_text(json.dumps(out, indent=2))
    print(f"wrote results/rafdb_per_class.json")

    # Markdown table
    md = [f"# RAF-DB per-class metrics (test n={len(labels)})",
          f"\nGenerated by `scripts/rafdb_per_class_eval.py`.\n",
          "## Per-class 3-seed mean ± std for each regime\n",
          "| Class | Support (s42/s43/s44) | Regime | Precision | Recall | F1 |",
          "|---|---|---|---:|---:|---:|"]
    for v in VARIANTS:
        agg = out["per_class_by_regime"][v]["per_class_aggregate"]
        for cls in EMOTION:
            d = agg[cls]
            md.append(f"| {cls} | {d['support_per_seed'][0]}/{d['support_per_seed'][1]}/{d['support_per_seed'][2]} | {v} | "
                      f"{d['precision_mean']*100:.2f} ± {d['precision_std']*100:.2f} | "
                      f"{d['recall_mean']*100:.2f} ± {d['recall_std']*100:.2f} | "
                      f"{d['f1_mean']*100:.2f} ± {d['f1_std']*100:.2f} |")

    md.append("\n## 6-net variant-mean v2+v4 per-class point estimate\n")
    md.append(f"Overall: accuracy {out['variant_mean_v2_v4_6net_overall']['accuracy']*100:.2f}%, "
              f"macro F1 {out['variant_mean_v2_v4_6net_overall']['macro_f1']*100:.2f}%, "
              f"balanced acc {out['variant_mean_v2_v4_6net_overall']['balanced_accuracy']*100:.2f}%.\n")
    md.append("\n| Class | Support | Precision | Recall | F1 |\n|---|---:|---:|---:|---:|")
    for cls in EMOTION:
        d = out["variant_mean_v2_v4_6net_per_class"][cls]
        md.append(f"| {cls} | {d['support']} | {d['precision']*100:.2f}% | "
                  f"{d['recall']*100:.2f}% | {d['f1']*100:.2f}% |")

    RESULTS.joinpath("rafdb_per_class.md").write_text("\n".join(md) + "\n")
    print(f"wrote results/rafdb_per_class.md")

    # Print quick variant-mean summary
    print(f"\n=== 6-net variant-mean v2+v4 (overall) ===")
    print(f"  accuracy={out['variant_mean_v2_v4_6net_overall']['accuracy']*100:.2f}%  "
          f"macro_f1={out['variant_mean_v2_v4_6net_overall']['macro_f1']*100:.2f}%  "
          f"balanced={out['variant_mean_v2_v4_6net_overall']['balanced_accuracy']*100:.2f}%")
    print(f"\n=== Per-class (variant-mean v2+v4) ===")
    print(f"{'Class':<11} {'N':>5} {'P':>7} {'R':>7} {'F1':>7}")
    for cls in EMOTION:
        d = out["variant_mean_v2_v4_6net_per_class"][cls]
        print(f"  {cls:<10} {d['support']:>5} {d['precision']*100:>6.2f}% "
              f"{d['recall']*100:>6.2f}% {d['f1']*100:>6.2f}%")


if __name__ == "__main__":
    main()
