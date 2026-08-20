"""Regenerate all paper figures/tables from from-scratch 30ep 3-seed checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from fer.config import Config
from fer.data.datasets import FERDataset
from fer.data.transforms import eval_transform
from fer.eval.ensemble import collect_probs, weighted_average
from fer.models.factory import get_model
from fer.training.metrics import compute_metrics
from fer.utils.constants import EMOTION_LABELS
from fer.utils.device import get_device
from fer.viz.gradcam import gradcam_heatmap
from fer.viz.style import apply_style, save_fig

ROOT = Path(__file__).resolve().parents[1]
device = get_device("auto")
apply_style()
out_dir = ROOT / "figures"
out_dir.mkdir(exist_ok=True)
cfg = Config.from_yaml("configs/train/efficientnet_b0_v2_fer2013.yaml")  # for eval_transform

def load_model(variant: str, seed: int):
    ckpt = ROOT / f"runs/efficientnet_b0_{variant}_fer2013/seed_{seed}.pth"
    # Hybrid has different path
    if variant == "hybrid":
        ckpt = ROOT / f"runs/efficientnet_b0_hybrid_fer2013/seed_{seed}.pth"
    m = get_model("efficientnet_b0", 7, 1, False)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return m.to(device).eval()

def make_loader(split: str):
    ds = FERDataset(ROOT / "data/fer2013", split, transform=eval_transform(48), dataset="fer2013")
    return DataLoader(ds, batch_size=128, shuffle=False)

# 1. Confusion matrices for 6 regimes (semantic names, V-tags in Appendix Table A1)
SEMANTIC = {"v1":"Baseline","v2":"Spatial","v3":"Photometric","v4":"Occlusion","v5":"Mixing","hybrid":"Hybrid"}
print("== Confusion matrices ==")
test_loader = make_loader("test")
for variant in ["v1", "v2", "v3", "v4", "v5"]:
    model = load_model(variant, 42)
    y_true, y_pred = [], []
    with torch.no_grad():
        for img, lab in test_loader:
            out = model(img.to(device))
            y_true.extend(lab.numpy())
            y_pred.extend(out.argmax(1).cpu().numpy())
    met = compute_metrics(y_true, y_pred, 7)
    cm = np.array(met["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(EMOTION_LABELS.values()), yticklabels=list(EMOTION_LABELS.values()), ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    save_fig(fig, f"confusion_{variant}", out_dir)
    # Also save semantic symlink name for paper (Overleaf copy, not symlink for foolproof)
    try:
        import shutil
        shutil.copy(out_dir / f"confusion_{variant}.pdf", out_dir / f"confusion_{SEMANTIC[variant].lower()}.pdf")
        shutil.copy(out_dir / f"confusion_{variant}.eps", out_dir / f"confusion_{SEMANTIC[variant].lower()}.eps")
    except: pass

# 2-model Photometric+Occlusion and Spatial+Occlusion ensemble (50ep ensemble SOTA is Spatial+Occlusion)
for name, variants, weights, title in [
    ("v3v4", ["v3", "v4"], [0.4, 0.6], "Photometric+Occlusion (30ep, 69.96%)"),
    ("ensemble", ["v2", "v4"], [0.55, 0.45], "Spatial+Occlusion ensemble (50ep, 72.39% ensemble SOTA)"),
    ("5model", ["v1", "v2", "v3", "v4", "v5"], [0.2, 0.2, 0.2, 0.2, 0.2], "5-model (71.06%)"),
]:
    models = [load_model(v, 42) for v in variants]
    probs, labels = collect_probs(models, test_loader, device)
    fused = weighted_average(probs, weights=weights)
    preds = fused.argmax(1)
    acc = (preds == labels).mean()
    cm = np.zeros((7, 7), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(EMOTION_LABELS.values()), yticklabels=list(EMOTION_LABELS.values()), ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    save_fig(fig, f"confusion_{name}", out_dir)
    print(f"{name} acc {acc:.4f}")
    if name=="ensemble":
        import shutil
        try:
            shutil.copy(out_dir / f"confusion_{name}.pdf", out_dir / f"confusion_ensemble.pdf")
            shutil.copy(out_dir / f"confusion_{name}.eps", out_dir / f"confusion_ensemble.eps")
        except: pass

# 2. Weight sweep combined Fig 3 — both Photometric+Occlusion (30ep) and Spatial+Occlusion (50ep SOTA) side-by-side, large for readability
print("== Weight sweep combined Fig 3 ==")
val_loader = make_loader("validation")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (title, v_a, v_b) in zip(axes, [("Photometric+Occlusion (30ep)", "v3", "v4"), ("Spatial+Occlusion ensemble (50ep SOTA)", "v2", "v4")]):
    models_a = [load_model(v_a, 42)]; models_b = [load_model(v_b, 42)]
    probs_a_val, labels_val = collect_probs(models_a, val_loader, device)
    probs_b_val, _ = collect_probs(models_b, val_loader, device)
    probs_stack_val = np.stack([probs_a_val[0], probs_b_val[0]])
    ws = np.arange(0, 1.0001, 0.05)
    accs = []
    for w in ws:
        fused = weighted_average(probs_stack_val, weights=[w, 1-w])
        accs.append((fused.argmax(1) == labels_val).mean())
    ax.plot(ws, accs, "o-", linewidth=2)
    best_idx = int(np.argmax(accs))
    ax.axvline(ws[best_idx], color="red", linestyle="--", alpha=0.5, label=f"best w={ws[best_idx]:.2f} (val {accs[best_idx]:.4f})")
    ax.set_xlabel(f"w ({SEMANTIC[v_a]}); {SEMANTIC[v_b]}=1-w", fontsize=9)
    ax.set_ylabel("Validation accuracy")
    ax.legend(fontsize=7)
    print(f"{title} best w={ws[best_idx]:.2f} val {accs[best_idx]:.4f}")
plt.tight_layout()
save_fig(fig, "weight_sweep", out_dir)
# Also save individual for legacy
try:
    import shutil
    shutil.copy(out_dir / "weight_sweep.pdf", out_dir / "weight_sweep_v3_v4.pdf")
    shutil.copy(out_dir / "weight_sweep.pdf", out_dir / "weight_sweep_v2_v4.pdf")
except: pass

# 3. Grad-CAM for Spatial and Mixing (the diversity pair described in the paper) — semantic
# Single-row layout: 7 images (one per class, 0..6) with label beneath each.
print("== Grad-CAM ==")
for variant in ["v2", "v5"]:
    m = load_model(variant, 42)
    target = m.model.features[7]
    ds = FERDataset(ROOT / "data/fer2013", "test", transform=eval_transform(48), dataset="fer2013")
    np.random.seed(0)
    by_class = {i: [] for i in range(7)}
    for idx in np.random.permutation(len(ds)):
        if len(by_class[ds.samples[idx][1]]) < 1:
            by_class[ds.samples[idx][1]].append(int(idx))
        if all(len(v) >= 1 for v in by_class.values()):
            break
    fig, axes = plt.subplots(1, 7, figsize=(14, 2.2), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()
    for cls in range(7):
        idx = by_class[cls][0]
        img, _ = ds[idx]
        cam = gradcam_heatmap(m, img.to(device), target, class_idx=cls)
        ax = axes[cls]
        ax.imshow(img.squeeze().cpu().numpy(), cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(EMOTION_LABELS[cls], fontsize=8, labelpad=2)
    plt.tight_layout(pad=0.5, w_pad=0.8)
    fig.subplots_adjust(bottom=0.22)
    save_fig(fig, f"gradcam_{variant}", out_dir, eps=False, tiff=True)
    # Semantic copy for paper (Overleaf) + paper/src/fig for SIVP bundle
    try:
        import shutil
        shutil.copy(out_dir / f"gradcam_{variant}.pdf", out_dir / f"gradcam_{SEMANTIC[variant].lower()}.pdf")
        shutil.copy(out_dir / f"gradcam_{variant}.tiff", out_dir / f"gradcam_{SEMANTIC[variant].lower()}.tiff")
        # also sync to paper/src/fig for make bundle
        src_fig = ROOT / "paper" / "src" / "fig" / f"gradcam_{SEMANTIC[variant].lower()}.pdf"
        src_fig_tiff = ROOT / "paper" / "src" / "fig" / f"gradcam_{SEMANTIC[variant].lower()}.tiff"
        src_fig.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_dir / f"gradcam_{variant}.pdf", src_fig)
        shutil.copy(out_dir / f"gradcam_{variant}.tiff", src_fig_tiff)
        # legacy alias gradcam_spatial for paper
        if variant == "v2":
            shutil.copy(out_dir / f"gradcam_{variant}.pdf", ROOT / "paper" / "src" / "fig" / "gradcam_spatial.pdf")
            shutil.copy(out_dir / f"gradcam_{variant}.tiff", ROOT / "paper" / "src" / "fig" / "gradcam_spatial.tiff")
    except Exception as e:
        print(f"  copy warn: {e}")
    print(f"Grad-CAM {SEMANTIC[variant]} done (1x7, labels beneath)")

print("== Done: figures/ now has from-scratch 67-71% figures ==")
