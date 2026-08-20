"""Generate paper figures from legacy V2/V5 checkpoints.

Outputs (all under figures/):
  - confusion_v2.png, confusion_v5.png, confusion_ensemble.png
  - gradcam_v2.png, gradcam_v5.png    (per-class examples)
  - weight_sweep.png                   (ensemble acc vs w1)
  - weight_sweep.json                  (raw sweep numbers)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fer.config import Config
from fer.data.datasets import FERDataset
from fer.eval.ensemble import collect_probs, weighted_average
from fer.eval.weight_sweep import weight_sweep
from fer.models.factory import get_model
from fer.training.metrics import compute_metrics
from fer.utils.constants import EMOTION_LABELS
from fer.utils.device import get_device
from fer.viz.confusion import confusion_matrix_figure
from fer.viz.gradcam import gradcam_heatmap
from fer.viz.style import apply_style, save_fig
from fer.data.transforms import eval_transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LEGACY = ROOT / "plan" / "CV-Facial-Expression-main-legacy-proj"
V2_CKPT = LEGACY / "model_efficientnetb0_v2_tuned_2.pth"
V5_CKPT = LEGACY / "model_efficientnetb0_v5_tuned.pth"


def load_legacy_model(ckpt_path: Path, device) -> torch.nn.Module:
    m = get_model("efficientnet_b0", num_classes=7, in_channels=1, pretrained=False)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    return m.to(device).eval()


def make_loader(cfg: Config, split: str, device) -> DataLoader:
    ds = FERDataset(
        root=Path(cfg.data.root) / cfg.data.name,
        split=split,
        label_mode="hard",
        transform=eval_transform(cfg.data.image_size),
        dataset=cfg.data.name,
    )
    return DataLoader(ds, batch_size=128, shuffle=False)


def main() -> None:
    apply_style()
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    device = get_device("mps")
    cfg = Config()
    cfg.data.name = "fer2013"

    print("== Loading legacy V2 + V5 weights ==")
    m_v2 = load_legacy_model(V2_CKPT, device)
    m_v5 = load_legacy_model(V5_CKPT, device)

    test_loader = make_loader(cfg, "test", device)

    # ---------- 1. Confusion matrices ----------
    print("== Confusion matrices (test) ==")
    for name, model, w in [("v2", m_v2, 1.0), ("v5", m_v5, 1.0), ("ensemble", None, 0.45)]:
        if name == "ensemble":
            probs, labels = collect_probs([m_v2, m_v5], test_loader, device)
            fused = weighted_average(probs, weights=[w, 1 - w])
            preds = fused.argmax(axis=1)
        else:
            preds, labels = [], []
            with torch.no_grad():
                for img, lab in test_loader:
                    p = model(img.to(device)).argmax(1).cpu().numpy()
                    preds.extend(p)
                    labels.extend(lab.numpy())
            preds = np.asarray(preds)
            labels = np.asarray(labels)
        m = compute_metrics(labels, preds, 7)
        print(
            f"  {name:10s} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
            f"bal_acc={m['balanced_accuracy']:.4f}"
        )
        cm = np.asarray(m["confusion_matrix"])
        title = f"Confusion: {name.upper()} (test, acc={m['accuracy']:.3f})"
        fig, ax = plt.subplots(figsize=(7, 6))
        import seaborn as sns

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(EMOTION_LABELS.values()),
            yticklabels=list(EMOTION_LABELS.values()),
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        save_fig(fig, f"confusion_{name}", out)

    # ---------- 2. Weight sensitivity sweep (validation) ----------
    print("== Weight sensitivity sweep (validation) ==")
    val_loader = make_loader(cfg, "validation", device)
    probs_val, labels_val = collect_probs([m_v2, m_v5], val_loader, device)
    sweep = weight_sweep(probs_val, labels_val, num_classes=7, metric="balanced_accuracy")
    accs = [r["balanced_accuracy"] for r in sweep["results"]]
    ws = [r["w_a"] for r in sweep["results"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ws, accs, "o-", linewidth=2)
    ax.axvline(sweep["best"]["w_a"], color="red", linestyle="--", alpha=0.5,
               label=f"best w_a={sweep['best']['w_a']:.2f} (bal_acc={sweep['best']['balanced_accuracy']:.4f})")
    ax.set_xlabel("w_a (V2 weight); w_b = 1 - w_a (V5 weight)")
    ax.set_ylabel("Validation Balanced Accuracy")
    ax.set_title("Ensemble weight sensitivity (V2 spatial + V5 mixing)")
    ax.legend()
    save_fig(fig, "weight_sweep", out)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "weight_sweep.json").write_text(json.dumps(sweep, indent=2))
    print(f"  best w_a = {sweep['best']['w_a']:.2f}, val bal_acc = {sweep['best']['balanced_accuracy']:.4f}")

    # ---------- 3. Grad-CAM heatmaps ----------
    print("== Grad-CAM heatmaps ==")
    target_layer = m_v2.model.features[7]  # last MBConv block, rich spatial features

    ds = FERDataset(
        root=Path(cfg.data.root) / cfg.data.name,
        split="test",
        label_mode="hard",
        transform=eval_transform(cfg.data.image_size),
        dataset=cfg.data.name,
    )
    # pick 1 sample per class, prefer correctly-classified V2 examples for clarity
    np.random.seed(0)
    by_class: dict[int, list[int]] = {i: [] for i in range(7)}
    for idx in np.random.permutation(len(ds)):
        if len(by_class[ds.samples[idx][1]]) < 1:
            by_class[ds.samples[idx][1]].append(int(idx))
        if all(len(v) >= 1 for v in by_class.values()):
            break

    for variant_name, model in [("v2", m_v2), ("v5", m_v5)]:
        target = model.model.features[7]
        fig, axes = plt.subplots(2, 8, figsize=(20, 6))
        for cls in range(7):
            row = cls // 4
            col_in = (cls % 4) * 2
            if not by_class[cls]:
                continue
            idx = by_class[cls][0]
            img, _ = ds[idx]
            cam = gradcam_heatmap(model, img.to(device), target, class_idx=cls)
            ax0 = axes[row, col_in]
            ax0.imshow(img.squeeze().cpu().numpy(), cmap="gray")
            ax0.set_title(f"{EMOTION_LABELS[cls]}", fontsize=9)
            ax0.axis("off")
            ax1 = axes[row, col_in + 1]
            ax1.imshow(img.squeeze().cpu().numpy(), cmap="gray", alpha=0.5)
            ax1.imshow(cam, cmap="jet", alpha=0.4)
            ax1.set_title(f"Grad-CAM (V{variant_name[1]})", fontsize=9)
            ax1.axis("off")
        # hide last empty col
        axes[0, 7].axis("off")
        axes[1, 7].axis("off")
        fig.suptitle(f"Grad-CAM attention: EfficientNet-B0 {variant_name.upper()}")
        save_fig(fig, f"gradcam_{variant_name}", out)

    print("== Done. All figures saved to figures/ ==")


if __name__ == "__main__":
    main()
