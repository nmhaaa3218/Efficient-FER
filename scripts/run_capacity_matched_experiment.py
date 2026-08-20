from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fer.config import Config
from fer.data.loaders import build_loaders
from fer.data.datasets import FERDataset
from fer.models.factory import get_model
from fer.training.losses import build_criterion
from fer.training.trainer import Trainer
from fer.utils.device import get_device, to_device
from fer.utils import set_seed


def evaluate_dataset(model: torch.nn.Module, dataset_split: str, cfg: Config, device: torch.device):
    """Evaluate model on a specific split (validation or test) and return predictions & metrics."""
    model.eval()
    loader = build_loaders(cfg, device, split=dataset_split)

    all_preds = []
    all_targets = []
    all_logits = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
            all_logits.append(logits.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_logits = np.concatenate(all_logits, axis=0)

    acc = float(np.mean(all_preds == all_targets))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro"))
    bal_acc = float(balanced_accuracy_score(all_targets, all_preds))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_acc": bal_acc,
        "predictions": all_preds.tolist(),
        "targets": all_targets.tolist(),
        "logits": all_logits,
    }


def compute_bootstrap_ci(y_true, preds_ens, preds_b2, n_boot=2000, seed=42):
    """Paired percentile bootstrap 95% CI of the difference (ens - B2)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        acc_e = np.mean(preds_ens[idx] == y_true[idx])
        acc_b = np.mean(preds_b2[idx] == y_true[idx])
        diffs.append(acc_e - acc_b)
    diffs = np.array(diffs) * 100.0
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    config_path = "configs/train/efficientnet_b2_v4_fer2013_50ep_ls01.yaml"
    cfg = Config.from_yaml(config_path)
    device = get_device("auto")
    print(f"Using device: {device}")

    out_root = Path(cfg.train.output_dir) / "efficientnet_b2_v4_fer2013_50ep_ls01"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.yaml").write_text(Path(config_path).read_text())

    criterion = build_criterion(cfg.data.label_mode, cfg.train.label_smoothing, cfg.model.num_classes)
    trainer = Trainer(
        cfg.train,
        device,
        criterion,
        cfg.model.num_classes,
        aug=cfg.aug,
        label_mode=cfg.data.label_mode,
    )

    per_seed_results = {}
    seeds = [42, 43, 44]

    for seed in seeds:
        ckpt_path = out_root / f"seed_{seed}.pth"
        if not ckpt_path.exists():
            print(f"\n==========================================")
            print(f"Training EfficientNet-B2 on FER-2013 | Seed {seed}")
            print(f"==========================================")
            set_seed(seed)
            model = to_device(get_model(cfg.model.name, cfg.model.num_classes, cfg.model.in_channels, cfg.model.pretrained), device)
            train_loader = build_loaders(cfg, device, "train")
            valid_loader = build_loaders(cfg, device, "validation")

            history = trainer.train(model, train_loader, valid_loader, lr=cfg.train.lr)
            torch.save(model.state_dict(), ckpt_path)
            (out_root / f"seed_{seed}_history.json").write_text(json.dumps(history))
        else:
            print(f"Found existing checkpoint: {ckpt_path}")
            model = to_device(get_model(cfg.model.name, cfg.model.num_classes, cfg.model.in_channels, cfg.model.pretrained), device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Evaluate on PublicTest (validation) and PrivateTest (test)
        val_res = evaluate_dataset(model, "validation", cfg, device)
        test_res = evaluate_dataset(model, "test", cfg, device)

        per_seed_results[seed] = {
            "val_acc": val_res["accuracy"],
            "val_macro_f1": val_res["macro_f1"],
            "val_bal_acc": val_res["balanced_acc"],
            "test_acc": test_res["accuracy"],
            "test_macro_f1": test_res["macro_f1"],
            "test_bal_acc": test_res["balanced_acc"],
            "test_preds": test_res["predictions"],
            "test_targets": test_res["targets"],
        }
        print(f"Seed {seed} -> PrivateTest Acc: {test_res['accuracy']*100:.2f}%, F1: {test_res['macro_f1']*100:.2f}%, Bal Acc: {test_res['balanced_acc']*100:.2f}%")

    test_accs = [per_seed_results[s]["test_acc"] * 100 for s in seeds]
    test_f1s = [per_seed_results[s]["test_macro_f1"] * 100 for s in seeds]
    test_bals = [per_seed_results[s]["test_bal_acc"] * 100 for s in seeds]

    mean_acc = float(np.mean(test_accs))
    std_acc = float(np.std(test_accs, ddof=1))
    mean_f1 = float(np.mean(test_f1s))
    std_f1 = float(np.std(test_f1s, ddof=1))
    mean_bal = float(np.mean(test_bals))
    std_bal = float(np.std(test_bals, ddof=1))

    print(f"\n==========================================")
    print(f"Capacity-Matched EfficientNet-B2 Results (3 seeds):")
    print(f"PrivateTest Accuracy:  {mean_acc:.2f} +/- {std_acc:.2f}%")
    print(f"PrivateTest Macro F1:  {mean_f1:.2f} +/- {std_f1:.2f}%")
    print(f"PrivateTest Bal Acc:   {mean_bal:.2f} +/- {std_bal:.2f}%")
    print(f"==========================================")

    # Compare with our primary 2-model ensemble (Spatial + S-Oc, 50ep, equal fusion alpha=0.5)
    # Ensemble seeds: 42 -> 69.657%, 43 -> 71.134%, 44 -> 69.574% (mean: 70.122 +/- 0.878%)
    ens_stats_path = Path("results/ensemble_stats.json")
    if ens_stats_path.exists():
        with open(ens_stats_path) as f:
            ens_stats = json.load(f)
        ens_accs = [ens_stats[str(s)]["ens_acc"] * 100 for s in seeds]
        paired_gains = [ens_accs[i] - test_accs[i] for i in range(len(seeds))]
        mean_paired_gain = float(np.mean(paired_gains))
        std_paired_gain = float(np.std(paired_gains, ddof=1))
        print(f"2-Model Ensemble vs B2 Capacity-Matched Paired Gain: +{mean_paired_gain:.2f} +/- {std_paired_gain:.2f} pt")
    else:
        mean_paired_gain = None
        std_paired_gain = None

    summary = {
        "model": "efficientnet_b2_1ch",
        "params_m": 7.71,
        "gflops": 0.0426,
        "augmentation": "v4 (Spatial-Occlusion: CRP + RandomErasing)",
        "epochs": 50,
        "label_smoothing": 0.1,
        "seeds": seeds,
        "per_seed": {
            str(s): {
                "val_acc": per_seed_results[s]["val_acc"],
                "test_acc": per_seed_results[s]["test_acc"],
                "test_macro_f1": per_seed_results[s]["test_macro_f1"],
                "test_bal_acc": per_seed_results[s]["test_bal_acc"],
            }
            for s in seeds
        },
        "aggregate": {
            "mean_test_acc": mean_acc,
            "std_test_acc": std_acc,
            "mean_test_f1": mean_f1,
            "std_test_f1": std_f1,
            "mean_test_bal_acc": mean_bal,
            "std_test_bal_acc": std_bal,
        },
        "comparison_vs_2model_ensemble": {
            "ensemble_mean_acc": 70.1217,
            "ensemble_std_acc": 0.8782,
            "mean_paired_gain_pt": mean_paired_gain,
            "std_paired_gain_pt": std_paired_gain,
        },
    }

    results_out = Path("results/efficientnet_b2_capacity_matched.json")
    results_out.write_text(json.dumps(summary, indent=2))
    print(f"Saved capacity-matched results to {results_out}")


if __name__ == "__main__":
    main()
