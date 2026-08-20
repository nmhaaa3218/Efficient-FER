from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fer.config import Config
from fer.data.loaders import build_loaders
from fer.models.factory import get_model
from fer.training.logging import TBLogger
from fer.training.losses import build_criterion
from fer.training.trainer import Trainer
from fer.utils.device import get_device, to_device
from fer.utils import set_seed


def main():
    p = argparse.ArgumentParser(description="Train FER model")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--seeds", type=int, default=None, help="Override num_seeds")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--tag", default="", help="Extra output tag")
    p.add_argument("--no-tb", action="store_true", help="Disable TensorBoard logging")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.seeds: cfg.train.num_seeds = args.seeds
    if args.epochs: cfg.train.epochs = args.epochs
    if args.lr: cfg.train.lr = args.lr
    if args.batch_size: cfg.train.batch_size = args.batch_size

    device = get_device(cfg.train.device)
    # Class-weighted loss for imbalance (1/sqrt(count) per class)
    class_weights = None
    if cfg.train.class_weighted_loss:
        from collections import Counter
        from fer.data.datasets import FERDataset as _DS

        ds_tmp = _DS(Path(cfg.data.root) / cfg.data.name, "train", dataset=cfg.data.name)
        counts = Counter(label for _, label in ds_tmp.samples)
        total = len(ds_tmp)
        # Weight = sqrt(total / count) or 1/sqrt(count) — use 1/sqrt for stability
        weights = []
        for i in range(cfg.model.num_classes):
            cnt = counts.get(i, 1)
            weights.append(1.0 / (cnt**0.5))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"Class weights (1/sqrt(count)): {weights}")

    criterion = build_criterion(cfg.data.label_mode, cfg.train.label_smoothing, cfg.model.num_classes, class_weights=class_weights, loss_type=getattr(cfg.train, "loss_type", "ce"))
    trainer = Trainer(
        cfg.train,
        device,
        criterion,
        cfg.model.num_classes,
        aug=cfg.aug,
        label_mode=cfg.data.label_mode,
    )

    tag = args.tag or f"{cfg.model.name}_{cfg.aug.variant}_{cfg.data.name}"
    # Distinguish variants that would otherwise collide (epochs, eca, label smoothing, sampler, optimizer).
    # Skipped when --tag is supplied so users can fully override the run-dir name (e.g., new
    # ablations that differ only in weight_decay / grad_clip, which train.py's auto-tag does not
    # natively differentiate).
    if not args.tag:
        if cfg.train.epochs != 30:
            tag += f"_{cfg.train.epochs}ep"
        if cfg.model.eca:
            tag += "_eca"
        if cfg.train.label_smoothing != 0:
            tag += f"_ls{str(cfg.train.label_smoothing).replace('.', '').replace('0', '', 1) or '0'}"
            # 0.1 -> ls01, 0.05 -> ls05
            if tag.endswith("_ls1"):
                tag = tag[:-4] + "_ls01"
        if cfg.train.sampler != "none":
            tag += f"_{cfg.train.sampler}"
        if cfg.train.class_weighted_loss:
            tag += "_cw"
        if getattr(cfg.train, "loss_type", "ce") != "ce":
            tag += f"_{cfg.train.loss_type}"
        if cfg.train.optimizer != "adamw":
            tag += f"_{cfg.train.optimizer}"
    out_root = Path(cfg.train.output_dir) / tag
    if out_root.exists() and any(out_root.glob("seed_*.pth")) and not args.tag:
        print(f"WARNING: {out_root} already contains checkpoints — will overwrite! Use --tag to avoid.")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.yaml").write_text(Path(args.config).read_text())

    for seed_idx in range(cfg.train.num_seeds):
        seed = cfg.train.seed + seed_idx
        set_seed(seed)
        model = to_device(get_model(cfg.model.name, cfg.model.num_classes, cfg.model.in_channels, cfg.model.pretrained, eca=cfg.model.eca), device)
        train_loader = build_loaders(cfg, device, "train")
        valid_loader = build_loaders(cfg, device, "validation")

        tb = None
        if not args.no_tb:
            tb = TBLogger(out_root / f"tb_seed_{seed}")

        history = trainer.train(model, train_loader, valid_loader, lr=cfg.train.lr, tb_logger=tb)
        ckpt = out_root / f"seed_{seed}.pth"
        torch.save(model.state_dict(), ckpt)
        (out_root / f"seed_{seed}_history.json").write_text(json.dumps(history))
        if tb is not None:
            tb.log_hparams(
                {k: getattr(cfg.train, k) for k in ["lr", "epochs", "batch_size"]},
                {"hparam/best_val_acc": max(history["val_acc"])},
            )
            tb.close()
        print(f"Seed {seed}: saved {ckpt}")


if __name__ == "__main__":
    main()
