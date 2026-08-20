from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fer.config import Config
from fer.data.loaders import build_loaders
from fer.models.factory import get_model
from fer.training.losses import build_criterion
from fer.training.trainer import Trainer
from fer.utils.device import get_device, to_device
from fer.utils import set_seed


def main():
    p = argparse.ArgumentParser(description="Fine-tune FERPlus checkpoint on FER-2013 (10 epochs, lr 1e-4)")
    p.add_argument("--src-ckpt", required=True, help="FERPlus V2 checkpoint (e.g. runs/efficientnet_b0_v2_ferplus/seed_42.pth)")
    p.add_argument("--config", default="configs/train/efficientnet_b0_v2_fer2013.yaml", help="FER-2013 V2 config (defines data split, aug v2)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="Output dir (default: runs/efficientnet_b0_v2_ferplus_to_fer2013/seed_<seed>)")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    device = get_device(cfg.train.device)
    criterion = build_criterion(cfg.data.label_mode, cfg.train.label_smoothing, cfg.model.num_classes)
    trainer = Trainer(cfg.train, device, criterion, cfg.model.num_classes, aug=cfg.aug, label_mode=cfg.data.label_mode)

    set_seed(args.seed)
    model = to_device(get_model(cfg.model.name, cfg.model.num_classes, cfg.model.in_channels, pretrained=False, eca=cfg.model.eca), device)
    sd = torch.load(args.src_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    print(f"Loaded {args.src_ckpt}")

    train_loader = build_loaders(cfg, device, "train")
    valid_loader = build_loaders(cfg, device, "validation")
    history = trainer.train(model, train_loader, valid_loader, lr=args.lr)

    out_dir = Path(args.out) if args.out else Path(f"runs/efficientnet_b0_v2_ferplus_to_fer2013/seed_{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"seed_{args.seed}.pth")
    import json
    (out_dir / f"seed_{args.seed}_history.json").write_text(json.dumps(history))
    print(f"Saved {out_dir}/seed_{args.seed}.pth")


if __name__ == "__main__":
    main()
