from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fer.config import AugConfig, TrainConfig
from fer.data.mixup_cutmix import cutmix_data, mixup_criterion, mixup_data
from fer.training.logging import TBLogger
from fer.training.metrics import compute_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device,
    num_classes: int = 7,
    soft_labels: bool = False,
) -> dict:
    """Evaluate model on loader. If soft_labels=True, the dataset returns (N, num_classes)
    probability vectors; loss uses the criterion as-is, but accuracy/F1 metrics are
    computed against the argmax (hard) ground truth.
    """
    model.eval()
    total_loss, n = 0.0, 0
    all_y_hard, all_pred = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        total_loss += criterion(out, labels).item() * labels.size(0)
        n += labels.size(0)
        preds = out.argmax(dim=1).cpu().numpy()
        if soft_labels and labels.dim() > 1:
            hard = labels.argmax(dim=1).cpu().numpy()
        else:
            hard = labels.cpu().numpy().reshape(-1)
        all_y_hard.extend(hard)
        all_pred.extend(preds)
    metrics = compute_metrics(all_y_hard, all_pred, num_classes=num_classes)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def _soft_mix_loss(criterion, out, y_a, y_b, lam):
    """Linear interpolation of soft targets for KLD/soft-label losses."""
    target = lam * y_a + (1 - lam) * y_b
    return criterion(out, target)


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        device,
        criterion: nn.Module,
        num_classes: int = 7,
        aug: AugConfig | None = None,
        label_mode: str = "hard",
    ):
        self.cfg = cfg
        self.device = device
        self.criterion = criterion
        self.num_classes = num_classes
        self.aug = aug or AugConfig()
        self.label_mode = label_mode

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float | None = None,
        tb_logger: TBLogger | None = None,
    ) -> dict:
        lr = lr or self.cfg.lr
        if self.cfg.optimizer == "ranger":
            try:
                import ranger21  # type: ignore

                optimizer = ranger21.Ranger21(model.parameters(), lr=lr, weight_decay=self.cfg.weight_decay, num_epochs=self.cfg.epochs, num_batches_per_epoch=len(train_loader))
            except ImportError:
                print("Ranger21 not installed, falling back to AdamW (pip install ranger21)")
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=self.cfg.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=self.cfg.epochs * len(train_loader),
            pct_start=self.cfg.warmup_frac,
        )

        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
        best_acc = 0.0
        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            running_loss, n = 0.0, 0
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                if self.aug.variant in ("v5", "hybrid"):
                    mixed, y_a, y_b = cutmix_data(images, labels, self.aug.cutmix_alpha)
                    out = model(mixed)
                    if self.label_mode == "soft":
                        loss = _soft_mix_loss(self.criterion, out, y_a, y_b, 0.5)
                    else:
                        loss = mixup_criterion(self.criterion, out, y_a, y_b, 0.5)
                else:
                    out = model(images)
                    loss = self.criterion(out, labels)
                optimizer.zero_grad()
                loss.backward()
                if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                running_loss += loss.item() * labels.size(0)
                n += labels.size(0)

            val = evaluate(
                model,
                valid_loader,
                self.criterion,
                self.device,
                self.num_classes,
                soft_labels=(self.label_mode == "soft"),
            )
            history["train_loss"].append(running_loss / max(n, 1))
            history["val_loss"].append(val["loss"])
            history["val_acc"].append(val["accuracy"])
            history["val_macro_f1"].append(val["macro_f1"])
            print(
                f"Epoch {epoch}/{self.cfg.epochs} "
                f"train_loss={history['train_loss'][-1]:.4f} "
                f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f} macro_f1={val['macro_f1']:.4f}"
            )
            if tb_logger is not None:
                tb_logger.log_epoch(epoch, history["train_loss"][-1], val)
            if val["accuracy"] > best_acc:
                best_acc = val["accuracy"]
        return history
