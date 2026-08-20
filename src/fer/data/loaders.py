from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from fer.config import Config
from fer.data.datasets import FERDataset
from fer.data.transforms import eval_transform, train_transform


def build_loaders(cfg: Config, device=None, split: str = "train"):
    root = Path(cfg.data.root) / cfg.data.name
    split = {"train": "train", "validation": "validation", "test": "test"}[split]
    if split == "train":
        if cfg.aug.variant == "hybrid":
            from fer.data.hybrid import HybridTransform

            transform = HybridTransform(cfg.aug, p_v2=0.5)
        else:
            transform = train_transform(cfg.aug.variant, cfg.aug)
        shuffle = True
        sampler = None
        if cfg.train.sampler == "weighted":
            import torch
            from collections import Counter

            # Compute per-sample weights: 1/sqrt(count) per class
            # Need to know class distribution without loading all images twice — use dataset samples
            tmp_ds = FERDataset(root=root, split=split, label_mode="hard", transform=None, dataset=cfg.data.name)
            counts = Counter(label for _, label in tmp_ds.samples)
            # Weight per sample
            weights = [1.0 / (counts[label] ** 0.5) for _, label in tmp_ds.samples]
            sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle = False
    else:
        transform = eval_transform(cfg.data.image_size)
        shuffle = False
        sampler = None

    dataset = FERDataset(
        root=root,
        split=split,
        label_mode=cfg.data.label_mode,
        transform=transform,
        dataset=cfg.data.name,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size if split == "train" else cfg.train.batch_size * 2,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=(device is not None and device.type == "cuda"),
    )
