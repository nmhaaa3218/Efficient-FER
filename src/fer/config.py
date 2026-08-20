from __future__ import annotations

import dataclasses
import yaml
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    name: str = "fer2013"  # fer2013 | ferplus
    root: str = "data"
    label_mode: str = "hard"  # hard | soft (ferplus only)
    image_size: int = 48
    in_channels: int = 1


@dataclass
class ModelConfig:
    name: str = "efficientnet_b0"  # efficientnet_b0 | mobilenetv3_small | shufflenetv2_0_5x
    num_classes: int = 7
    in_channels: int = 1
    pretrained: bool = True
    eca: bool = False


@dataclass
class AugConfig:
    variant: str = "v1"  # v1 | v2 | v3 | v4 | v5
    image_size: int = 48
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.5
    crop_padding: int = 4
    rotation_deg: float = 10.0
    flip_prob: float = 0.5
    color_brightness: float = 0.2
    color_contrast: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "onecycle"
    warmup_frac: float = 0.1
    num_seeds: int = 3
    seed: int = 42
    output_dir: str = "runs"
    device: str = "auto"
    num_workers: int = 0
    log_freq: int = 50
    label_smoothing: float = 0.0
    grad_clip: float | None = None
    sampler: str = "none"  # none | weighted
    class_weighted_loss: bool = False
    loss_type: str = "ce"  # ce | focal


@dataclass
class EnsembleConfig:
    enabled: bool = True
    model_aug: str = "v2"  # spatial/CRP model
    model_b_aug: str = "v5"  # mixing model
    weight_a: float = 0.45
    fusion: str = "weighted_avg"  # weighted_avg | rrf
    rrf_k: int = 60


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            ftype = f.type
            ftype = ftype.replace(" | None", "").replace("| None", "").strip()
            if f.name in raw:
                kwargs[f.name] = raw[f.name]
        obj = cls()
        for name, value in raw.items():
            sub = getattr(obj, name, None)
            if sub is not None and dataclasses.is_dataclass(sub) and isinstance(value, dict):
                for k, v in value.items():
                    if hasattr(sub, k):
                        setattr(sub, k, v)
        return obj

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
