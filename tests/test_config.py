from __future__ import annotations

from fer.config import Config


def test_config_roundtrip(tmp_path):
    cfg = Config()
    cfg.model.name = "mobilenetv3_small"
    cfg.aug.variant = "v5"
    cfg.train.num_seeds = 3
    assert cfg.model.name == "mobilenetv3_small"
    assert cfg.aug.variant == "v5"


def test_config_from_yaml():
    import yaml

    raw = {"model": {"name": "efficientnet_b0"}, "aug": {"variant": "v2"}, "train": {"lr": 0.003}}
    cfg = Config.from_dict(raw)
    assert cfg.model.name == "efficientnet_b0"
    assert cfg.aug.variant == "v2"
    assert cfg.train.lr == 0.003
