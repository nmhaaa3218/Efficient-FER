# Augmentation Diversity & Lightweight Ensembles for 48x48 FER

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)

Official open-source implementation and reproduction artifacts for the paper:  
**"A Systematic Study of Augmentation Diversity and Lightweight Ensembles for 48×48 Facial Expression Recognition"**

---

## Overview

This repository provides the complete PyTorch implementation, training configurations, pre-trained checkpoints, and evaluation scripts for lightweight Facial Expression Recognition (FER) on tiny $48\times48$ grayscale inputs using adapted **1-channel EfficientNet-B0** backbones ($0.023$ GFLOPs, $4.06$ M parameters).

### Key Performance Summary
- **FER-2013 PrivateTest:**
  - Best Single Model: **$68.18\pm0.46\%$** (Spatial-Occlusion 50ep LS)
  - Headline 2-Model Ensemble: **$70.12\pm0.88\%$** at **$0.046$ GFLOPs** (Macro F1: $69.86\pm0.74\%$)
  - G1 Controlled Regularization Ensemble: **$69.38\pm0.70\%$** ($+2.38$ pt gain over matched singles)
  - 6-Model Variant-Mean Ensemble: **$72.39\%$** at **$0.138$ GFLOPs**
- **FERPlus Crowd-Consensus Test:**
  - Hard Labels: **$80.86\pm0.31\%$** | Soft KLD Labels: **$81.62\pm0.24\%$**
- **RAF-DB Zero-Shot Transfer Test:**
  - 2-Model Ensemble: **$80.27\pm0.13\%$** | 6-Model Ensemble: **$81.88\%$** (no dataset tuning)

---

## Repository Structure

```
.
├── configs/train/      # YAML training configs for all regimes (FER-2013, FERPlus, RAF-DB, G1 ablation)
├── data/               # Dataset preparation scripts and split mappings (see DATASETS.md)
├── figures/            # Generated figures (confusion matrices, Grad-CAM, Pareto frontiers)
├── results/            # Complete audited JSON evaluation logs, bootstraps, and McNemar tests
├── runs/               # Pre-trained checkpoint weights (.pth), configs, and training histories
├── scripts/            # Autonomous evaluation, audit, and reproduction scripts
├── src/fer/            # Core library (models, transforms, datasets, trainer, evaluation, profiling)
├── tests/              # Unit tests
├── DATASETS.md         # Detailed dataset provenance, download links, and preparation guide
└── requirements.txt    # Python dependencies
```

---

## Installation

```bash
git clone https://github.com/nmhaaa3218/FER-efficient-ensemble.git
cd FER-efficient-ensemble

python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements.txt
```

---

## Dataset Preparation

Detailed dataset download links and folder layouts are documented in **[DATASETS.md](DATASETS.md)**.

```bash
# Fetch and build FER-2013 and FERPlus
./scripts/fetch_datasets.sh
python -m fer.scripts.prepare_fer2013 --csv data/_sources/fer2013.csv --out data/fer2013
python -m fer.scripts.prepare_ferplus --fer data/_sources/fer2013.csv --ferplus data/_sources/fer2013new.csv --out data/ferplus

# Build RAF-DB 48x48 1ch (after placing RAF-DB in data/rafdb/)
python -m fer.data.rafdb --in data/rafdb --out data/rafdb
```

---

## Evaluation & Reproduction

All metrics, statistical significance tests, and tables reported in the manuscript can be verified directly:

```bash
# 1. Evaluate FER-2013 single models and 2-model ensemble
python scripts/validate_50ep.py
python scripts/decomposition_audit.py

# 2. Evaluate G1 controlled regularization ablation
python scripts/g1_evaluate.py

# 3. Evaluate FERPlus hard vs. soft label training
python scripts/ferplus_soft_hard_eval.py

# 4. Evaluate RAF-DB zero-shot transfer test
python scripts/rafdb_per_class_eval.py

# 5. Measure ONNX Runtime CPU latency & fvcore FLOPs
fer-benchmark --models efficientnet_b0 mobilenetv3_small shufflenetv2_0_5x --onnx
```

---

## License & Citation

This project is released under the **MIT License**.
