#!/usr/bin/env bash
# push_to_limit.sh
# Master sweep for absolute peak — remaining runs after V2/V5 fer2013+ferplus are done.
# Fire-and-forget on MPS (~8-12 hrs) or RTX 3070 Ti (~4-6 hrs).
#
# Already done (do not re-run):
#   efficientnet_b0_v2_fer2013  (3 seeds, 0.6662)
#   efficientnet_b0_v5_fer2013  (3 seeds, 0.6446)
#   efficientnet_b0_v2_ferplus  (3 seeds, 0.8027)
#   efficientnet_b0_v5_ferplus  (3 seeds, 0.7699 soft)
#
# This script runs the remaining peak-push + baselines + novelty.
#
# Usage:
#   ./scripts/push_to_limit.sh                 # all remaining, 30 epochs, 3 seeds
#   EPOCHS=10 DRY_RUN=1 ./scripts/push_to_limit.sh
set -euo pipefail

SEEDS="${SEEDS:-3}"
EPOCHS="${EPOCHS:-30}"
DRY_RUN="${DRY_RUN:-0}"

run() {
  local cfg="$1"
  echo "=== $cfg | $EPOCHS epochs | $SEEDS seeds ==="
  if [ "$DRY_RUN" = "1" ]; then
    echo "  python -m fer.scripts.train --config $cfg --epochs $EPOCHS --seeds $SEEDS"
  else
    python -m fer.scripts.train --config "$cfg" --epochs "$EPOCHS" --seeds "$SEEDS"
  fi
}

# 1. Baselines (Pareto frontier — must have for paper)
run "configs/train/mobilenetv3_fer2013.yaml"
run "configs/train/mobilenetv3_small_ferplus.yaml"
run "configs/train/shufflenetv2_fer2013.yaml"
run "configs/train/shufflenetv2_0_5x_ferplus.yaml"

# 2. V1 baselines (no augmentation — ablation)
run "configs/train/efficientnet_b0_v1_fer2013.yaml"
run "configs/train/efficientnet_b0_v1_ferplus.yaml"

# 3. ECA novelty (tiny +3 params, 1-run ablation per dataset)
run "configs/train/efficientnet_b0_eca_v2_fer2013.yaml"
run "configs/train/efficientnet_b0_eca_v2_ferplus.yaml"

# 4. Hyperparameter ablations (label smoothing + longer training)
# LS01 variants use 30 epochs (label_smoothing 0.1), 50ep variant uses 50 epochs
run "configs/train/efficientnet_b0_v1_fer2013_ls01.yaml"
run "configs/train/efficientnet_b0_v5_fer2013_ls01.yaml"
# 50 epochs with LS01 (uses config's 50, not $EPOCHS)
echo "=== configs/train/efficientnet_b0_v2_fer2013_50ep_ls01.yaml | 50 epochs | $SEEDS seeds ==="
if [ "$DRY_RUN" = "1" ]; then
  echo "  python -m fer.scripts.train --config configs/train/efficientnet_b0_v2_fer2013_50ep_ls01.yaml --seeds $SEEDS"
else
  python -m fer.scripts.train --config configs/train/efficientnet_b0_v2_fer2013_50ep_ls01.yaml --seeds "$SEEDS"
fi

# 5. Fine-tune FERPlus V2 -> FER-2013 (10 epochs, lr 1e-4, per seed)
if [ "$DRY_RUN" = "1" ]; then
  echo "=== finetune FERPlus V2 -> FER-2013 (10 epochs, 3 seeds) ==="
  for seed in 42 43 44; do
    echo "  python -m fer.scripts.finetune --src-ckpt runs/efficientnet_b0_v2_ferplus/seed_${seed}.pth --seed $seed"
  done
else
  for seed in 42 43 44; do
    echo "=== finetune seed $seed ==="
    python -m fer.scripts.finetune --src-ckpt "runs/efficientnet_b0_v2_ferplus/seed_${seed}.pth" --seed "$seed"
  done
fi

echo "Push to limit complete."
