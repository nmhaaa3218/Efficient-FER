#!/usr/bin/env bash
# run_sweep.sh
# From-scratch training sweep for the paper. 5 model variants x 2 datasets x 3 seeds.
# Fire-and-forget on the RTX 3070 Ti (~5-10 hrs).
#
# Produces (16 runs x 3 seeds = 48 checkpoints, ~15h MPS):
#   runs/efficientnet_b0_v1_fer2013/seed_{42,43,44}.pth (Baseline 30ep)
#   runs/efficientnet_b0_v2_fer2013/seed_{42,43,44}.pth (Spatial 30ep) + _50ep_ls01 (Spatial 50ep LS) → 68.08% mean, 68.90 peak
#   runs/efficientnet_b0_v3_fer2013/seed_{42,43,44}.pth (Photometric 30ep) + _50ep_ls01 (68.13%)
#   runs/efficientnet_b0_v4_fer2013/seed_{42,43,44}.pth (Occlusion 30ep) + _50ep_ls01 (68.18%)
#   runs/efficientnet_b0_v5_fer2013/seed_{42,43,44}.pth (Mixing 30ep, soft KLD for FERPlus)
#   runs/efficientnet_b0_hybrid_fer2013/seed_{42,43,44}.pth (Hybrid 30ep)
#   runs/efficientnet_b0_{v1,v2,v5}_ferplus/seed_{42,43,44}.pth (FERPlus hard/soft, same images 10 annotators)
#   runs/mobilenetv3_small_fer2013 (+ ferplus) / shufflenetv2_0_5x_fer2013 (+ ferplus) (lightweight baselines 0.002-0.004 GFLOPs)
#   → results/ensemble_2model_sweep_50ep.json (15-pair 0.5/0.5 + 0.05 weight sweep, best 72.39% Spatial+Occlusion 50ep w0.55)
#
# All numbers in the paper come from these checkpoints (legacy .pth used only
# for pipeline validation in figures/preliminary/).
#
# Usage:
#   ./scripts/run_sweep.sh                       # default 30 epochs, 3 seeds
#   EPOCHS=10 ./scripts/run_sweep.sh             # quick smoke
#   DRY_RUN=1 ./scripts/run_sweep.sh             # print commands only
set -euo pipefail

SEEDS="${SEEDS:-3}"
EPOCHS="${EPOCHS:-30}"
DRY_RUN="${DRY_RUN:-0}"

declare -a RUNS=(
  # 30ep core — Table 1 singles (Baseline/Spatial/Photometric/Occlusion/Mixing/Hybrid)
  "efficientnet_b0_v1_fer2013"
  "efficientnet_b0_v2_fer2013"
  "efficientnet_b0_v3_fer2013"
  "efficientnet_b0_v4_fer2013"
  "efficientnet_b0_v5_fer2013"
  "efficientnet_b0_hybrid_fer2013"
  # 50ep long schedule — SOTA 72.39% Spatial+Occlusion 6-logit variant-mean (v2/v3/v4 LS01 wd0.1 clip2)
  "efficientnet_b0_v2_fer2013_50ep_ls01"
  "efficientnet_b0_v3_fer2013_50ep_ls01"
  "efficientnet_b0_v4_fer2013_50ep_ls01"
  # FERPlus clean hard/soft ablations (same images, 10 annotators)
  "efficientnet_b0_v1_ferplus"
  "efficientnet_b0_v2_ferplus"
  "efficientnet_b0_v5_ferplus"
  # Lightweight baselines 1ch 48 (same Spatial)
  "mobilenetv3_small_fer2013"
  "mobilenetv3_small_ferplus"
  "shufflenetv2_0_5x_fer2013"
  "shufflenetv2_0_5x_ferplus"
)

for run in "${RUNS[@]}"; do
  cfg="configs/train/${run}.yaml"
  echo "=== $run | $EPOCHS epochs | $SEEDS seeds ==="
  if [ "$DRY_RUN" = "1" ]; then
    echo "  python -m fer.scripts.train --config $cfg --epochs $EPOCHS --seeds $SEEDS"
  else
    python -m fer.scripts.train --config "$cfg" --epochs "$EPOCHS" --seeds "$SEEDS"
  fi
done
echo "Sweep complete: ${#RUNS[@]} runs x $SEEDS seeds = $(( ${#RUNS[@]} * SEEDS )) checkpoints."
