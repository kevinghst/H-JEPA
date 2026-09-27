#!/usr/bin/env bash
# Level-1 planning of the H-JEPA models with the cost measured in each upper level's
# latent (the cost-ladder tables). An n-level model is evaluated with the l2..ln
# projections. The native level-1 column uses the flat planner (<env>_flat).
set -e
cd "$(dirname "$0")/.."

ENVS=${ENVS:-"ant fourroom cube pusht"}
SEEDS=${SEEDS:-"42 43 44"}

for env in $ENVS; do
  for seed in $SEEDS; do
    for n in 2 3 4; do
      run="$STABLEWM_HOME/ckpts/${env}_hjepa_l$n/seed${seed}"
      for plan in flat $(seq -f "l%g_project" 2 "$n"); do
        python eval.py --config-name "${env}_${plan}" seed="$seed" \
          policy="$run/${env}_hjepa_l${n}_object.ckpt" output.dir="$run/eval_${plan}"
      done
    done
  done
done
