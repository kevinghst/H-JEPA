#!/usr/bin/env bash
# Standalone planning evals behind the bottom row of the depth figure: LeWM with flat
# planning, H-JEPA and HWM with 2/3/4-level planning. Model and planner seeds are paired.
# Training already runs the same evals at the end; this script re-runs them from saved
# checkpoints ($HJEPA_HOME/ckpts/<env>/<env>_<model>/seed<seed>/).
set -e
cd "$(dirname "$0")/.."

ENVS=${ENVS:-"ant fourroom cube pusht"}
SEEDS=${SEEDS:-"42 43 44"}

eval_model() {
  local env=$1 model=$2 plan=$3 seed=$4
  local run="$HJEPA_HOME/ckpts/${env}/${env}_${model}/seed${seed}"
  python eval.py --config-name "${env}_${plan}" seed="$seed" \
    policy="$run/${env}_${model}_object.ckpt" output.dir="$run/eval_${plan}"
}

for env in $ENVS; do
  for seed in $SEEDS; do
    eval_model "$env" lewm flat "$seed"
    for n in 2 3 4; do
      eval_model "$env" "hjepa_l$n" "l$n" "$seed"
      eval_model "$env" "hwm_l$n" "l$n" "$seed"
    done
  done
done
