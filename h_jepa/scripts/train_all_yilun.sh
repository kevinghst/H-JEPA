#!/usr/bin/env bash
# Same as train_all.sh, except that Cube and Push-T runs (every model) also run the planning
# eval after every epoch (planning_eval.every_n_epochs=1), not only at the end of training.
set -e
cd "$(dirname "$0")/.."

ENVS=${ENVS:-"ant fourroom cube pusht"}

for env in $ENVS; do
  if [[ $env == droid ]]; then
    models=${MODELS:-"lewm hwm_l2 hjepa_l2"} seeds=${SEEDS:-"1 1000 10000"} launch="srun --ntasks-per-node=2"
  else
    models=${MODELS:-"lewm hjepa_l2 hjepa_l3 hjepa_l4 hwm_l2 hwm_l3 hwm_l4"} seeds=${SEEDS:-"42 43 44"} launch=
  fi
  extra=
  if [[ $env == cube || $env == pusht ]]; then
    extra="planning_eval.every_n_epochs=1"
  fi
  for model in $models; do
    for seed in $seeds; do
      $launch python main_hjepa.py --config-name "${env}_${model}" seed="$seed" $extra
    done
  done
done
