#!/usr/bin/env bash
# Train every model behind the depth figure and the cost-ladder tables: 4 environments
# x 7 models x 3 seeds. Each run ends with its planning eval and final probing/decoding
# eval. The runs are independent; on a cluster, submit each line as its own job.
# ENVS=droid trains the DROID models (3 models x 3 seeds) instead; each DROID run takes one
# process per GPU (srun task), so run it inside a SLURM allocation with 2 GPUs and 2 tasks per node.
set -e
cd "$(dirname "$0")/.."

ENVS=${ENVS:-"ant fourroom cube pusht"}

for env in $ENVS; do
  if [[ $env == droid ]]; then
    models=${MODELS:-"lewm hwm_l2 hjepa_l2"} seeds=${SEEDS:-"1 1000 10000"} launch="srun --ntasks-per-node=2"
  else
    models=${MODELS:-"lewm hjepa_l2 hjepa_l3 hjepa_l4 hwm_l2 hwm_l3 hwm_l4"} seeds=${SEEDS:-"42 43 44"} launch=
  fi
  for model in $models; do
    for seed in $seeds; do
      $launch python main_hjepa.py --config-name "${env}_${model}" seed="$seed"
    done
  done
done
