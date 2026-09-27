#!/usr/bin/env bash
# Train every model behind the depth figure and the cost-ladder tables: 4 environments
# x 7 models x 3 seeds. Each run ends with its planning eval and final probing/decoding
# eval. The runs are independent; on a cluster, submit each line as its own job.
set -e
cd "$(dirname "$0")/.."

ENVS=${ENVS:-"ant fourroom cube pusht"}
MODELS=${MODELS:-"lewm hjepa_l2 hjepa_l3 hjepa_l4 hwm_l2 hwm_l3 hwm_l4"}
SEEDS=${SEEDS:-"42 43 44"}

for env in $ENVS; do
  for model in $MODELS; do
    for seed in $SEEDS; do
      python main_hjepa.py --config-name "${env}_${model}" seed="$seed"
    done
  done
done
