#!/usr/bin/env bash
# Train the DROID models behind fig:cls-ladder-droid and fig:compute-pareto-real:
# 3 models x 3 seeds. Each run takes one process per GPU (srun task), so run this inside
# a SLURM allocation with 4 GPUs and 4 tasks per node. The runs are independent; on a
# cluster, submit each line as its own job.
set -e
cd "$(dirname "$0")/.."

MODELS=${MODELS:-"lewm hwm_l2 hjepa_l2"}
SEEDS=${SEEDS:-"1 1000 10000"}
declare -A GPUS=([lewm]=4 [hwm_l2]=2 [hjepa_l2]=2)

for model in $MODELS; do
  for seed in $SEEDS; do
    srun --ntasks-per-node="${GPUS[$model]}" python main_hjepa.py \
      --config-name "droid_${model}" seed="$seed" trainer.devices="${GPUS[$model]}"
  done
done
