#!/usr/bin/env bash
# Offline DROID planning evals behind fig:cls-ladder-droid: the paper planner cell of
# each model on the 16 evaluation clips, with planner seeds 1/2/3 for every trained
# model, then the mean Frechet fidelity per model. Checkpoints come from `ENVS=droid train_all.sh`
# ($STABLEWM_HOME/ckpts/droid/droid_<model>/seed<seed>/); H-JEPA is read at epoch 100.
# The eval config follows the model name (lewm*: droid_flat, *_l<N>*: droid_l<N>).
# EPOCHS="50 100 150" evaluates droid_<model>_epoch_<N>_object.ckpt instead, into
# eval_<plan>/epoch_<N>/, skipping missing checkpoints.
set -e
cd "$(dirname "$0")/.."

MODELS=${MODELS:-"lewm hwm_l2 hjepa_l2"}
SEEDS=${SEEDS:-"1 1000 10000"}
PLAN_SEEDS=${PLAN_SEEDS:-"1 2 3"}
EPOCHS=${EPOCHS:-""}
RUNS=${RUNS:-$STABLEWM_HOME/ckpts/droid}

for model in $MODELS; do
  case $model in
    lewm*) cell="--lr 0.01 --num-samples 16" ;;
    hwm*) cell="--lr 0.01 0.1 --num-samples 16" ;;
    hjepa*) cell="--lr 0.03 0.01 --num-samples 4" ;;
  esac
  plan=flat
  [[ $model =~ _l([0-9]) ]] && plan=l${BASH_REMATCH[1]}
  for seed in $SEEDS; do
    run="$RUNS/droid_${model}/seed${seed}"
    for epoch in ${EPOCHS:-final}; do
      if [[ $epoch == final ]]; then
        ckpt="$run/droid_${model}_object.ckpt" out="$run/eval_${plan}"
        [[ $model == hjepa_l2 ]] && ckpt="$run/droid_hjepa_l2_epoch_100_object.ckpt"
      else
        ckpt="$run/droid_${model}_epoch_${epoch}_object.ckpt" out="$run/eval_${plan}/epoch_${epoch}"
      fi
      if [[ ! -f $ckpt ]]; then
        echo "skip: missing $ckpt"
        continue
      fi
      for ps in $PLAN_SEEDS; do
        python droid_plan_eval.py --ckpt "$ckpt" --config "droid_${plan}" $cell --seed "$ps" \
          --out "$out" --tag "plan_seed${ps}"
      done
    done
  done
done

python - "$MODELS" "$SEEDS" "$PLAN_SEEDS" "${EPOCHS:-final}" "$RUNS" <<'PY'
import csv, os, re, sys
import numpy as np

models, seeds, plan_seeds, epochs = (a.split() for a in sys.argv[1:5])
for model in models:
    m = re.search(r"_l(\d)", model)
    plan = f"l{m.group(1)}" if m else "flat"
    for epoch in epochs:
        sub = "" if epoch == "final" else f"/epoch_{epoch}"
        per_seed = []
        for seed in seeds:
            run = f"{sys.argv[5]}/droid_{model}/seed{seed}/eval_{plan}{sub}"
            paths = [f"{run}/plan_seed{ps}/eval.csv" for ps in plan_seeds]
            if all(os.path.exists(p) for p in paths):
                rows = [next(csv.DictReader(open(p))) for p in paths]
                per_seed.append(100 * np.mean([float(r["frechet/skill_mean"]) for r in rows]))
        if not per_seed:
            continue
        se = np.std(per_seed, ddof=1) / np.sqrt(len(per_seed))
        print(
            f"droid_{model} {epoch}: Frechet fidelity {np.mean(per_seed):.2f} +- {se:.2f} "
            f"(%, mean +- SE over {len(per_seed)} train seeds)"
        )
PY
