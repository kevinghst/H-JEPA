#!/bin/bash
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --open-mode=append
# Job body for h_jepa/launch.py, which passes partition/account/qos/time/gpus/cpus/mem/output on the
# sbatch command line and exports HJ_CODE (code snapshot root), HJ_VENV, HJ_GPUS, HJEPA_HOME.
#   launch.sbatch train <main_hjepa.py args...>        (also used by `launch.py resume`)
#   launch.sbatch eval "<config>|<seed>|<ckpt>|<outdir>|<tasks or empty>"... -- [eval.py overrides...]
# A requeue re-runs this script: training resumes from <run_dir>/lightning_resume/last.ckpt and
# evals whose <outdir>/metrics.yaml exists are skipped.
set -uo pipefail
mode=$1; shift
source "$HJ_VENV/bin/activate"
cd "$HJ_CODE/h_jepa"
export PYTHONPATH=$HJ_CODE:$HJ_CODE/h_jepa PYTHONDONTWRITEBYTECODE=1 HYDRA_FULL_ERROR=1 PYTHONFAULTHANDLER=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset MUJOCO_EGL_DEVICE_ID
echo "mode=$mode code=$HJ_CODE job=${SLURM_JOB_ID} restart=${SLURM_RESTART_COUNT:-0} host=$(hostname) $(date)"
if [ "$mode" == "train" ]; then
  export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
  srun --ntasks-per-node="$HJ_GPUS" python main_hjepa.py "$@"
  exit $?
fi
specs=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do specs+=("$1"); shift; done
[ $# -gt 0 ] && shift
export WANDB_MODE=disabled
rc=0
for spec in "${specs[@]}"; do
  IFS='|' read -r config seed ckpt outdir tasks <<< "$spec"
  if [ -f "$outdir/metrics.yaml" ]; then echo "SKIP (done) $outdir"; continue; fi
  mkdir -p "$outdir"
  t0=$(date +%s)
  echo "EVAL $config seed=$seed $ckpt -> $outdir"
  python eval.py --config-name "$config" seed="$seed" policy="$ckpt" output.dir="$outdir" \
    ${tasks:+load_eval_trajs_path="$tasks"} hydra.run.dir="$outdir/hydra" "$@" || { echo "FAILED $outdir"; rc=1; }
  echo "TIME $(( $(date +%s) - t0 ))s $outdir"
  [ -f "$outdir/metrics.yaml" ] && grep -E '^success_rate' "$outdir/metrics.yaml"
done
exit $rc
