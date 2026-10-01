#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=normal
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=47:59:59
#SBATCH --job-name="hjepa"
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# usage: sbatch run.sbatch [<code_root>] "<command>" ["<command>" ...]
# <code_root> defaults to the live repo; pass a snapshot dir for training runs.
set -euo pipefail

# activate the stable_wm env here (e.g. source "$VENV/bin/activate")

code_root="${REPO:?set REPO to the repo root}"
if [ "$#" -gt 0 ] && [ -d "$1" ]; then
  code_root="$1"
  shift
fi
echo "Using code root: $code_root"
export PYTHONPATH="$code_root:$code_root/h_jepa${PYTHONPATH:+:$PYTHONPATH}"
export STABLEWM_HOME="${STABLEWM_HOME:?set STABLEWM_HOME}"
cd "$code_root/h_jepa"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset MUJOCO_EGL_DEVICE_ID
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for command in "$@"; do
  echo "Running: $command"
  eval "$command"
done
