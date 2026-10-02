---
name: slurm-submit
description: Submit one or more shell commands (training, planning eval, probing, data collection, eval-task generation) to SLURM as one-off jobs via the skill's run.sbatch. Use when the user asks to submit, launch, or run something on the cluster / SLURM.
---

# SLURM Submit

Submit commands through `.agents/skills/slurm-submit/run.sbatch` (in this skill's folder). The
script requests 1 GPU / 10 CPUs / 100G / 48h on `<partition>` (edit the `#SBATCH` lines for your cluster), sets `PYTHONPATH` to the code
root (so this repo's `stable_worldmodel` is imported), `cd`s into `<code_root>/h_jepa`, sets the
MuJoCo/EGL and threading env, and `eval`s each command string in order.

## Confirm before submitting

Never run `sbatch` on the first pass. Print the exact `sbatch` line(s), the resource profile, and
whether the batch runs on the live repo or a snapshot, then wait for the user's go-ahead. Skip
this only if the request already says to submit without asking.

## Submitting

Submit from `h_jepa/` so `logs/%j.out` / `logs/%j.err` land in `h_jepa/logs/`:

```bash
cd $REPO/h_jepa
mkdir -p logs
sbatch .agents/skills/slurm-submit/run.sbatch "<command>"
```

- One `sbatch` call is one job; several command strings in one call run sequentially in that job.
  Submit one call per command to run them in parallel. Training, resume and eval sweeps normally go through `launch.py` (it submits itself; logs in `<run_dir>/slurm/`); this skill is for one-off commands.
- Commands run with cwd `h_jepa/`. Scripts outside it (e.g. `scripts/data/*.py` at the repo root)
  need absolute paths.
- Hydra scripts take `--config-name <stem>`, resolved against the script's own config dir.
- CPU-only jobs (data collection, analysis): `sbatch --partition=<cpu partition> --gres=none .agents/skills/slurm-submit/run.sbatch "<command>"`.
  CLI flags override the in-script `#SBATCH` lines. Say "0 GPU / cpu partition" in the preview.

## Training runs: snapshot the code

`main_hjepa.py` writes a resume checkpoint and is auto-resumed after a preemption requeue. If the
requeued job reads the live repo and the code has changed, the resume can fail. For any batch with
a training command, copy the code once and pass the snapshot as the first argument:

```bash
cd $REPO
SNAP=${REPO}_slurm_snapshots/$(date +%Y%m%d_%H%M%S)
mkdir -p "$SNAP"
git ls-files -co --exclude-standard | rsync -a --files-from=- . "$SNAP"
ln -sfn $REPO/h_jepa/assets "$SNAP/h_jepa/assets"
git rev-parse HEAD > "$SNAP/GIT_COMMIT"
git status --short > "$SNAP/GIT_STATUS"
git diff HEAD > "$SNAP/UNCOMMITTED.diff"
cd h_jepa
sbatch .agents/skills/slurm-submit/run.sbatch "$SNAP" "python main_hjepa.py --config-name ant_hjepa_l3 seed=42"
```

The snapshot includes uncommitted edits (tracked and untracked, non-ignored files) and records the
commit it was taken from plus the uncommitted changes. Share one snapshot across all jobs of a batch.
When the run dir is known up front (absolute `subdir=`), also write `<run_dir>/snapshot.txt` with the
snapshot path, commit and SLURM job id (`sbatch --parsable`), so the run traces back to its code. Evals, probing and data jobs are single-shot, so they run on
the live repo.

## Reporting

Parse `Submitted batch job <id>` and report, for each command in the order given: the job id, the
log path `h_jepa/logs/<id>.out`, and the output dir (training: `$HJEPA_HOME/ckpts/<subdir>`;
eval: `output.dir`, relative paths resolve next to the checkpoint).

Before submitting, check whether the same command was already submitted (`squeue -u $USER -o
"%i %j %Z"`, recent `h_jepa/logs/`) and flag a likely duplicate.

## Example

> 3 jobs (1 GPU / 100G / 48h each), training → snapshot. Confirm?
>
> ```bash
> # snapshot as above, then:
> for s in 42 43 44; do
>   sbatch .agents/skills/slurm-submit/run.sbatch "$SNAP" "python main_hjepa.py --config-name fourroom_hjepa_l3 seed=$s"
> done
> ```
