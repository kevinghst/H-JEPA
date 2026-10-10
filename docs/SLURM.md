# Running on SLURM

Complete the [installation](../README.md#1-installation) first and set `HJEPA_HOME`.
Run the commands below from `h_jepa/` (`cd h_jepa` from the repository root); paths are relative to that directory.

Set cluster options in `scripts/slurm/local.yaml` (gitignored), overriding `scripts/slurm/default.yaml`:

```yaml
venv: /path/to/.venv     # its bin/python runs the jobs
partition: gpu
account: my_account
qos: normal
```

```bash
# one job per seed: $HJEPA_HOME/ckpts/cube/my_sweep_<timestamp>/cube_hjepa_l3/seed<seed>/
python scripts/slurm/launch.py train --config-name cube_hjepa_l3 --sweep my_sweep --seeds 42,43,44
# relaunch one run from lightning_resume/last.ckpt
python scripts/slurm/launch.py resume <run_dir>
# one eval.py job per saved epoch checkpoint: <run_dir>/eval_epoch/epoch_<N>/
python scripts/slurm/launch.py eval <sweep_dir|run_dir> --epochs last
```

Use `--grid key=v1,v2` (repeatable) for Hydra grids and `--dry` to preview submissions.
Sweeps run from a submission-time repo copy (`<sweep_dir>/code`). Wall-clock limits are `time`
(training) and `eval.time_by_env` (eval); `--time` overrides both.
