<h1 align="center">
    <p><b>H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning</b></p>
</h1>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2610.06805" target="_blank" style="margin: 2px;"><img alt="ArXiv" src="https://img.shields.io/badge/arXiv-2610.06805-b5212f?logo=arxiv" style="display: inline-block; vertical-align: middle;"/></a>
  <a href="https://h-jepa.com/" target="_blank" style="margin: 2px;"><img alt="Website" src="https://img.shields.io/badge/Website-h--jepa.com-blue" style="display: inline-block; vertical-align: middle;"/></a>
  <a href="https://huggingface.co/datasets/jepa-world-models/h-jepa" target="_blank" style="margin: 2px;"><img alt="HuggingFace Dataset" src="https://img.shields.io/badge/🤗%20Dataset-jepa--world--models/h--jepa-ffc107" style="display: inline-block; vertical-align: middle;"/></a>
  <a href="https://huggingface.co/jepa-world-models/h-jepa" target="_blank" style="margin: 2px;"><img alt="HuggingFace Models" src="https://img.shields.io/badge/🤗%20Models-jepa--world--models/h--jepa-ffc107" style="display: inline-block; vertical-align: middle;"/></a>
</div>

<br>

<p align="center">
  Wancong Zhang*,
  Basile Terver*,
  Mike Rabbat,
  Yann LeCun&dagger;,
  Randall Balestriero&dagger;
</p>

<p align="center">
  <b>AMI Labs</b>, NYU, INRIA Paris, Brown University<br>
  <sub>* equal contribution, &dagger; equal advising</sub>
</p>

<p align="center">
  <img src="assets/teaser.png" alt="H-JEPA hierarchical planning in Visual AntMaze and planning success versus planner compute" width="800">
</p>

<p align="center"><sub><b>Left:</b> hierarchical planning in Visual AntMaze; each level's first predicted state becomes the subgoal of the level below. <b>Right:</b> AntMaze planning success versus planner compute, flat LeWM against two- and three-level H-JEPA.</sub></p>

---

Code for H-JEPA's planning results: model depth and projected-cost comparisons on Visual AntMaze,
FourRoom Distractors, OGBench Cube and Push-T, plus open-loop planning on DROID at 5 fps.
Built on [LeWM](https://github.com/lucas-maes/le-wm) and
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel).

## 1) Layout

```
stable_worldmodel/       vendored environments, dataset loader, solvers and policy
scripts/data/           dataset collection and preparation
h_jepa/
  main_hjepa.py          training, followed by planning and probing/decoding evals
  eval.py               planning eval (DROID: droid_eval.py)
  main_probing_decoding_eval.py   standalone probing/decoding
  models/               JEPA levels, encoders and predictors
  config/{train,eval,probing}/    training recipes and evaluation settings
  scripts/              train/eval scripts, task generators and optional SLURM launcher
```

See [ARCHITECTURE.md](h_jepa/ARCHITECTURE.md) for model and planner internals.
Simulation configs use `<env>` = `ant`, `fourroom`, `cube` or `pusht`:

| Model (`config/train/<env>_<model>.yaml`) | Description |
|---|---|
| `lewm` | Flat LeWM (one level) |
| `hjepa_l2`, `hjepa_l3`, `hjepa_l4` | End-to-end H-JEPA with 2/3/4 levels |
| `hwm_l2`, `hwm_l3`, `hwm_l4` | HWM: identity upper encoders, no upper-level SIGReg |

| Planner (`config/eval/<env>_<planner>.yaml`) | Description |
|---|---|
| `flat` | Level-1 planning |
| `l2`, `l3`, `l4` | Hierarchical planning for H-JEPA and HWM |
| `l2_project`, `l3_project`, `l4_project` | Level-1 planning with cost in the level-2/3/4 latent |

## 2) Installation

```bash
git clone https://github.com/kevinghst/H-JEPA.git && cd H-JEPA
conda create -n hjepa python=3.10 && conda activate hjepa
pip install -e ".[train,env]"
export HJEPA_HOME=/path/to/data   # datasets, expert policies and checkpoints live here
```

**Choose your workflow:**

- **Evaluate pretrained models:** download [evaluation tasks](#42-evaluation-tasks) and
  [checkpoints](#44-pretrained-checkpoints-optional), then run the [depth comparison](#45-depth-comparison-figure-6)
  or [cost ladder](#46-cost-ladder-tables-1-and-9).
- **Train from scratch:** prepare [datasets](#41-datasets) and [evaluation tasks](#42-evaluation-tasks),
  then [train](#43-training).
- **DROID:** follow the separate [data, training and evaluation instructions](#5-reproducing-droid).

## 3) Usage

Run commands from `h_jepa/` unless noted (`cd h_jepa` after installation).

### 3.1) Training

```bash
python main_hjepa.py --config-name <env>_<model> seed=<seed>
```

Runs write to `$HJEPA_HOME/ckpts/<env>/<env>_<model>/seed<seed>/`:

- `<env>_<model>_object.ckpt`, `config.yaml`, logs and the stable-pretraining cache `spt/`;
- `planning_eval/epoch_XXXX/metrics.yaml`: final planning eval with the matching planner and model seed;
- `final_probing_decoding_eval/`: frozen-model probes and decoders (`config/probing/<env>.yaml`).

Set `planning_eval.every_n_epochs=N` for periodic planning evals or `wandb.enabled=true` for W&B;
both are off by default.

### 3.2) Probing and decoding

Evaluate a saved checkpoint:

```bash
python main_probing_decoding_eval.py --config-name <env> \
  policy=$HJEPA_HOME/ckpts/<env>/<env>_<model>/seed<seed>/<env>_<model>_object.ckpt
```

### 3.3) Planning evaluation

```bash
python eval.py --config-name <env>_<planner> seed=<seed> output.dir=<dir> \
  policy=$HJEPA_HOME/ckpts/<env>/<env>_<model>/seed<seed>/<env>_<model>_object.ckpt
```

Use `flat` for LeWM and `l<n>` for n-level H-JEPA/HWM. `l<k>_project` encodes planned level-1
states and the goal up to level k to measure cost; it requires an H-JEPA model with at least k levels.
`metrics.yaml` reports `success_rate`; `output.dir` is relative to the checkpoint directory.

Evals resume by skipping saved chunks (`chunks/tasks_<start>-<end>.json`). Each chunk uses
seed + first task index. Set `+eval.chunk_size=N` (default 1), or `null` to plan all tasks together.
[DROID evals](#54-planning-evaluation) similarly skip saved clips and use seed + clip index.

### 3.4) Running on SLURM (optional)

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

## 4) Reproducing Push-T, OGBench Cube, Visual AntMaze, FourRoom

Reproduces the bottom row of Figure 6 and the level-1 columns of Tables 1 and 9.
Tables report paper success rates
(%, mean ± SE over seeds 42, 43, 44; planner seed = model seed).
**For evaluation only, download [tasks](#42-evaluation-tasks) and [checkpoints](#44-pretrained-checkpoints-optional);
datasets are needed for training and probing.**

### 4.1) Datasets

The datasets are HDF5 files under `$HJEPA_HOME`.

| Environment | Files | Source |
|---|---|---|
| Push-T | `pusht_expert_train.h5`, `pusht_expert_val.h5` | download (LeWM + HF) |
| OGBench Cube | `cube_single_expert_train.h5`, `cube_single_expert_val.h5` | download (LeWM) + split |
| Visual AntMaze | `visual_antmaze_medium_{explore_stitch_train,stitch_val,probing_train,probing_eval}.h5` | download (HF) or generate |
| FourRoom Distractors | `fourroom_tp35_d1{,_val,_probing,_probing_val}.h5` | download (HF) or generate |

**Push-T and Cube.** The training data is LeWM's release
([`quentinll/lewm-pusht`](https://huggingface.co/datasets/quentinll/lewm-pusht),
[`quentinll/lewm-cube`](https://huggingface.co/datasets/quentinll/lewm-cube), MIT). The Push-T
validation set is in our release. From the repository root:

```bash
hf download quentinll/lewm-pusht --repo-type dataset --local-dir $HJEPA_HOME --include "*.zst"
hf download quentinll/lewm-cube --repo-type dataset --local-dir $HJEPA_HOME --include "*.zst"
hf download jepa-world-models/h-jepa pusht_expert_val.h5 --repo-type dataset --local-dir $HJEPA_HOME
unzstd $HJEPA_HOME/pusht_expert_train.h5.zst
tar -I unzstd -xf $HJEPA_HOME/cube_single_expert.tar.zst -C $HJEPA_HOME
python scripts/data/add_pusht_block_ori.py $HJEPA_HOME/pusht_expert_train.h5
python scripts/data/split_cube.py
```

The scripts add Push-T's probing column `block_ori` (block-angle `[cos, sin]`; already in validation)
and split Cube into the first 9,800 training / last 200 validation episodes. The archives and unsplit Cube file can then be
removed. Push-T validation was converted from DINO-WM with `scripts/data/convert_pusht_noise_to_h5.py`.

**Visual AntMaze and FourRoom Distractors.** Download the eight files from
[`jepa-world-models/h-jepa`](https://huggingface.co/datasets/jepa-world-models/h-jepa):

```bash
hf download jepa-world-models/h-jepa --repo-type dataset --local-dir $HJEPA_HOME \
  --include "visual_antmaze_medium_*" --include "fourroom_tp35_d1*" --include SHA256SUMS
(cd "$HJEPA_HOME" && sha256sum -c --ignore-missing SHA256SUMS)
```

<details>
<summary>Optional: collect AntMaze and FourRoom datasets</summary>

From the repository root:

```bash
bash scripts/data/collect_datasets.sh
```

Uses the matching configs in `scripts/data/config/`. AntMaze requires the
[OGBench](https://github.com/seohongpark/ogbench) ant expert policy in `$HJEPA_HOME/ogbench_experts/ant/`:

```bash
mkdir -p $HJEPA_HOME/ogbench_experts
wget -qO- https://rail.eecs.berkeley.edu/datasets/ogbench/experts.tar.gz \
  | tar xz -C $HJEPA_HOME/ogbench_experts --strip-components=1 experts/ant
```

FourRoom collection reproduced the downloads in our checks. AntMaze follows the same distribution
but is not byte-identical because some reset randomness is unseeded. Its downloaded training set
contains 12,500 explore + 12,500 stitch episodes (the seed-42 paper training data), with only training
columns; collected files include extra columns. Probing evaluation uses 155 explore + 155 stitch episodes.

</details>

### 4.2) Evaluation tasks

Each environment has a fixed set of 50 start/goal tasks under `h_jepa/assets/eval_trajs/`.
From `h_jepa/`, download them from
[`jepa-world-models/h-jepa`](https://huggingface.co/datasets/jepa-world-models/h-jepa):

```bash
hf download jepa-world-models/h-jepa --repo-type dataset --local-dir assets --include "eval_trajs/*"
(cd assets/eval_trajs && sha256sum -c SHA256SUMS)
```

<details>
<summary>Optional: regenerate evaluation tasks</summary>

```bash
# Visual AntMaze: start/goal cells 3 grid cells apart, reached by the expert policy
# (needs the OGBench ant expert in $HJEPA_HOME/ogbench_experts/ant/, see 4.1)
python scripts/generate_maze_expert_grid_eval_tasks.py --config-name ant_flat \
  --output-path assets/eval_trajs/ant/expert_grid_d3_n50.pt \
  --d-low 3 --d-high 3 --num-episodes 50 --rollout-budget 125 --seed 42

# FourRoom Distractors: writes one file per active-distractor count; the evals use _d1
# (no dataset needed)
python scripts/generate_fourroom_eval_tasks.py \
  --data-config-path ../scripts/data/config/fourroom_tp35_d0to5.yaml \
  --output-dir assets/eval_trajs/fourroom --output-stem fourroom_tp35 \
  --cross-n-rooms 2 --max-steps 75 --num-episodes 50

# OGBench Cube: 20-step windows centred on the grasp, from the val split
# (needs cube_single_expert_val.h5, see 4.1)
python scripts/generate_dataset_eval_trajs.py --config-name cube_flat \
  --dataset-name cube_single_expert_val --traj-sampling-mode cube_pickup_centered \
  --goal-offset-steps 20 --eval-budget 50 --seed 42 \
  --output-path assets/eval_trajs/ogbench/goal_offset_20_pickup_val.pt

# Push-T: 75-step windows from the val split, stratified over episodes
# (needs pusht_expert_val.h5, see 4.1)
python eval.py --config-name pusht_flat policy=random load_eval_trajs_path=null \
  eval.goal_offset_steps=75 eval.dataset_name=pusht_expert_val seed=42 \
  dump_eval_trajs_path=assets/eval_trajs/pusht/goal_offset_75_val.pt
```

</details>

### 4.3) Training

Train all 4 environments × 7 models × 3 seeds (84 runs):

```bash
scripts/train_all.sh
```

Runs are sequential. Filter with `ENVS`, `MODELS`, `SEEDS`, e.g.
`ENVS=cube MODELS="lewm hjepa_l3" SEEDS=42 scripts/train_all.sh`, or submit each training command
as a separate cluster job. Outputs follow §3.1.

### 4.4) Pretrained checkpoints (optional)

To skip training (§4.3), download all 84 trained models (4 environments × 7 models × 3 seeds, 9.7 GB) from
[`jepa-world-models/h-jepa`](https://huggingface.co/jepa-world-models/h-jepa)
(the DROID models are in §5.3):

```bash
hf download jepa-world-models/h-jepa --local-dir $HJEPA_HOME/ckpts --exclude "droid/*"
# one environment only: --include "cube/*" instead of --exclude; one seed: add --include "*/seed42/*"
```

Checkpoints and `normalizer.pt` use the training layout (§3.1), ready for the eval scripts below.

### 4.5) Depth comparison (Figure 6)

Training runs this eval automatically; to evaluate saved checkpoints:

```bash
scripts/eval_depth.sh
```

Uses `flat` for LeWM and `l<n>` for n-level H-JEPA/HWM, writing `eval_<planner>/metrics.yaml`
next to each checkpoint. Both this script and the cost-ladder script accept `ENVS` and `SEEDS` filters.

| | LeWM | H-JEPA 2 | H-JEPA 3 | H-JEPA 4 | HWM 2 | HWM 3 | HWM 4 |
|---|---|---|---|---|---|---|---|
| Visual AntMaze | 18.0 ± 3.5 | 39.3 ± 3.7 | 73.3 ± 3.5 | 67.3 ± 2.4 | 34.0 ± 1.2 | 38.7 ± 3.5 | 14.0 ± 2.0 |
| FourRoom Distractors | 40.7 ± 6.8 | 81.3 ± 6.7 | 96.0 ± 1.1 | 88.0 ± 2.3 | 75.3 ± 3.3 | 99.3 ± 0.7 | 92.7 ± 4.4 |
| OGBench Cube | 32.0 ± 6.1 | 47.3 ± 2.7 | 60.0 ± 5.0 | 62.7 ± 2.4 | 46.7 ± 2.9 | 53.3 ± 5.2 | 34.0 ± 3.1 |
| Push-T | 40.0 ± 2.3 | 45.3 ± 1.3 | 17.3 ± 3.5 | 0.7 ± 0.7 | 42.0 ± 6.4 | 12.0 ± 2.0 | 0.0 ± 0.0 |

Planner sample counts maximize mean success under 100 TFLOPs/episode. Cube's levels above 2 have
horizon 1 and are skipped. Push-T had no four-level compute sweep; `pusht_l4` keeps its original setting.

### 4.6) Cost ladder (Tables 1 and 9)

Evaluate each H-JEPA model with level-1 planning and either native or projected cost:

```bash
scripts/eval_cost_ladder.sh
```

Results go beside each checkpoint in `eval_flat/metrics.yaml` (native L1) and
`eval_l<k>_project/metrics.yaml` (level-k cost, k = 2 … model depth).

| | Model | Native L1 | L2 cost | L3 cost | L4 cost |
|---|---|---|---|---|---|
| Visual AntMaze | 2 levels | 23.3 ± 3.3 | 31.3 ± 2.7 | | |
| | 3 levels | 16.7 ± 1.8 | 20.7 ± 4.8 | 22.0 ± 2.3 | |
| | 4 levels | 10.0 ± 0.0 | 20.7 ± 1.8 | 21.3 ± 4.7 | 26.7 ± 2.9 |
| FourRoom Distractors | 2 levels | 9.3 ± 0.7 | 38.7 ± 6.6 | | |
| | 3 levels | 43.3 ± 8.7 | 65.3 ± 5.5 | 71.3 ± 4.1 | |
| | 4 levels | 68.7 ± 4.7 | 72.0 ± 3.1 | 66.0 ± 3.1 | 70.7 ± 2.7 |
| OGBench Cube | 2 levels | 24.0 ± 1.2 | 13.3 ± 0.7 | | |
| | 3 levels | 24.7 ± 1.3 | 28.0 ± 4.2 | 14.0 ± 3.5 | |
| | 4 levels | 26.7 ± 2.7 | 25.3 ± 2.4 | 12.0 ± 2.0 | 12.0 ± 2.3 |
| Push-T | 2 levels | 36.7 ± 1.8 | 38.0 ± 3.1 | | |
| | 3 levels | 30.7 ± 1.8 | 28.0 ± 4.0 | 24.7 ± 4.1 | |
| | 4 levels | 1.3 ± 0.7 | 1.3 ± 0.7 | 0.7 ± 0.7 | 1.3 ± 1.3 |

## 5) Reproducing DROID

Reproduces Figure 9(b) and the planner ladder of Figure 9(c).
Train seeds are 1, 1000, 10000; each model is planned with planner seeds 1, 2, 3.
**For evaluation only, download the [evaluation clips](#51-data) (252 MB) and
[checkpoints](#53-pretrained-checkpoints-optional); the 91 GB training set is needed for training only.**

### 5.1) Data

The DROID 1.0.1 data (CC BY 4.0, see the
[dataset card](https://huggingface.co/datasets/jepa-world-models/h-jepa)) goes to `$HJEPA_HOME/droid/`.

**Evaluation clips.** The 16 evaluation episodes, raw 1280×720 in `droid_raw/1.0.1/`:

```bash
hf download jepa-world-models/h-jepa --repo-type dataset --local-dir $HJEPA_HOME \
  --include droid/droid_raw_eval16.tar --include droid/SHA256SUMS
(cd $HJEPA_HOME/droid && sha256sum -c --ignore-missing SHA256SUMS && tar -xf droid_raw_eval16.tar)
```

`h_jepa/droid_assets/` provides the clip manifest `droid_clips_waypoint_curated16v2_5fps_gw36.json`
(16 clips × 37 frames, goal at step 36) and `norm_stats_droid.json` (action/proprio stats, key `full_fps5`).

**Training data.** 91 GB of tar shards, re-encoded at 256×256 in `droid_256p/1.0.1/`:

```bash
hf download jepa-world-models/h-jepa --repo-type dataset --local-dir $HJEPA_HOME --include "droid/*"
bash $HJEPA_HOME/droid/extract.sh    # checks SHA256SUMS, untars all shards; --delete drops the tars
```

- `droid_paths_minus16_256p.csv`: 74,896 training episodes, all of DROID 1.0.1 minus the 16 eval clips.
- `droid_val_indist_256p.csv`: 64 validation episodes.

Both list episode directories relative to `$HJEPA_HOME/droid` (decoded with `decord`, `droid_data.py`).
Set them with `data.dataset.name` / `data.dataset.val_name` in `config/train/base/droid.yaml`;
absolute paths also work.

Models run at 5 fps (`data.fps: 5`, 0.2 s/step). DROID mp4s are tagged 60 fps but hold 15 Hz footage,
so the loader keeps every third frame.

### 5.2) Training

Train 3 models × 3 seeds (9 runs):

```bash
ENVS=droid scripts/train_all.sh
```

| Config | Model | GPUs × batch | Epochs |
|---|---|---|---|
| `droid_lewm` | flat LeWM + IDM | 2 × 128 | 100 |
| `droid_hwm_l2` | HWM: identity level 2, trained end-to-end | 2 × 128 | 100 |
| `droid_hjepa_l2` | H-JEPA: latent-MLP level 2, trained end-to-end | 2 × 128 | 100 |

Each epoch is 292 steps. Run inside a SLURM allocation with 2 GPUs and 2 tasks per node:
the script uses `srun --ntasks-per-node=2 python main_hjepa.py --config-name droid_<model> seed=<seed>`.
Alternatively, submit each command separately. `MODELS` and `SEEDS` filter runs.

Outputs follow §3.1, with snapshots `droid_<model>_epoch_<N>_object.ckpt` every `save_every_n_epochs`.

### 5.3) Pretrained checkpoints (optional)

To skip training (§5.2), download the 9 trained models (3 models × 3 seeds, 2.7 GB) from
[`jepa-world-models/h-jepa`](https://huggingface.co/jepa-world-models/h-jepa):

```bash
hf download jepa-world-models/h-jepa --local-dir $HJEPA_HOME/ckpts --include "droid/*"
# one model only: --include "droid/droid_hjepa_l2/*"; one seed: --include "droid/*/seed1/*"
```

`droid_<model>_object.ckpt` (the epoch-100 model) and `normalizer.pt` use the training layout (§5.2),
without the epoch snapshots, ready for the eval script below.

### 5.4) Planning evaluation

Plan each clip start → goal in one open-loop call and score against ground-truth actions.
Evaluate all models with planner seeds 1, 2, 3, writing `eval_{flat,l2}/plan_seed<ps>/eval.csv` beside
each checkpoint and printing mean ± SE per model (filter with `MODELS`, `SEEDS`, `PLAN_SEEDS`):

```bash
scripts/eval_droid.sh
```

One checkpoint and planner seed (`droid_flat` for LeWM, `droid_l2` for HWM and H-JEPA):

```bash
python eval.py --config-name droid_l2 seed=<planner seed> output.dir=<dir> \
  policy=$HJEPA_HOME/ckpts/droid/droid_hjepa_l2/seed<seed>/droid_hjepa_l2_object.ckpt
```

Settings in `config/eval/droid_{flat,l2}.yaml`: AdamW, 90 iterations, early stopping after 30
(patience 5 at 1%), weight decay 0.01, initial std 1.5, normalized actions clipped to ±2σ, horizon 36.
Level 2 plans 12 steps/subgoals; level 1 weights intermediate subgoal costs by β = 0.5.
Override sample counts with `solver.num_samples=<S>` (flat) or
`solver.solvers.level<k>.num_samples=<S>` (hierarchical).

Fréchet fidelity = 1 − F(planned, GT) / F(zero motion, GT), where F is discrete Fréchet distance
between cumulative xyz paths, averaged over 16 clips. Outputs: `plan_config.yaml`, per-clip
`ep_<k>/{actions.pt,planning_compute.json}`, and `eval.csv` (`frechet/skill_mean`).
`output.dir` is relative to the checkpoint directory; shards (`start_index=k num_eval=1`) can share it,
with metrics aggregated over saved clips.

| Model | Checkpoint | Config | S | η (level 1, level 2) | planner TFLOPs / episode |
|---|---|---|---|---|---|
| LeWM + IDM | `droid_lewm_object.ckpt` | `droid_flat` | 32 | 0.01 | 14.5 |
| HWM | `droid_hwm_l2_object.ckpt` | `droid_l2` | 16, 4 | 0.01, 0.3 | 10.0 |
| H-JEPA | `droid_hjepa_l2_object.ckpt` | `droid_l2` | 16, 4 | 0.01, 0.3 | 10.1 |

For the compute ladder, sweep only sample counts (per level for hierarchical models).

| Model | Fréchet fidelity (%) |
|---|---|
| LeWM + IDM | 33.57 ± 1.52 |
| HWM | 35.65 ± 2.22 |
| H-JEPA | 38.42 ± 0.94 |

Mean ± SE over train seeds, each averaged over planner seeds, with the planner cells above
(10–15 TFLOPs/episode). The LeWM bar without IDM in Figure 9(b) is the zero-action floor.

## 6) Citation

```bibtex
@article{zhang2026hjepa,
  title   = {{H-JEPA}: End-to-End Learning of Hierarchical World Models for Visual Planning},
  author  = {Zhang, Wancong and Terver, Basile and Rabbat, Mike and LeCun, Yann and Balestriero, Randall},
  journal = {arXiv preprint arXiv:2610.06805},
  year    = {2026}
}
```

## 7) License

MIT (see `LICENSE`).
