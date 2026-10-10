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

Train hierarchical world models and use them for visual planning, with pretrained checkpoints
and evaluation tools across four simulated environments and DROID.
Built on [LeWM](https://github.com/lucas-maes/le-wm) and
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel).

## 1) Installation

```bash
git clone https://github.com/kevinghst/H-JEPA.git && cd H-JEPA
conda create -n hjepa python=3.11.13 && conda activate hjepa
pip install -e ".[train,env]" --extra-index-url https://download.pytorch.org/whl/cu128
export HJEPA_HOME=/path/to/data   # datasets, expert policies and checkpoints live here
```

**Choose your workflow:**

- **[Train and evaluate on Visual AntMaze](#2-run-h-jepa)** — start with a concrete example.
- **[Reproduce the paper's simulation results](#3-reproducing-push-t-ogbench-cube-visual-antmaze-fourroom)**
- **[Reproduce the paper's DROID results](#4-reproducing-droid)**

## 2) Run H-JEPA

Run commands from `h_jepa/` unless noted (`cd h_jepa` after installation).

### 2.1) Training

Train three-level H-JEPA on Visual AntMaze with seed 42. First prepare the
[AntMaze datasets](#31-datasets), including the probing files, and [evaluation tasks](#32-evaluation-tasks).

```bash
python main_hjepa.py --config-name ant_hjepa_l3 seed=42
```

Training configs follow `<env>_<model>`; see the [model/config tables](#5-configuration-reference) for other choices.

Training automatically runs planning evaluation at the end.

Runs write to `$HJEPA_HOME/ckpts/<env>/<env>_<model>/seed<seed>/`:

- `<env>_<model>_object.ckpt`, `config.yaml`, logs and the stable-pretraining cache `spt/`;
- `planning_eval/epoch_XXXX/metrics.yaml`: final planning eval with the matching planner and model seed;
- `final_probing_decoding_eval/`: frozen-model probes and decoders (`config/probing/<env>.yaml`).

### 2.2) Standalone planning evaluation

Use this to evaluate a pretrained checkpoint or rerun evaluation with different planner settings.

Here we evaluate the three-level H-JEPA model from above using three-level hierarchical planning on Visual AntMaze. Prepare the
[evaluation tasks](#32-evaluation-tasks) and either train the model or
[download its checkpoint](#34-pretrained-checkpoints-optional) first.

```bash
python eval.py --config-name ant_l3 seed=42 output.dir=eval_l3 \
  policy="$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt"
```

Results are saved beside the checkpoint in `eval_l3/metrics.yaml`; `success_rate` is reported as a percentage.
Completed tasks are skipped on rerun. Use a new `output.dir` when changing planner settings.

For cluster training and evaluation, see [Running on SLURM](docs/SLURM.md).

## 3) Reproducing Push-T, OGBench Cube, Visual AntMaze, FourRoom

Reproduces the bottom row of Figure 6 and the level-1 columns of Tables 1 and 9.
Results report paper success rates
(%, mean ± SE over seeds 42, 43, 44; planner seed = model seed).
**For evaluation only, download [tasks](#32-evaluation-tasks) and [checkpoints](#34-pretrained-checkpoints-optional);
datasets are needed for training and probing.**

### 3.1) Datasets

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
python scripts/data/add_pusht_block_ori.py "$HJEPA_HOME/pusht_expert_train.h5"  # add block_ori for probing
python scripts/data/split_cube.py  # first 9,800 episodes for training; last 200 for validation
```

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

### 3.2) Evaluation tasks

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
# (needs the OGBench ant expert in $HJEPA_HOME/ogbench_experts/ant/, see 3.1)
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
# (needs cube_single_expert_val.h5, see 3.1)
python scripts/generate_dataset_eval_trajs.py --config-name cube_flat \
  --dataset-name cube_single_expert_val --traj-sampling-mode cube_pickup_centered \
  --goal-offset-steps 20 --eval-budget 50 --seed 42 \
  --output-path assets/eval_trajs/ogbench/goal_offset_20_pickup_val.pt

# Push-T: 75-step windows from the val split, stratified over episodes
# (needs pusht_expert_val.h5, see 3.1)
python eval.py --config-name pusht_flat policy=random load_eval_trajs_path=null \
  eval.goal_offset_steps=75 eval.dataset_name=pusht_expert_val seed=42 \
  dump_eval_trajs_path=assets/eval_trajs/pusht/goal_offset_75_val.pt
```

</details>

### 3.3) Training

Train all 4 environments × 7 models × 3 seeds (84 runs):

```bash
scripts/train_all.sh
```

Runs are sequential. Filter with `ENVS`, `MODELS`, `SEEDS`, e.g.
`ENVS=cube MODELS="lewm hjepa_l3" SEEDS=42 scripts/train_all.sh`, or submit each training command
as a separate cluster job. Outputs follow §2.1.

### 3.4) Pretrained checkpoints (optional)

To skip training (§3.3), download all 84 trained models (4 environments × 7 models × 3 seeds, 9.7 GB) from
[`jepa-world-models/h-jepa`](https://huggingface.co/jepa-world-models/h-jepa)
(the DROID models are in §4.3):

```bash
hf download jepa-world-models/h-jepa --local-dir $HJEPA_HOME/ckpts --exclude "droid/*"
# one environment only: --include "cube/*" instead of --exclude; one seed: add --include "*/seed42/*"
```

Checkpoints and `normalizer.pt` use the training layout (§2.1), ready for the eval scripts below.

### 3.5) Depth comparison (Figure 6)

First complete [training (§3.3)](#33-training) or [download pretrained checkpoints (§3.4)](#34-pretrained-checkpoints-optional),
and prepare the [evaluation tasks (§3.2)](#32-evaluation-tasks). Trained and downloaded checkpoints use the same directory layout expected by this script.

Training runs this eval automatically; to evaluate saved checkpoints:

```bash
scripts/eval_depth.sh
```

Results are saved next to each checkpoint in `eval_<planner>/metrics.yaml`.
Both this script and the cost-ladder script accept `ENVS` and `SEEDS` filters.
Each requires all relevant model checkpoints for the selected environments and seeds; neither accepts a `MODELS` filter.
For a single checkpoint, use [standalone evaluation (§2.2)](#22-standalone-planning-evaluation).

See the [expected results reported in the paper](docs/RESULTS.md#depth-comparison-figure-6).

### 3.6) Cost ladder (Tables 1 and 9)

First complete [training (§3.3)](#33-training) or [download pretrained checkpoints (§3.4)](#34-pretrained-checkpoints-optional),
and prepare the [evaluation tasks (§3.2)](#32-evaluation-tasks).

Evaluate each H-JEPA model with level-1 planning and either native or projected cost:

```bash
scripts/eval_cost_ladder.sh
```

Results go beside each checkpoint in `eval_flat/metrics.yaml` (native L1) and
`eval_l<k>_project/metrics.yaml` (level-k cost, k = 2 … model depth).

See the [expected results reported in the paper](docs/RESULTS.md#cost-ladder-tables-1-and-9).

## 4) Reproducing DROID

Evaluate open-loop planning on real-robot videos from DROID. Plans are scored against recorded expert trajectories;
this evaluation measures trajectory fidelity, not closed-loop task success.

Reproduces Figure 9(b).
**For evaluation only, download the [evaluation clips](#41-data) (252 MB) and
[checkpoints](#43-pretrained-checkpoints-optional); the 91 GB training set is needed for training only.**

### 4.1) Data

Download DROID data to `$HJEPA_HOME/droid/` ([dataset details and license](https://huggingface.co/datasets/jepa-world-models/h-jepa)).

**Evaluation clips.** The 16 evaluation episodes, raw 1280×720 in `droid_raw/1.0.1/`:

```bash
hf download jepa-world-models/h-jepa --repo-type dataset --local-dir $HJEPA_HOME \
  --include droid/droid_raw_eval16.tar --include droid/SHA256SUMS
(cd $HJEPA_HOME/droid && sha256sum -c --ignore-missing SHA256SUMS && tar -xf droid_raw_eval16.tar)
```

**Training data.** 91 GB of tar shards, re-encoded at 256×256 in `droid_256p/1.0.1/`:

```bash
hf download jepa-world-models/h-jepa --repo-type dataset --local-dir $HJEPA_HOME --include "droid/*"
bash $HJEPA_HOME/droid/extract.sh    # checks SHA256SUMS, untars all shards; --delete drops the tars
```

Models train at 5 fps on 74,896 episodes, excluding the 16 evaluation clips, with 64 validation episodes.

### 4.2) Training

Train 3 models × 3 seeds (1, 1000, 10000), for 9 runs:

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

Outputs follow §2.1, with snapshots `droid_<model>_epoch_<N>_object.ckpt` every `save_every_n_epochs`.

### 4.3) Pretrained checkpoints (optional)

To skip training (§4.2), download the 9 trained models (3 models × 3 seeds, 2.7 GB) from
[`jepa-world-models/h-jepa`](https://huggingface.co/jepa-world-models/h-jepa):

```bash
hf download jepa-world-models/h-jepa --local-dir $HJEPA_HOME/ckpts --include "droid/*"
# one model only: --include "droid/droid_hjepa_l2/*"; one seed: --include "droid/*/seed1/*"
```

`droid_<model>_object.ckpt` (the epoch-100 model) and `normalizer.pt` use the training layout (§4.2),
without the epoch snapshots, ready for the eval script below.

### 4.4) Planning evaluation

First complete [training (§4.2)](#42-training) or [download pretrained checkpoints (§4.3)](#43-pretrained-checkpoints-optional),
and prepare the [evaluation clips (§4.1)](#41-data). Trained and downloaded checkpoints use the same directory layout expected by this script.

Plan each clip start → goal in one open-loop call and score against ground-truth actions.
Evaluate all models with planner seeds 1, 2, 3, writing `eval_{flat,l2}/plan_seed<ps>/eval.csv` beside
each checkpoint and printing mean ± SE per model (filter with `MODELS`, `SEEDS`, `PLAN_SEEDS`):

```bash
scripts/eval_droid.sh
```

For standalone evaluation, adapt the [command in §2.2](#22-standalone-planning-evaluation)
to your DROID checkpoint, using `droid_flat` for LeWM or `droid_l2` for HWM and H-JEPA.

Fréchet fidelity measures how closely the planned end-effector path follows the expert trajectory:
0% corresponds to zero motion, and 100% to an exact match. The `frechet/skill_mean` column in `eval.csv`
reports mean fidelity across the 16 clips; multiply by 100 to express it as a percentage.

See the [expected results reported in the paper](docs/RESULTS.md#droid-planning-figure-9b).

## 5) Configuration reference

See [ARCHITECTURE.md](h_jepa/ARCHITECTURE.md) for model and planner internals.
Configs use `<env>` = `ant`, `fourroom`, `cube`, `pusht` or `droid`.
The simulation environments support the options below. DROID supports models `lewm`, `hjepa_l2` and `hwm_l2`,
with planners `flat` for LeWM and `l2` for H-JEPA/HWM.

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
