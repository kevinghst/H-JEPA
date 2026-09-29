# H-JEPA: Hierarchical JEPA World Models

Code to reproduce the planning results of the H-JEPA paper:

- the bottom row of the depth figure (`planning/depth_planning_combined`): planning success of flat
  LeWM, H-JEPA and HWM with 2, 3 and 4 levels on Visual AntMaze, FourRoom Distractors, OGBench Cube and
  Push-T, all with planners under 100 TFLOPs per episode;
- the level-1 planning columns of the cost-ladder tables (`tab:cost-ladder`, `tab:cost-ladder-appendix`):
  the H-JEPA level-1 planner with its cost measured in the level-2, 3 or 4 latent;
- the DROID planning figures (`fig:cls-ladder-droid`, `fig:compute-pareto-real`): Fréchet fidelity of
  LeWM + IDM, HWM and H-JEPA with 2 levels on DROID at 5 fps.

The code builds on [LeWM](https://github.com/lucas-maes/le-wm) and
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel), which is vendored in
`stable_worldmodel/` with the environments, solvers and planning policy used here.

## 1) Layout

```
stable_worldmodel/          environments, dataset loader, planning solvers and policy
scripts/data/               dataset collection (AntMaze, FourRoom) and Push-T utilities
h_jepa/
  main_hjepa.py             training (ends with a planning eval and a probing/decoding eval)
  eval.py                   standalone planning eval
  main_probing_decoding_eval.py   standalone probing/decoding eval
  models/                   JEPA levels, H-JEPA container, encoders, predictors
  config/train/             28 training configs: <env>_<model>.yaml (+ base/<env>.yaml),
                            3 DROID configs: droid_{lewm,hwm_l2,hjepa_l2}.yaml
  config/eval/              28 planning configs: <env>_<planner>.yaml, DROID: droid_{flat,l2}.yaml
  config/probing/           final probing/decoding configs, one per environment
  droid_data.py             DROID mp4 loader (training) and evaluation-clip reader
  droid_plan_eval.py        offline DROID planning eval on the 16 evaluation clips
  droid_assets/             DROID normalization stats and evaluation-clip manifest
  scripts/                  eval-task generators and the train/eval helper scripts
  ARCHITECTURE.md           how training, the hierarchy and planning fit together
```

`<env>` is `ant`, `fourroom`, `cube` or `pusht`.

| Model (`config/train/<env>_<model>.yaml`) | |
|---|---|
| `lewm` | flat LeWM baseline (one level) |
| `hjepa_l2`, `hjepa_l3`, `hjepa_l4` | H-JEPA with 2/3/4 levels, trained end-to-end |
| `hwm_l2`, `hwm_l3`, `hwm_l4` | HWM baseline: identity upper encoders, no upper-level SIGReg |

| Planner (`config/eval/<env>_<planner>.yaml`) | |
|---|---|
| `flat` | flat level-1 planning (LeWM, and the native level-1 column of the cost ladder) |
| `l2`, `l3`, `l4` | 2/3/4-level hierarchical planning, shared by H-JEPA and HWM |
| `l2_project`, `l3_project`, `l4_project` | level-1 planning with the cost in the level-2/3/4 latent |

## 2) Installation

```bash
git clone https://github.com/kevinghst/H-JEPA.git && cd H-JEPA
conda create -n hjepa python=3.10 && conda activate hjepa
pip install -e ".[train,env]"
export STABLEWM_HOME=/path/to/data   # datasets, expert policies and checkpoints live here
```

## 3) Usage

How to run each part of the pipeline. All commands run from `h_jepa/` unless noted.

### 3.1) Datasets

The simulation datasets are HDF5 files under `$STABLEWM_HOME`; DROID is read from mp4 (§5.1).

| Environment | Files | Source |
|---|---|---|
| Push-T | `pusht_expert_train.h5`, `pusht_expert_val.h5` | download (LeWM) |
| OGBench Cube | `cube_single_expert_train.h5`, `cube_single_expert_val.h5` | download (LeWM) |
| Visual AntMaze | `visual_antmaze_medium_{explore_stitch_train,stitch_val_2_5x}.h5`, `visual_antmaze_medium_probing_{train_2_5x,eval_explore_2_5x,eval_stitch_2_5x}.h5` | generated |
| FourRoom Distractors | `fourroom_7_21/tp35/fourroom_tp35_d1{,_val,_probing,_probing_val}.h5` | generated |
| DROID | `droid_paths_minus16_256p.csv`, `droid_val_indist_256p.csv` and the 256p mp4 episodes they list | download (`<LINK: DROID 256p corpus>`) |

**Push-T and Cube.** Follow the LeWM data instructions: `<LINK: Push-T data>`, `<LINK: Cube data>`.
The Push-T probing config also reads a `block_ori` column (`[cos, sin]` of the block angle). If the
downloaded files lack it, add it in place (from the repository root):

```bash
python scripts/data/add_pusht_block_ori.py $STABLEWM_HOME/pusht_expert_train.h5 $STABLEWM_HOME/pusht_expert_val.h5
```

**Visual AntMaze.** The data is collected by rolling out the OGBench AntMaze expert policies
(`<LINK: OGBench expert policies>`); put the ant expert in `$STABLEWM_HOME/ogbench_experts/ant/`
(`params_400000.pkl`, `flags.json`). From the repository root:

```bash
for name in explore_stitch_train stitch_val_2_5x \
            probing_train_2_5x probing_eval_explore_2_5x probing_eval_stitch_2_5x; do
  python scripts/data/collect_antmaze.py --config-name visual_antmaze_medium_$name
done
```

The training set is half explore, half stitch trajectories (12,500 episodes each). The released
file is the exact data the seed-42 paper models were trained on; the collection config produces an
equivalent file, not an identical one.

**FourRoom Distractors.** From the repository root:

```bash
for name in fourroom_tp35_d1 fourroom_tp35_d1_val fourroom_tp35_d1_probing fourroom_tp35_d1_probing_val; do
  python scripts/data/collect_fourroom_distractors.py --config-name $name
done
```

Each config under `scripts/data/config/` is the collection config saved next to the dataset used in
the paper.

### 3.2) Generate evaluation tasks

Each environment has a fixed set of 50 start/goal tasks under `h_jepa/assets/eval_trajs/`.
Download them (`<LINK: eval tasks>`) or regenerate them:

```bash
# Visual AntMaze: start/goal cells 3 grid cells apart, reached by the expert policy
python scripts/generate_maze_expert_grid_eval_tasks.py --config-name ant_flat \
  --output-path assets/eval_trajs/ant/expert_grid_d3_n50.pt \
  --d-low 3 --d-high 3 --num-episodes 50 --rollout-budget 125 --seed 42

# FourRoom Distractors: writes one file per active-distractor count; the evals use _d1
python scripts/generate_fourroom_eval_tasks.py \
  --data-config-path ../scripts/data/config/fourroom_tp35_d0to5.yaml \
  --output-dir assets/eval_trajs/fourroom --output-stem fourroom_tp35 \
  --cross-n-rooms 2 --max-steps 75 --num-episodes 50

# OGBench Cube: 20-step windows centred on the grasp, from the val split
python scripts/generate_dataset_eval_trajs.py --config-name cube_flat \
  --dataset-name cube_single_expert_val --traj-sampling-mode cube_pickup_centered \
  --goal-offset-steps 20 --eval-budget 50 --seed 42 \
  --output-path assets/eval_trajs/ogbench/goal_offset_20_pickup_val.pt

# Push-T: 75-step windows from the val split, stratified over episodes
python eval.py --config-name pusht_flat policy=random load_eval_trajs_path=null \
  eval.goal_offset_steps=75 eval.dataset_name=pusht_expert_val seed=42 \
  dump_eval_trajs_path=assets/eval_trajs/pusht/goal_offset_75_val.pt
```

### 3.3) Training

```bash
python main_hjepa.py --config-name <env>_<model> seed=<seed>
```

A run writes to `$STABLEWM_HOME/ckpts/<env>_<model>/seed<seed>/`:

- `<env>_<model>_object.ckpt`: the trained model;
- `planning_eval/epoch_XXXX/metrics.yaml`: the planning eval run at the end of training with the
  matching planner (`<env>_flat` for LeWM, `<env>_l<n>` for an n-level model), planner seed = model seed;
- `final_probing_decoding_eval/`: probes and decoders trained on the frozen model
  (`config/probing/<env>.yaml`).

W&B logging is off by default (`wandb.enabled=true` to turn it on).

### 3.4) Probing and decoding

Training ends with `final_probing_decoding_eval`. To run it on a saved checkpoint:

```bash
python main_probing_decoding_eval.py --config-name <env> \
  policy=$STABLEWM_HOME/ckpts/<env>_<model>/seed<seed>/<env>_<model>_object.ckpt
```

### 3.5) Planning evaluation

To run a planner on a saved checkpoint:

```bash
python eval.py --config-name <env>_<planner> seed=<seed> output.dir=<dir> \
  policy=$STABLEWM_HOME/ckpts/<env>_<model>/seed<seed>/<env>_<model>_object.ckpt
```

The eval writes `metrics.yaml` (`success_rate`) to `output.dir`, relative to the checkpoint's directory.
Use `flat` for `lewm` and `l<n>` for `hjepa_l<n>` and `hwm_l<n>`. The `l<k>_project` planners run
level-1 planning with the cost measured in the level-k latent (the upper levels are skipped; the planned
level-1 states and the goal are encoded up to level k); they apply to any H-JEPA model with at least k
levels, since planning levels above the model's depth are dropped.

## 4) Reproducing Fourroom, Visual AntMaze, OGBench Cube, Push-T

This covers the bottom row of the depth figure and the level-1 columns of the cost-ladder tables. The
paper uses seeds 42, 43 and 44 for every model, with the planner seed equal to the model seed. Reference
numbers are success rates (%, mean ± SE over the three seeds).

### 4.1) Training

Train all 4 environments × 7 models × 3 seeds (84 runs):

```bash
scripts/train_all.sh
```

The runs are sequential; restrict them with `ENVS`, `MODELS` and `SEEDS` (e.g.
`ENVS=cube MODELS="lewm hjepa_l3" SEEDS=42 scripts/train_all.sh`), or on a cluster submit each
`main_hjepa.py` command as its own job.

### 4.2) Depth figure

Each training run ends with the planning eval behind the depth figure (flat planning
for LeWM, n-level planning for H-JEPA and HWM with n levels), so the numbers come out as a byproduct of
training:

```
$STABLEWM_HOME/ckpts/
  <env>_<model>/                    <env> in {ant, fourroom, cube, pusht}
    seed<seed>/                     <model> in {lewm, hjepa_l2..4, hwm_l2..4}, <seed> in {42, 43, 44}
      <env>_<model>_object.ckpt
      planning_eval/epoch_XXXX/metrics.yaml    success_rate
      final_probing_decoding_eval/
```

`scripts/eval_depth.sh` re-runs the same evals from the saved checkpoints and writes
`<env>_<model>/seed<seed>/eval_<planner>/metrics.yaml`.

| | LeWM | H-JEPA 2 | H-JEPA 3 | H-JEPA 4 | HWM 2 | HWM 3 | HWM 4 |
|---|---|---|---|---|---|---|---|
| Visual AntMaze | 18.0 ± 3.5 | 39.3 ± 3.7 | 73.3 ± 3.5 | 67.3 ± 2.4 | 34.0 ± 1.2 | 38.7 ± 3.5 | 14.0 ± 2.0 |
| FourRoom Distractors | 40.7 ± 6.8 | 81.3 ± 6.7 | 96.0 ± 1.1 | 88.0 ± 2.3 | 75.3 ± 3.3 | 99.3 ± 0.7 | 92.7 ± 4.4 |
| OGBench Cube | 32.0 ± 6.1 | 47.3 ± 2.7 | 60.0 ± 5.0 | 62.7 ± 2.4 | 46.7 ± 2.9 | 53.3 ± 5.2 | 34.0 ± 3.1 |
| Push-T | 40.0 ± 2.3 | 45.3 ± 1.3 | 17.3 ± 3.5 | 0.7 ± 0.7 | 42.0 ± 6.4 | 12.0 ± 2.0 | 0.0 ± 0.0 |

The planner sample counts are the best mean success rate under 100 TFLOPs/episode from a sweep over
sample counts per level. Cube's 3- and 4-level planners keep the levels above 2 at horizon 1, so those
levels are skipped (2-level planning of a deeper model). No four-level compute sweep was run on Push-T,
so `pusht_l4` keeps the original planner setting.

### 4.3) Cost ladder

Evaluate every H-JEPA model with the flat level-1 planner (the native column) and
with the cost measured in each upper level's latent (`l2_project` … `l<n>_project` for an n-level model):

```bash
scripts/eval_cost_ladder.sh
```

`ENVS` and `SEEDS` restrict it as above. Each eval writes next to the checkpoint:

```
$STABLEWM_HOME/ckpts/
  <env>_hjepa_l<n>/                 <n> in {2, 3, 4}
    seed<seed>/
      eval_flat/metrics.yaml        native L1
      eval_l2_project/metrics.yaml  L2 cost
      eval_l3_project/metrics.yaml  L3 cost (n >= 3)
      eval_l4_project/metrics.yaml  L4 cost (n = 4)
```

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

This covers `fig:cls-ladder-droid` and the planner ladder of `fig:compute-pareto-real`. The paper uses
seeds 1, 1000 and 10000 for every model; each trained model is planned with planner seeds 1, 2 and 3.
Reference numbers are Fréchet fidelity (%, mean ± SE over the three train seeds, each the mean over
the three planner seeds).

### 5.1) Data

DROID episodes are mp4 files decoded with `decord` (`droid_data.py`). Training reads
`droid_paths_minus16_256p.csv`: 74,896 episodes, all of DROID 1.0.1 minus the 16 evaluation clips,
re-encoded at 256p (`<LINK: DROID 256p corpus>`). `droid_val_indist_256p.csv` (64 episodes) is the
validation split used for monitoring. Both CSVs list episode directories by absolute path; set them
with `data.dataset.name` and `data.dataset.val_name` (`config/train/base/droid.yaml`).

`h_jepa/droid_assets/` holds the action/proprio normalization stats (`norm_stats_droid.json`, key
`full_fps5`) and the evaluation-clip manifest `droid_clips_waypoint_curated16v2_5fps_gw36.json`
(16 clips of 37 frames, goal 36 steps after the start), which reads the clips from the raw DROID
1.0.1 release.

The models run at 5 fps (`data.fps: 5`, one step = 0.2 s). DROID mp4s are tagged 60 fps but hold
15 Hz footage, so the loader keeps every 3rd frame.

### 5.2) Training

Train 3 models × 3 seeds (9 runs):

```bash
scripts/train_droid.sh
```

| Config | Model | GPUs × batch | Epochs |
|---|---|---|---|
| `droid_lewm` | flat LeWM + IDM | 4 × 64 | 100 |
| `droid_hwm_l2` | HWM: identity level 2, trained end-to-end | 2 × 128 | 100 |
| `droid_hjepa_l2` | H-JEPA: latent-MLP level 2, trained end-to-end | 2 × 128 | 150 (the paper reads epoch 100) |

An epoch is 292 steps. Each run takes one process per GPU: the script launches
`srun --ntasks-per-node=<GPUs> python main_hjepa.py --config-name droid_<model> seed=<seed> trainer.devices=<GPUs>`,
so run it inside a SLURM allocation with 4 GPUs and 4 tasks per node, or submit each command as its
own job. `MODELS` and `SEEDS` restrict it as in §4.1. A run writes:

```
$STABLEWM_HOME/ckpts/
  droid_<model>/                    <model> in {lewm, hwm_l2, hjepa_l2}
    seed<seed>/                     <seed> in {1, 1000, 10000}
      droid_<model>_object.ckpt     final model
      droid_<model>_epoch_<N>_object.ckpt   snapshot every save_every_n_epochs (H-JEPA: epoch 100)
```

### 5.3) Planning evaluation

`droid_plan_eval.py` plans each evaluation clip start → goal in one open-loop call and scores the
planned actions against the ground-truth ones:

```bash
python droid_plan_eval.py --ckpt <run>/droid_<model>_object.ckpt [--hier] \
  --lr <η1> [--l2-lr <η2>] --num-samples <S> --seed <planner seed> --out <dir>
```

The planner settings are in `config/eval/droid_flat.yaml` (LeWM) and `config/eval/droid_l2.yaml`
(`--hier`, HWM and H-JEPA): AdamW on the actions, 90 iterations with early stopping after 30
(patience 5 at 1%), weight decay 0.01, initial std 1.5, actions clipped to ±2σ in normalized action
space, horizon 36. Level 2 plans 12 steps and passes 12 subgoals to level 1 (weight β = 0.5 on the
intermediate subgoal costs).

The metric is Fréchet fidelity, 1 − F(planned, GT) / F(zero motion, GT), with F the discrete Fréchet
distance between cumulative xyz paths, averaged over the 16 clips. The eval writes, under
`<out>/<tag>/`, `ep_<k>/actions.pt` and `ep_<k>/planning_compute.json` per clip and `eval.csv`
(`frechet/skill_mean`) over every clip present, so one-clip shards (`--start-index k --num-eval 1`)
can share one output dir.

| Model | Checkpoint | Planner | S | η (level 1, level 2) |
|---|---|---|---|---|
| LeWM + IDM | `droid_lewm_object.ckpt` | flat | 16 | 0.01 |
| HWM | `droid_hwm_l2_object.ckpt` | `--hier` | 16 | 0.01, 0.1 |
| H-JEPA | `droid_hjepa_l2_epoch_100_object.ckpt` | `--hier` | 4 | 0.03, 0.01 |

`scripts/eval_droid.sh` runs these cells with planner seeds 1, 2 and 3 on every trained model, writes
`droid_<model>/seed<seed>/eval_{flat,l2}/plan_seed<ps>/eval.csv` and prints the mean ± SE per model.
The ladder of `fig:compute-pareto-real` is the same eval on the same checkpoints with
`--num-samples`, `--lr` and `--l2-lr` swept.

| | Paper | This release |
|---|---|---|
| LeWM + IDM | 34.06 ± 1.26 | |
| HWM | 34.96 ± 0.32 | |
| H-JEPA | 39.95 ± 2.91 | |

The LeWM bar without IDM in `fig:cls-ladder-droid` is the zero-action floor, not a trained model.

## 6) License

MIT (see `LICENSE`).
