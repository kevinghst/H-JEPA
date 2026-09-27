# H-JEPA: Hierarchical JEPA World Models

Code to reproduce the planning results of the H-JEPA paper:

- the bottom row of the depth figure (`planning/depth_planning_combined`): planning success of flat
  LeWM, H-JEPA and HWM with 2, 3 and 4 levels on Visual AntMaze, FourRoom Distractors, OGBench Cube and
  Push-T, all with planners under 100 TFLOPs per episode;
- the level-1 planning columns of the cost-ladder tables (`tab:cost-ladder`, `tab:cost-ladder-appendix`):
  the H-JEPA level-1 planner with its cost measured in the level-2, 3 or 4 latent.

The code builds on [LeWM](https://github.com/lucas-maes/le-wm) and
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel), which is vendored in
`stable_worldmodel/` with the environments, solvers and planning policy used here.

## Layout

```
stable_worldmodel/          environments, dataset loader, planning solvers and policy
scripts/data/               dataset collection (AntMaze, FourRoom) and Push-T utilities
h_jepa/
  main_hjepa.py             training (ends with a planning eval and a probing/decoding eval)
  eval.py                   standalone planning eval
  main_probing_decoding_eval.py   standalone probing/decoding eval
  models/                   JEPA levels, H-JEPA container, encoders, predictors
  config/train/             28 training configs: <env>_<model>.yaml (+ base/<env>.yaml)
  config/eval/              28 planning configs: <env>_<planner>.yaml
  config/probing/           final probing/decoding configs, one per environment
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

## Installation

```bash
git clone https://github.com/kevinghst/H-JEPA.git && cd H-JEPA
conda create -n hjepa python=3.10 && conda activate hjepa
pip install -e ".[train,env]"
export STABLEWM_HOME=/path/to/data   # datasets, expert policies and checkpoints live here
```

All commands below run from `h_jepa/` unless noted.

## Datasets

All datasets are HDF5 files under `$STABLEWM_HOME`.

| Environment | Files | Source |
|---|---|---|
| Push-T | `pusht_expert_train.h5`, `pusht_expert_val.h5` | download (LeWM) |
| OGBench Cube | `cube_single_expert_train.h5`, `cube_single_expert_val.h5` | download (LeWM) |
| Visual AntMaze | `visual_antmaze_medium_{explore_train,stitch_train_2_5x,stitch_val_2_5x}.h5`, `visual_antmaze_medium_probing_{train_2_5x,eval_explore_2_5x,eval_stitch_2_5x}.h5` | generated |
| FourRoom Distractors | `fourroom_7_21/tp35/fourroom_tp35_d1{,_val,_probing,_probing_val}.h5` | generated |

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
for name in explore_train stitch_train_2_5x stitch_val_2_5x \
            probing_train_2_5x probing_eval_explore_2_5x probing_eval_stitch_2_5x; do
  python scripts/data/collect_antmaze.py --config-name visual_antmaze_medium_$name
done
```

Training mixes the explore and stitch sets 50/50 on the fly (`data.dataset.sources` in
`config/train/base/ant.yaml`).

**FourRoom Distractors.** From the repository root:

```bash
for name in fourroom_tp35_d1 fourroom_tp35_d1_val fourroom_tp35_d1_probing fourroom_tp35_d1_probing_val; do
  python scripts/data/collect_fourroom_distractors.py --config-name $name
done
```

Each config under `scripts/data/config/` is the collection config saved next to the dataset used in
the paper.

## Evaluation tasks

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

## Training

```bash
python main_hjepa.py --config-name <env>_<model> seed=<seed>
```

The paper uses seeds 42, 43 and 44. A run writes to `$STABLEWM_HOME/ckpts/<env>_<model>/seed<seed>/`:

- `<env>_<model>_object.ckpt`: the trained model;
- `planning_eval/epoch_XXXX/metrics.yaml`: the planning eval run at the end of training with the
  matching planner (`<env>_flat` for LeWM, `<env>_l<n>` for an n-level model), planner seed = model seed;
- `final_probing_decoding_eval/`: probes and decoders trained on the frozen model
  (`config/probing/<env>.yaml`).

`scripts/train_all.sh` trains all 84 runs sequentially (restrict with `ENVS`, `MODELS`, `SEEDS`); on a
cluster, submit each command as its own job. W&B logging is off by default (`wandb.enabled=true` to turn
it on).

## Planning: depth figure

The planning eval at the end of training already produces these numbers. To re-run it from saved
checkpoints:

```bash
python eval.py --config-name <env>_<planner> seed=<seed> output.dir=eval_<planner> \
  policy=$STABLEWM_HOME/ckpts/<env>_<model>/seed<seed>/<env>_<model>_object.ckpt
```

with `flat` for `lewm` and `l<n>` for `hjepa_l<n>` and `hwm_l<n>`. `scripts/eval_depth.sh` runs all of
them. Each eval writes `metrics.yaml` (`success_rate`) to `output.dir`, relative to the checkpoint's directory.

Reference success rates (%, mean ± SE over three seeds):

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

## Planning: cost ladder

The H-JEPA level-1 planner with its cost measured in the latent of level 2, 3 or 4 (the upper levels
are skipped; the planned level-1 states and the goal are encoded up to the target level):

```bash
python eval.py --config-name <env>_l<k>_project seed=<seed> output.dir=eval_l<k>_project \
  policy=$STABLEWM_HOME/ckpts/<env>_hjepa_l<n>/seed<seed>/<env>_hjepa_l<n>_object.ckpt
```

for every n-level H-JEPA model and 2 ≤ k ≤ n. The same config serves 2-, 3- and 4-level models: planning
levels above the model's depth are dropped. The native level-1 column is `<env>_flat` on the same model.
`scripts/eval_cost_ladder.sh` runs everything, including the native column.

Reference success rates (%, mean ± SE over three seeds):

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

## Probing and decoding

Training ends with `final_probing_decoding_eval`. To run it on a saved checkpoint:

```bash
python main_probing_decoding_eval.py --config-name <env> \
  policy=$STABLEWM_HOME/ckpts/<env>_<model>/seed<seed>/<env>_<model>_object.ckpt
```

## License

MIT (see `LICENSE`).
