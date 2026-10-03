# Provenance: which paper runs back the release

Paths are relative to `$HJEPA_HOME/ckpts`.
Run ids like `9-6-1/1` are `<sweep id>/<grid cell>`; seeds 42/43/44 live in `seed<N>/`.
`.agents/skills/release-check/scripts/runs.py` holds the same ledger in code, and `verify_train.py` /
`verify_eval.py` check the configs against it.

## Training configs (`config/train/<env>_<model>.yaml`)

The release configs were generated from each run's saved `config.yaml` (seed 42), then stripped of
keys whose features were removed. H-JEPA/HWM configs compose to exactly the paper configs for all
three seeds; LeWM configs differ only in behavior-neutral keys (listed in the release-check skill).

| env | LeWM | H-JEPA 2 / 3 / 4 | HWM 2 / 3 / 4 |
|---|---|---|---|
| ant | `ant/7-28-2/ant_level1_union_ds_seed<N>__stage1` | `ant/9-6-3/1`, `9-6-1/1`, `9-6-2/1` | `ant/9-12-5/0`, `9-12-6/0`, `9-12-7/0` |
| fourroom | `fourroom_distractors/7-23-2/0` | `fourroom_distractors/9-11-1/4`, `9-11-2/4`, `9-11-3/4` | `fourroom_distractors/9-12-5/0`, `9-12-6/0`, `9-12-7/0` |
| cube | `ogb/7-28-1/ogb_level1_seed<N>__stage1` | `ogb/9-10-1/0`, `9-10-2/0`, `9-12-8/0` | `ogb/9-12-5/0`, `9-12-6/0`, `9-12-7/0` |
| pusht | `pusht/7-28-1/pusht_level1_seed<N>__stage1` | `pusht/9-9-1/0`, `9-9-2/0`, `9-9-3/0` | `pusht/9-12-5/0`, `9-12-6/0`, `9-12-7/0` |

What the grid cells fixed (now the config defaults):
- Ant H-JEPA `9-6-*/1`: abstract-level predictor depth 5 / heads 12 / mlp 1536 / hidden 224 (10.25M).
- FourRoom H-JEPA `9-11-*/4`: uniform SIGReg weight 0.18 at every level.
- Cube `9-10-*/0` and Push-T `9-9-*/0`: abstract predictor depth 6 / heads 16 / mlp 2048 / hidden 192,
  default epochs. Cube 4 levels `9-12-8/0` and all HWM runs had no grid.
- LeWM: Ant/Cube/Push-T are the level-1 stage of the stagewise runs (batch-capped:
  `max_train_batches_total` 498912 / 191722 / 206982, 50-epoch cap); FourRoom is cell 0 of the
  level-1 SIGReg sweep `7-23-2` (weight 0.72, 11 epochs).

## Planning configs (`config/eval/<env>_<planner>.yaml`)

Generated from the saved `eval_config.yaml` of the paper evals, with paths made portable.

- `flat`, `l2`, `l3`, `l4`: the best mean success under 100 TFLOPs/episode in the `compute_depth`
  sample-count sweeps (the selection `depth_bars.py` makes for the depth figure). HWM evals used the
  same planners as H-JEPA (their eval configs are identical; HWM results in
  `planning_evals/9-16-1`, `9-16-2`, `9-20-1`).

| env | flat | l2 | l3 | l4 |
|---|---|---|---|---|
| ant | 100 | 25, 200 | 100, 100, 400 | 50, 200, 25, 50 |
| fourroom | 200 | 100, 800 | 50, 200, 200 | 50, 200, 200, 400 |
| cube | 800 | 400, 100 | 25, 800, 800 | 200, 50, 800, 800 |
| pusht | 800 | 50, 400 | 25, 50, 200 | 200, 800, 800, 800 |

  Samples per level, level 1 first. Source runs: `<env>/compute_depth/9-13-*/<cell>` and
  `9-19-1/<cell>` (the map is in `verify_eval.py`). `pusht_l4` has no compute sweep: it is the
  in-training eval config of `pusht/9-9-3/0` (FLOPs/episode unknown). Cube `l3`/`l4` keep levels 3+
  at horizon 1, so those solves are skipped (2-level planning of a deeper model), as in the paper.
- `l2_project`, `l3_project`, `l4_project`: the four-level configs of the cost-ladder evals
  (`<env>/planning_evals/9-16-3/l<k>_project`). The 2- and 3-level cost-ladder evals (`9-16-4`) used
  copies without the unused upper levels; the release drops levels above a model's depth instead
  (`planning_eval._drop_levels_above_model`), which is equivalent.
- The cost-ladder native level-1 column used exactly the `flat` config.

## Deviations from the paper protocol

- Planning at the end of training uses planner seed = model seed (`PlanningEvalCallback`),
  matching the paper's standalone paired-seed evals. The paper's in-training evals (only `pusht_l4`
  and the Push-T/HWM 4-level points come from those) used planner seed 42 for every model.
- Push-T, Ant and FourRoom planning evals are not bit-reproducible run to run, even with the paper
  code; Cube is.
- The four LeWM paper runs used the dataset's `precompute_levels=true` loader (level-1 crop in the
  dataset); the release only has the level-1-stream loader (crop in the model). For one level the two
  are equivalent up to RNG draws.

## Reference numbers

The README tables come from the paper-figure data in the development repo
(`stable-wm-lejepa`, branch `hjepa_6_24`, `lejepa_code/paper_plots/data/`):
- depth row: `<env>/compute_depth/compute_pareto.csv` (best under 100 TFLOPs), HWM from
  `<env>/hwm_matched/planning_sr.csv`, Push-T 4 levels from `pusht/depth/{planning_sr,hwm_planning_sr}.csv`;
- cost ladder: `codex/converged_results/<env>/9-16-cost-ladder-depth-tasks/{2,3,4}levels/canonical_metrics.csv`
  (git-ignored in the development repo, on disk only).

## Data

- Release dataset names = paper names without `_2_5x` (AntMaze `stitch_val`, `probing_train`) and
  without the FourRoom `fourroom_7_21/tp35/` dirs; the paper files keep the old names under
  `/mnt/vast/home/kevin/stable-wm-lejepa/datasets` (the release-check scripts map old to new).
  Beware: that dir also holds an unrelated older `visual_antmaze_medium_stitch_val.h5` (1x), so the
  release configs must not be run against it with `HJEPA_HOME` pointing there.
- AntMaze / FourRoom collection configs in `scripts/data/config/` (one per release file, same name)
  are the sidecar configs saved next to each paper dataset (`$HJEPA_HOME/*.yaml`,
  `$HJEPA_HOME/fourroom_7_21/tp35/*.collection.yaml`) with portable paths, except
  `visual_antmaze_medium_explore_stitch_train` and `visual_antmaze_medium_probing_eval`, which each
  replace two paper sets. `scripts/data/collect_datasets.sh` runs all eight.
- Ant training set: the paper runs mixed `visual_antmaze_medium_explore_train` (all 12,500 episodes)
  and `visual_antmaze_medium_stitch_train_2_5x` (12,500 of 31,250 episodes, drawn with
  `subset_seed = seed`) on the fly. `visual_antmaze_medium_explore_stitch_train.h5` holds the seed-42
  draw (training columns only), so it is exactly the seed-42 models' training data; seeds 43/44
  trained on different stitch episodes. The stitch set had no sidecar, so the stitch seed in its
  collection config is a placeholder.
- Ant probing eval set: the paper scored probes on `visual_antmaze_medium_probing_eval_{explore,stitch}_2_5x`
  separately and averaged the two NMSEs. The release scores one concatenated file
  (`visual_antmaze_medium_probing_eval`), so Ant probing numbers no longer match the paper exactly.
- Eval tasks: `h_jepa/assets/eval_trajs/{ant/expert_grid_d3_n50, fourroom/fourroom_tp35_cross2_goal75_n50_d1,
  ogbench/goal_offset_20_pickup_val, pusht/goal_offset_75_val}.pt`, the files the paper evals loaded.

## DROID

The release DROID models are trained with this repo (`scripts/slurm/launch.py train`), three seeds each
(1, 1000, 10000). Paths are relative to `$HJEPA_HOME/ckpts/droid`.

### Training configs (`config/train/droid_<model>.yaml`)

| config | run | checkpoint |
|---|---|---|
| `droid_lewm` | `droid_lewm_w1_2026-09-30_21-08/droid_lewm/seed<N>` | e-100 (final) |
| `droid_hwm_l2` | `droid_hwm_l2_l2knobs_2026-09-30_21-08/l2asc0.005/seed<N>` | e-100 (final) |
| `droid_hjepa_l2` | `droid_hjepa_l2_l2knobs_2026-09-30_21-08/l2asc0.005/seed<N>` | e-100 (final) |

- Recipe shared by the three configs: 100 epochs, prediction loss weight 1, native teacher forcing
  (history 7, rollout 1), target not detached; level-2 action SIGReg coefficient 0.005 for HWM and H-JEPA.
- Data: `droid_paths_minus16_256p.csv` (74,896 episodes) for training, `droid_val_indist_256p.csv` for
  monitoring. Assets in `h_jepa/droid_assets/`: norm stats `full_fps5`, clip manifest
  `droid_clips_waypoint_curated16v2_5fps_gw36.json` (16 clips x 37 frames, horizon 36).
- 5 fps (`data.fps: 5`): DROID mp4s are tagged 60 fps for 15 Hz footage, so the stride is 3 frames.

### Planning configs (`config/eval/droid_{flat,l2}.yaml`)

Paper cells are `eval_<plan>/epoch_<N>/` cells of `scripts/slurm/launch.py eval` with GD ladder `ns{S}_lr{η}[_l2lr{η2}]`:

| model | planner | S | η (level 1, level 2) | cell |
|---|---|---|---|---|
| LeWM + IDM | `droid_flat` | 16 | 0.01 | `ns16_lr0p01` |
| HWM | `droid_l2` | 16 | 0.01, 0.1 | `ns16_lr0p01_l2lr0p1` |
| H-JEPA | `droid_l2` | 4 | 0.03, 0.01 | `ns4_lr0p03_l2lr0p01` |

`fig:compute-pareto-real` uses the same runs with the other `ns{S}_lr{η}[_l2lr{η2}]` cells.

### Reference numbers

Paper Fréchet fidelity (%, mean ± SE over train seeds): LeWM + IDM 34.06 ± 1.26, HWM 34.96 ± 0.32,
H-JEPA 39.95 ± 2.91, measured on the original-code runs. The release runs above are being re-evaluated;
fill their numbers in the README "This release" column. The LeWM bar without IDM is the zero-action
floor, not a trained model.

### Not ported

Decoded-plans figure (needs the visual decoder), anticollapse 16-cell grid, crossval grids,
`tab:sf-idm0`, varcomp, selective-bars, and the TFLOPs measurement of `fig:compute-pareto-real`
(`FlopCounterMode` in the original code). See `release_todo.md`.
