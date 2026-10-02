# Code release: open items

H-JEPA (this repo) is the source of truth for the release since 2026-09-27. The development repo's
`code_release` branch (`stable-wm-lejepa`, commit `bc777ef`) is frozen; do not port changes back.
Provenance of every config: `.agents/provenance.md`. How to verify a change: the `release-check` skill.

## check lists

[] load model ckpts
[] upload datasets
[x] upload eval tasks (HF `jepa-world-models/h-jepa`, `eval_trajs/`)
[] verify code for generating datasets
[] verify code for generating eval tasks
[] check whether unifying with the DROID action clipping of the original code can work?


## Open questions / decisions

1. Push-T 4-level planner: no compute sweep, `pusht_l4` keeps the paper setting; FLOPs/episode
   unknown (may exceed 100 TFLOPs).
2. Ant train set: the stitch half of `visual_antmaze_medium_explore_stitch_train.h5` comes from
   `visual_antmaze_medium_stitch_train_2_5x.h5`, which has no collection sidecar; the stitch seed in
   `scripts/data/config/visual_antmaze_medium_explore_stitch_train.yaml` (1072) is a placeholder.
3. FourRoom val/probing sets are exact prefixes of the train set (all collected with seed 3072).
   Planning unaffected; val loss / probing numbers inflated.
4. Cube train/val split: resolved, first 9,800 / last 200 episodes of LeWM's `cube_single_expert.h5`
   (`scripts/data/split_cube.py`).
5. Dataset names: AntMaze and FourRoom renamed (no `_2_5x`, no FourRoom dirs; see provenance.md). The
   local copies still have the paper names: test the release against a data root with the new names.
6. README `<LINK: ...>` placeholders to fill (datasets done): OGBench AntMaze expert policies (`ogbench_experts/ant`), and
   the four eval-task files (150-570 MB each, not in git; see `provenance.md`). Where to host (HF?)
7. Planning evals are not bit-reproducible run to run on Push-T, Ant and FourRoom (same code, same
   seed: e.g. FourRoom 2-episode SR 0% vs 100%; Push-T 1 of 4 runs differed). Cube is deterministic.
   Likely unseeded env resets (`world.reset(seed=None)`, FourRoom distractors) and/or nondeterministic
   GPU kernels. Fix or document before release.
8. Ant eval tasks: regeneration does not reproduce the original file (see Checks). Ship the original
   `expert_grid_d3_n50.pt` (recommended) or accept regenerated tasks?
9. Release pretrained checkpoints? If yes: add a README section, and keep the
   `lejepa_training_normalizer_v1` format tag and the `JEPA.__setstate__` shim that the
   `*_object.ckpt` / `normalizer.pt` files depend on (the legacy module aliases were not needed: all
   84 paper checkpoints load without them).
10. Ant probing now uses one concatenated eval set, so the paper's Ant probe numbers (depth-probes
   figure, probing tables; `main.tex` probing-data caption says "averaged over both") are not
   reproduced exactly. Re-run the Ant probing evals and regenerate the figures, or accept the gap?
11. DROID data hosting and license: the 256p re-encoded corpus (`droid_paths_minus16_256p.csv`,
    `droid_val_indist_256p.csv` and their mp4 episodes; README `<LINK: DROID 256p corpus>`). The
    CSVs listed absolute paths under the authors' DROID 256p directory,
    `config/train/base/droid.yaml` hardcodes the CSV paths, and the clip manifest points into the raw
    release (`droid_raw/1.0.1/`). Where to host, and may a re-encoded DROID
    be redistributed?
12. `fig:compute-pareto-real` TFLOPs axis: the original code measured it with `FlopCounterMode`; not ported.
    The release ships the skill-vs-(S, η) ladder only. Port the measurement or drop the axis from the
    reproduced claims?
13. DROID planner-seed reporting: the README reports mean ± SE over train seeds, each the mean over
    planner seeds 1/2/3 (`eval_droid.sh`). The planner-seed spread is ±4 points at S=4. Keep this, or
    report the SE over all 9 train x planner seeds?
14. `h_jepa/droid_assets/` (tracked, 10 KB: norm stats + clip manifest) vs the git-ignored
    `h_jepa/assets/` dir that holds the simulation eval tasks. Keep the separate tracked dir?
15. Prediction-loss precision: fp32 (`F.mse_loss` under bf16 autocast), decided 2026-10-02 (Basile):
    standard mixed-precision practice and the DROID paper runs' arithmetic (eb_jepa `SquareLossSeq`). The
    sim paper code computed it by hand in bf16 (`_ensemble_mse`; branch `bf16-pred-loss`, reverted), so sim
    loss terms differ from `lejepa_code` by bf16 rounding. Still open: the experiments_3 retrains of Ant
    H-JEPA 3/4, Cube H-JEPA 4, FourRoom HWM 4 came out below the paper.

## Pending work

- Datasets on HF `jepa-world-models/h-jepa` (2026-10-01): the 8 AntMaze/FourRoom `.h5` files (image columns
  re-stored with lossless Blosc-Zstd, 90 -> 32 GB; `observation`, an exact copy of `pixels`, dropped from the
  non-training AntMaze files; every column verified against the originals), `pusht_expert_val.h5` and
  `SHA256SUMS`; documented in the dataset card. The Zstd files are under `/mnt/vast/home/kevin/hjepa_release_data`,
  which also works as a `HJEPA_HOME` with the release names.
- Push-T / Cube training data: LeWM's HF releases. Checked 2026-10-01: LeWM's `cube_single_expert.h5` is
  byte-identical to ours and `scripts/data/split_cube.py` reproduces our train/val files; LeWM's Push-T train file
  plus `add_pusht_block_ori.py` equals our `pusht_expert_train.h5` in every column.
- Full planning pass on the paper checkpoints: all 28 planning configs x 3 seeds (`eval_depth.sh` +
  `eval_cost_ladder.sh`, ~250 SLURM jobs) against the README reference tables. Only `fourroom_l3`
  seed 42 checked so far (98% = paper). Needs the user's go-ahead.
- Optionally one training reproduction per env.
- DROID reproduction pass: `ENVS=droid train_all.sh` (9 runs) + `eval_droid.sh` (27 evals), then fill the
  README "This release" column. Its multi-GPU `srun` launch has not been run yet.
- DROID `*_v2` train configs (v2 flat and 2/3/4-level CLS recipes, 150 epochs) and the stride-2 eval configs
  `droid_l3` / `droid_l4` live in the gitignored `config/train/dev/` and `config/eval/dev/`, outside the release.
  The release DROID configs are `droid_lewm`, `droid_hwm_l2`, `droid_hjepa_l2` (train) and `droid_flat`, `droid_l2` (eval).
- DROID plan-eval: `eval.py --config-name droid_{flat,l2} policy=<ckpt> seed=<s> output.dir=<dir>` (dispatched to
  `droid_eval.run_clip_eval` on the `clips` key); planner values are hydra overrides
  (`solver.solvers.level<k>.optimizer_kwargs.lr=`, flat `solver.optimizer_kwargs.lr=`). `eval_droid.sh` picks the config
  from the model name and takes `EPOCHS` / `RUNS` (`eval_<plan>/epoch_<N>/`, missing ckpts skipped).
- 2026-09-29 DROID data on HF (answers open question 10): private dataset repo `jepa-world-models/h-jepa`,
  `droid/` = relative-path CSVs, 61 tar shards of the loader-read 256p files (`droid_256p/shard-*.tar`, 90.9 GB),
  `droid_raw_eval16.tar` (16 raw eval episodes), `SHA256SUMS`, `extract.sh`; dataset card = repo README.md (CC BY 4.0,
  DROID citation); packing/upload scripts are not part of the release. `DROIDClipReader` resolves relative CSV names, CSV
  entries and manifest `episode_path`s against `$HJEPA_HOME/droid`; the manifest is now relative (`droid_raw/1.0.1/...`).
  Data gate 42/42; e2e plan-eval bit-identical to a control run with the absolute paths. Pending: `base/droid.yaml` `name`/`val_name` -> the bare CSV
  names, and `launch.py check_datasets` must then look up `.csv` under `home / "droid"` (it resolves them against cwd).
  Done 2026-09-30 (next bullet).
- 2026-09-30 run-dir layout: every run under `ckpts/<env>/` (`env` key in `base/<env>.yaml`,
  `subdir: ${env}/${output_model_name}/seed${seed}`; `launch.py train` sweeps at `ckpts/<env>/<sweep>_<ts>/`), and
  stable-pretraining's cache (runs/, environment*.json, heartbeat, checkpoints) in `<run_dir>/spt/` via
  `spt.set(cache_dir=...)` in `main_hjepa.py`. `base/droid.yaml` reads the relative CSV names; `check_datasets` looks
  them up under `$HJEPA_HOME/droid` (and `/`-containing h5 names such as FourRoom's under `$HJEPA_HOME`).
  Old-layout runs (`ckpts/<env>_<model>/`) move with `mv` + a `subdir: <env>/...` rewrite in their `config.yaml`
  (Basile's DROID runs moved 2026-09-30; Kevin's to do, not part of the release).
- 2026-09-30 fix D1 (level-2 action SIGReg): with T level-2 states the loader builds T action chunks, the last one
  padded past the clip end (a transition that does not exist); predictor and IDM used the first T-1, the action
  SIGReg used all T. `hjepa_forward` now applies it to `act_emb[:, : T - 1]` as the original code did. Gates: every
  loss term and module gradient matches the original code within 1.2e-6 relative (eb H-JEPA seed-1 e-100 weights);
  flat forward and Cube (no action SIGReg) unchanged bitwise; `droid_hjepa_l2` / `droid_hwm_l2` change only
  `sigreg_loss_action_level2`. DROID HWM / H-JEPA l2 fleets retrained with it (`droid_{hwm,hjepa}_l2_n1fix`).
- Delete the test outputs under `$HJEPA_HOME/ckpts/`: `smoke_release`, `smoke_release2`,
  `smoke_port`, `regress` (created 2026-09-27; outside the repo, delete only when
  the user says so).

## Checks (done 2026-09-27)

- FourRoom 3-level repro (release `fourroom_l3` on `fourroom_distractors/9-11-2/4/seed42`): 98% = paper 98%.
- Eval-task regeneration with the release configs: Cube and FourRoom bit-identical (FourRoom also
  with the merged `scripts/generate_fourroom_eval_tasks.py`); Push-T identical apart from an extra
  unused `block_ori` column. Ant not identical: the first 31/50 requested start/goal tasks match, then
  the sequence diverges (different expert rollout failures: 52 vs 53 attempts), and settled start
  states differ everywhere (the original file predates the script and used a different expert
  normalizer).
- Config equivalence (`verify_train.py`, `verify_eval.py`), seeded training step (`fwd_test.py`,
  bit-identical to the pre-cleanup code) and Cube evals (bit-identical) after both feature-removal
  rounds; smoke runs of training + end-of-training planning + probing passed.

- DROID port (2026-09-28): models max|d| = 0 vs the reference port (source-run weights); data pipeline
  bitwise-identical; planner unit check passed; eval-only reproduction on the source-run weights within
  planner-seed noise (flat 31.9 / 34.5 / 32.4 vs original code 35.9 / 32.2 / 31.2).
- 2026-09-28 DROID plan-eval on the refactored hierarchical solver (9158935): the DROID eval script imports
  `build_hierarchical_solver` (renamed) and passes `steps_taken=0` (now required by `solve()`). `droid_*.yaml` unchanged.
  e2e gate on the converted reference ckpts: flat and hier `eval.csv` string-identical to the pre-refactor port
  (flat ATE 0.2490 / Fréchet mean 0.2972 / median 0.4134; hier 0.3205 / 0.2072 / 0.3870).
- 2026-09-28: `main_hjepa.py` passed `<name>_weights.ckpt` to `spt.Manager` unconditionally; stable-pretraining raises
  `FileNotFoundError` when it is absent, so fresh non-`quick_debug` runs could not start. Now passed only when the file exists.
  `decord` (DROID mp4 decoding) moved from the dev group into the `train` extra.
- 2026-09-28 DROID e2e H-JEPA vs original-code seed-1 loss trajectories (1.5k-step bins, 6k-29.5k steps): every component within
  ~5-8 %, LR identical at matched steps. Two residual differences: (i) the original cosine floors at `min_lr 1e-5`, spt's anneals to 0
  (only the last ~1 % of the schedule; matters for flat's e-100 = end of schedule); (ii) level-2 IDM raw loss drifts ~10-13 %
  BELOW the original after ~24k steps in both ports (pulls total loss ~5 % low). Neither explains the H-JEPA fidelity gap (34 vs 40 on
  one train seed); seeds 1000/10000 decide.
- 2026-09-29 DROID rate renamed to its true 5 fps: `data.fps: 5` (the loader strides by `ceil(tag / (4 fps))`,
  DROID mp4s being tagged 60 fps for 15 Hz footage), norm-stats key `full_fps5`, manifest
  `droid_clips_waypoint_curated16v2_5fps_gw36.json`, default tag `wp_5fps_gw36_cur16v2`. Data gate vs the reference
  (fps 20 / `full_fps20`) 42/42 equal, e2e plan-eval eval.csv byte-identical (flat, hier).

## Removed features

Round 1
- predictor ensembles; uncertainty cost; freeze / freeze_encoder / load_ckpt; random_waypoints

Round 2
- temp_straight (config only; also the generic "extra loss" loop that nothing fed)
- augment_static_window_prob
- detach_lower_level_inputs
- levelN.train / data.dataset.levelN.load
- encoders/predictors: impala, conv (MeNet6 channel fusion, SpatialConvPredictor), resnet9 alias;
  `squeeze` action pooler (hf_vit kept: the ViT encoder wraps it)
- in-training debug decoders (+ decoder optimizer group, W&B unroll figures); the probing decoder
  architecture now lives in `config/probing/<env>.yaml` `decoder:` (depth 4 / dim 256 / heads 8)
- dead config keys: projector_loss_weight, train_value_function, wm.type, encoder_resnet9
- planning regularizers; pixel/proprio planning-cost components; cost_probe; the `cost:` config
  (planning cost is always the embedding MSE, coeff was 1.0 everywhere); `*_probes.ckpt` saving/loading
- decoder attachment / decoder overrides in planning_eval and AutoCostModel
- plan visualizations: rollout videos, per-episode plots and dirs, plan timelines, cost plots,
  executed goal-cost plots, video subgoal panel
- horizon_decay_ratio (1.0 == decreasing_horizon_schedule); warm_start (false everywhere)
- two_room env: explore policy moved into `four_room/explore_policy.py`; the FourRoom/TwoRoom task
  scripts merged into `scripts/generate_fourroom_eval_tasks.py`
- non-HDF5 dataset classes (FolderDataset, ImageDataset, VideoDataset, MergeDataset, ConcatDataset,
  GoalDataset)

Round 3
- ground-truth goal-cost monotonicity/backtracking metrics (`gt_cost_*_levelN` in metrics.yaml,
  `stable_worldmodel/goal_cost_metrics.py`, the policy's reference-trajectory encoding)
- action_cost_space (only `pooled` kept; the `encoded` action-cost space and the config key)
- held-out task options (eval.exclude_eval_trajs_path / exclude_min_start_gap)
- unused imports in data.py, models/jepa.py, generate_maze_expert_grid_eval_tasks.py
- non-batch eval path (eval.batch_eval): all episodes always run as one batched World
- World.record_video

Round 4
- precompute_levels=true dataset path (level-1 crop + level-2 build in the dataset,
  `_build_level2_action_chunks`, the `level2` / `precompute_levels` args, the per-level-keys branch
  of `encode_hierarchical_per_level_inputs`, `encode_hierarchical(return_last_only)`) and the
  `precompute_levels` config key; the level-1-stream path is the only one
- planning metrics `xy_goal_l2`, `state_l2`, `state_normalized_l2`, `non_proprio_state_l2`,
  `non_proprio_state_normalized_l2`, `expert_action_l2`, `peak_gpu_mem_bytes`, the code computing
  them (final-state/xy/proprio and executed-action tracking, `World.last_actions`) and the
  `eval.expert_action_distance_{horizon,dims}` keys
- planning: `World.__init__` args `goal_transform` / `image_transform` / `extra_wrappers` /
  `goal_conditioned`, the `World` space properties, `eval.start_index`, the goal_state assert loop;
  on-the-fly eval-task sampling in `planning_eval.py` (`random`, `cube_pickup_centered[_lift]`; the
  Push-T dump always uses `stratified`), `_build_process`; `resolve_horizon=False` paths and the
  `resolve_horizon` key (horizons always follow the decreasing schedule), list-valued horizon /
  `cost_last_n`, the `horizon=` override, `level_span` / `_level_span`, `history_len`, per-level
  `receding_horizon`, `_execution_plan_config`; upper-level pixel+proprio fusion in the hierarchical
  solver, the `goal_{pixel,proprio}_embed_0` / extra `predicted_*` cost keys, solver outputs other
  than `actions` / `predictions` (cost histories, per-level results, `goal_embeddings`,
  `video_subgoal`), the `get_cost_components` split in the level adapter, the `jepas` fallback and
  list-form `solvers`; `swm.policy.ExpertPolicy`, `Actionable`, the duplicate `action_buffer`,
  `_solver_supports_budget_args`; eval keys `output.filename`, `world.image_shape`,
  `eval.pickup_height_delta`, `eval.traj_sampling_mode`, FourRoom `world.max_episode_steps: 400`
  (now `???` like the others); `Dataset` base class merged into `HDF5Dataset`, its `transform`
  constructor arg (the attribute stays: training and probing set it), `load_episode`, the
  `kernel_size` alias
- training/models: legacy checkpoint module aliases (all 84 paper checkpoints pickle only `models.*`
  paths) and the `ARPredictor` re-export in `models.module`; `SLURM_TMPDIR` / `local_cache_dir`, the
  no-val `random_split` / `train_split` path, CPU image resizing (stored images always match
  `img_size`), filling `wm.<col>_dim` from the dataset and the unread `wm.*_dim` keys (kept:
  level-1 `action_dim`, Ant's `proprio_dim` / `proprio_emb_dim`, `embed_dim`), in-training probe
  input-stream / architecture / `prober` optimizer options, the empty `global` param group and lr
  fallbacks; `print_tensor_shapes`, `kernel_size` / `num_preds` guards, fusion-builder and projector
  fallbacks, predictor / action-encoder type dispatch, `_add_pred_stream_outputs`; JEPA
  `proprio_encoder` / `projector` / `pred_proj` / `extra_encoders` and their branches,
  `rollout_action_embeddings`, `encode(extra_keys, temporal_window_size)`, the
  `goal_{pixel,proprio}_embed_0` writes, the `get_cost_components` split; `MLP.final_norm` /
  `extra_blocks` getattr shims, `ResidualLatentMLP` `residual` / `num_blocks` (+ `_ResidualMLPBlock`,
  config key `residual`); `SequenceEncoder` `stochastic` / `use_cls` / `step_mlp` / `uniform_input`
  (+ config key) / `action_masks` / truncation; `models/encoders/vit.py` (custom size configs;
  `create_hf_vit` is called directly, without `pretrained`), the `models.encoders` re-exports;
  `MixedHDF5Dataset.source_names`; `print_parameter_counts`; `PlanningEvalCallback` periodic /
  on-train-start evals, `config_path`, multi-seed aggregation and `_se` keys (the end-of-training
  `metrics.yaml` now has the same schema as `eval.py`'s; planner seed = `cfg.seed`; config keys
  `planning_eval.{every_n_epochs,run_on_train_start,seeds}`)
- probing: W&B decoder image logging (the `decodings/` PNGs stay) and the `visualization` options
  (always 32 uniformly spaced clips of the first eval set); `_add_distractor_aggregate`; probe
  `architectures`, per-level `decoder.config` / `decoder.enabled` (`levelN.train_decoder` stays),
  the Adam branch and `optimizer.type` / `optimizer.prober` / `optimizer.decoder` groups,
  `img_size` / `patch_size` overrides, `probe_targets`, `policy_config_path`, `max_epochs`,
  `load_probing_config(config_path)`; `_validate_dataset_cfg`, `_validate_eval_targets`, the
  eval-set key regex, the `_limit_batches` positivity check; `ConvProber` and the `linear` / `conv`
  prober types (`build_prober` is the 512-unit MLP); probing config keys `env`, `description`,
  top-level `num_workers`, `probe_targets`, Ant's `visualization` block. Kept: the `output_dir`
  override (the debug-run skill uses it) and `_dataset_col_variance`'s fallback for datasets
  without `get_col_stats` (only the Ant mixture has it; the audit wrongly listed it as dead).
- probing outputs: probe metrics are now macro NMSE (mean over target dims of MSE_d / Var_d, the
  number the depth-probes figure plots), computed in the one train/eval pass (`_run_epoch`) and
  written to `metrics.yaml` as `levelN_probe_<col>_nmse`. Gone:
  the second eval pass (`_evaluate_probe_dim_metrics`), the pooled NMSE, `probe_dim_metrics.csv`,
  `probe_summary_metrics.csv`, the floor columns and their W&B keys, `heads.ckpt`, `manifest.yaml`,
  `eval_metrics.yaml`, `eval_metrics_history.yaml`, `policy_train_config.yaml`, CPU image resizing
  (stored images always match `img_size`) and the upper-level fusion branch of the dense encode.
  Output dir: `config.yaml`, `metrics.yaml`, `decoder_levelN.ckpt`, `decodings/`. Checked on seeded
  Cube/Ant/FourRoom runs: new NMSE = old summary-CSV macro NMSE (1e-16), losses, decoders and
  decodings bit-identical. The dev figure pipeline (`depth_family.py`) reads the summary CSV and
  would need to read `metrics.yaml` instead.
- `HJEPA.encode_hierarchical_per_level_inputs` merged into `encode_hierarchical` (its only caller);
  gone: the `key` / `levels_to_encode` / `chunk_temporal_inputs` / `start_level` args, the level-1
  branch of the loop (the dense path now runs `_encode_level1` on every frame), the action/proprio
  restore and the input checks. Sparse outputs and fwd_test losses bit-identical; the dense path
  no longer leaks stale `pixel_embed_{0,2,3}` / `proprio_embed_{0,2,3}` keys on Ant.

Round 5
- Ant on-the-fly 50/50 explore/stitch mixing (`MixedHDF5Dataset`, `_select_mixed_episode_subsets`,
  config keys `sources` / `mix_mode` / `subset_unit` and the train `total_transitions`). Ant trains on `visual_antmaze_medium_explore_stitch_train.h5`, built
  (2026-09-29) from the seed-42 selection: all 12,500 explore episodes + 12,500 of the 31,250 stitch
  episodes, columns `pixels action xy qpos qvel proprio` (+ episode index columns). Seeds 43/44 now
  train on the seed-42 subset instead of their own. The collection configs
  `visual_antmaze_medium_{explore_train,stitch_train_2_5x}.yaml` became one
  `visual_antmaze_medium_explore_stitch_train.yaml` (collects an equivalent, not identical, file).
- multiple probing eval sets (`eval_datasets:` dict, `eval_mean/` metrics, `_mean_metrics`): every
  probing config has one `eval_dataset:`. Ant's two eval sets (155 explore + 155 stitch episodes,
  25k transitions each) became one file, `visual_antmaze_medium_probing_eval.h5` (their
  concatenation, 50k transitions), and one collection config. Ant probe NMSE is now normalized by
  the combined set's target variance instead of averaging the two per-set NMSEs, so it differs from
  the paper's Ant probing numbers (estimated -5% to +1% on `ant/9-6-1/1/seed42`); the paper evals
  were not re-run. Metric keys `eval/<key>/...` and `eval_mean/...` became `eval/...`, and decoding
  PNGs lost their eval-key prefix.
- `hjepa_utils.py` cleanup (2026-09-29): the rollout loss is straight-line code (no 7-tuple
  closures), and these are gone: the missing-key / `rollout_n` / dimension / component-name checks,
  the mixed legacy+component loss-format error (component sections now take precedence), the
  unused `action_encoder.type` dispatch, the optional `normalize_batch`, and the
  `create_world_model` inner helpers. fwd_test is bit-identical, as is a scratch variant covering
  rollout_n 2-3 (incl. history_size 1), pixel/proprio components, the legacy flat format and a
  disabled pred term.

Round 6 (2026-09-30, Kevin's PR review)
- nsteps-1 parallel unroll: every nsteps-1 level is now Kevin's native form (history_size T-1, rollout_n 1,
  `loss.embed.pred.weight` (T-1)/T, same T): `droid_hwm_l2` level 1 (7, 1, 7/8) and level 2 (2, 1, 2/3),
  `droid_hjepa_l2` level 2 (2, 1, 2/3), the dev v2 configs (`nsteps: null`
  overrides went with it. Old (HEAD code, nsteps 1) vs new (native x weight) on one real DROID batch, same weights:
  weighted pred loss |d| <= 2e-8 and `loss_levelN` |d| 0 at every level, grad cosine 1.000000000. Level-2 no-grad metrics
  (`mse_loss`, `l1_loss`, `dim_*`) and the unweighted `pred_loss_level2` now follow the native form (T-1 targets).
- `wm.context_length` / `wm.stop_gradient` (1 / true in every recipe) and their plumbing; `JEPA.parallel_unroll`
  hardcodes both. `nsteps: 2` moved from `base/droid.yaml` to the two configs that use it (`droid_lewm`,
  `droid_hjepa_l2` level 1); `wm.detach_pred_target` is read only in the nsteps branch and set only in `droid_lewm`
  (true). Forward gate vs the source dumps max|d| 0 (flat; l2s on every level-1 key and on `loss_level2` / `loss`),
  Cube regress unchanged. Old run configs with `nsteps: 1` still run the parallel unroll (1 pass, same loss).
- renames: `ActionMLPEncoder` -> `SequenceMLPEncoder` (config type stays `mlp`), `HJEPAModule` -> `GradClipModule`;
  `models/predictors/causal.py` merged into `predictors.py`. Existing `*_object.ckpt` pickling
  `models.predictors.causal.*` or `ActionMLPEncoder` no longer unpickle (state_dict keys unchanged: re-save them
  through a module/class-renaming unpickler, or rebuild from the config and load the state_dict).
- mentions of the original training code by name in code, configs and docs (`provenance.md` keeps the paper run ids).

Round 7 (2026-10-01)
- in-training online probes (`spt.callbacks.OnlineProbe` on `embed_{level}` / `pred_embed_{level}`, W&B-only
  diagnostics; inputs detached, own optimizer), with `add_probe_targets` / `probe_targets` / `probe_target_key`,
  the `pred_embed_{level}` outputs of `hjepa_forward`, `create_world_model`'s `embed_dims` return value and the
  unread DROID `levelN.probes.inputs` keys. Reported probe numbers come from `final_probing_decoding_eval` (fresh
  heads on the frozen model), unaffected. fwd_test bit-identical, verify_train output unchanged, Cube smoke run ok.
  Old `lightning_resume/last.ckpt` files still hold the probe modules (`callbacks_modules`); weights / object
  checkpoints are unaffected.

DROID (not ported from the original code)
- decoded-plans figure (needs the visual decoder); anticollapse 16-cell grid; crossval grids;
  `tab:sf-idm0`; varcomp; selective-bars; the TFLOPs measurement of `fig:compute-pareto-real`
  (`FlopCounterMode`)

## Kept on purpose

- `droid_data.py`: the one non-HDF5 loader. DROID episodes are the 256x256 mp4s, decoded with `decord`.
  `DROIDDataset(HDF5Dataset)` reuses `HDF5Dataset._setup_levels` (level configs and span, hoisted out of
  `HDF5Dataset.__init__`, Kevin's behaviour unchanged) and the inherited `__getitem__` / `load_chunk` /
  `_load_slice_with_levels`; it overrides only `_load_slice` (one random view and `span`-frame window per
  episode per access, via `DROIDClipReader`, as in the original loader: one index per episode, not per window) and the
  precomputed `get_col_stats` / `get_dim`. `lengths` (span per episode) only feeds the normalizer-artifact count.
  No resize: the files are already `img_size` (dataset `img_size` key removed from `base/droid.yaml`).
- latent action queue (queue_size): GradientSolver's upper-level action prior and clipping.
- action_embed.patch_embed: level 1 uses the Embedder default (True); levels 2+ set false.
- window_size > 1 in training and planning (unused by the paper, 1 everywhere, but must keep
  working): `JEPA._state_windows` / `_action_windows` / `_chunk_temporal_info` and the
  `temporal_window_size` attribute, `HJEPA._level_geometry` sparse level-1 encode, the dataset span
  math, `seq_encoder` (`SequenceEncoder` / `FlattenedSequenceEncoder`) as the windowed state encoder,
  the solver's window padding and `_latest_context_window`, and the dense
  `sparse_level1_encode=False` path + `unit_tests/test_sparse_level1_encode.py`, which checks the
  sparse encode for window 1-3.
- multi-step rollout loss (`rollout_n` > 1), the nsteps > 1 parallel unroll (`JEPA.parallel_unroll`: level 1 of
  `droid_lewm` / `droid_hjepa_l2`, `wm.nsteps: 2`; context 1 frame and stop-gradient between passes, both fixed;
  `wm.detach_pred_target` only there), pixel/proprio loss components and the legacy flat loss format,
  intermediate_cost_weight, quick_debug + DebugArtifactCleanupCallback (per request).
- `JEPA.__setstate__` `temporal_kernel_size` -> `temporal_window_size` shim: the four LeWM paper
  checkpoints (level 1, all seeds) still carry the old attribute (the audit wrongly listed it as unused).
- `JEPA.__setstate__` action-module shim: modules were renamed on 2026-09-28 (`action_encoder` ->
  `action_embed`, `action_pooler` -> `action_encoder`; `queue_size` moved to the new
  `action_encoder` config block), and all paper checkpoints pickle the old names.
  `verify_train.py` maps the paper configs' old keys the same way.
- `CLSDecoder` in `models.module`: 3 paper checkpoints pickle it (probing also uses it).
- callables `in_dataset: false` args (Cube's constant `cube_id: 0`; the audit wrongly listed it as
  unused).

## Unused features (candidates for removal)

From the round-4 audit (2026-09-28). [x] = agreed to remove; [ ] = candidate, not yet decided.
Decided to keep: probing/decoding as a whole; standalone `main_probing_decoding_eval.py` training
probes only (no decoders: the `levelN.train_decoder` overrides live in the training configs) is fine.

Filtered to items that remove more than ~100 lines (sizes are approximate).

### Envs
- [ ] `envs/ogbench/expert_policy.py` (manipulation oracle, 287) and `envs/pusht/expert_policy.py`
  (WeakPolicy, 86): only re-exported.
- [ ] `cube_env.py` (~500): double/triple/quadruple/octuple tasks (~400), `data_collection` mode,
  multiview render, ob_type `pixels` branch.
- [ ] `pusht/env.py` (~300): unused shapes (L, Z, square, I, small_tee, plus, box; ~230), `human`
  render mode, never-passed constructor args, dead attributes.
- [ ] `four_room/env.py` (~150+): `expert` / `expert_explore` / `static` distractor policies,
  `distractor_action_noise`, periodic `teleport_mode`, `target_min_steps`, `render_target`,
  `set_distractor_policy`, `init_value`.
- [ ] `wrapper.py` (~130): `EnsureImageShape` (57), `EnsureGoalInfoWrapper` (57), `VariationWrapper`
  non-`same` modes.

### Planning
- [ ] `World.evaluate` (151).

### Scripts
- [ ] `generate_dataset_eval_trajs.py` (~400): `random` and `_lift` modes, upright/moving filters
  (9 flags incl. legacy aliases), videos, `--max-attempts`, `--pickup-height-delta`,
  `--config-path`, positional overrides, `--num-episodes`. Check the output stays bit-identical.
- [ ] `generate_maze_expert_grid_eval_tasks.py` (~150): HumanoidMaze, `--goal-cell-set vertex`,
  `--dataset-name`, `--policy-dataset-type navigate`, `--policy-noise`, `--min/max-steps`,
  `--max-attempts`, videos, `--config-path`, `cfg.policy`-as-config branch.

### Needs a bit-exact check (changes seeding order or reset code)
- [ ] Cube / Push-T variation sub-spaces no config uses (colors, sizes, camera, light, shape spaces)
  and the matching `modify_mjcf_model` code (>100).
- [ ] `PointMazeEnv` visual-variation machinery (~300; fold the rest into `LocomazeEnv`).
