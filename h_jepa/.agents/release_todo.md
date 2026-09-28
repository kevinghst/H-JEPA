# Code release: open items

H-JEPA (this repo) is the source of truth for the release since 2026-09-27. The development repo's
`code_release` branch (`stable-wm-lejepa`, commit `bc777ef`) is frozen; do not port changes back.
Provenance of every config: `.agents/provenance.md`. How to verify a change: the `release-check` skill.

## Open questions / decisions

1. Push-T 4-level planner: no compute sweep, `pusht_l4` keeps the paper setting; FLOPs/episode
   unknown (may exceed 100 TFLOPs).
2. Ant `visual_antmaze_medium_stitch_train_2_5x.h5` has no collection sidecar; the seed in
   `scripts/data/config/visual_antmaze_medium_stitch_train_2_5x.yaml` (1072) is a placeholder.
3. FourRoom val/probing sets are exact prefixes of the train set (all collected with seed 3072).
   Planning unaffected; val loss / probing numbers inflated.
4. Cube train/val split (9,800/200 episodes of `cube_single_expert.h5`): splitting script not
   found. Does the LeWM download come pre-split?
5. Dataset names kept as-is (e.g. `fourroom_7_21/tp35/fourroom_tp35_d1`). Rename?
6. README `<LINK: ...>` placeholders to fill: Push-T data (`pusht_expert_{train,val}.h5`), Cube data
   (`cube_single_expert_{train,val}.h5`), OGBench AntMaze expert policies (`ogbench_experts/ant`), and
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

## Pending work

- Full planning pass on the paper checkpoints: all 28 planning configs x 3 seeds (`eval_depth.sh` +
  `eval_cost_ladder.sh`, ~250 SLURM jobs) against the README reference tables. Only `fourroom_l3`
  seed 42 checked so far (98% = paper). Needs the user's go-ahead.
- Optionally one training reproduction per env.
- Delete the test outputs under `$STABLEWM_HOME/ckpts/`: `smoke_release`, `smoke_release2`,
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

## Kept on purpose

- latent action queue (queue_size): GradientSolver's upper-level action prior and clipping.
- action_encoder.patch_embed: level 1 uses the Embedder default (True); levels 2+ set false.
- window_size > 1 in training and planning (unused by the paper, 1 everywhere, but must keep
  working): `JEPA._state_windows` / `_action_windows` / `_chunk_temporal_info` and the
  `temporal_window_size` attribute, `HJEPA._level_geometry` sparse level-1 encode, the dataset span
  math, `seq_encoder` (`SequenceEncoder` / `FlattenedSequenceEncoder`) as the windowed state encoder,
  the solver's window padding and `_latest_context_window`, and the dense
  `sparse_level1_encode=False` path + `unit_tests/test_sparse_level1_encode.py`, which checks the
  sparse encode for window 1-3.
- multi-step rollout loss, pixel/proprio loss components and the
  legacy flat loss format, intermediate_cost_weight, quick_debug + DebugArtifactCleanupCallback (per
  request).
- `JEPA.__setstate__` `temporal_kernel_size` -> `temporal_window_size` shim: the four LeWM paper
  checkpoints (level 1, all seeds) still carry the old attribute (the audit wrongly listed it as unused).
- `CLSDecoder` in `models.module`: 3 paper checkpoints pickle it (probing also uses it).
- callables `in_dataset: false` args (Cube's constant `cube_id: 0`; the audit wrongly listed it as
  unused).

## Unused features (candidates for removal)

From the round-4 audit (2026-09-28). [x] = agreed to remove; [ ] = candidate, not yet decided.
Decided to keep: probing/decoding as a whole; standalone `main_probing_decoding_eval.py` training
probes only (no decoders: the `levelN.train_decoder` overrides live in the training configs) is fine.

### Safe dead code: envs
- [ ] `envs/ogbench/expert_policy.py` (manipulation oracle) and `envs/pusht/expert_policy.py`
  (WeakPolicy): only re-exported.
- [ ] HumanoidMaze: `humanoidmaze_env.py`, `HumanoidMazeExplorePolicy`, its `ENV_SPECS` entry in
  `generate_maze_expert_grid_eval_tasks.py` (not registered, cannot be built).
- [ ] `cube_env.py`: double/triple/quadruple/octuple tasks (~400 lines), `data_collection` mode,
  multiview render, ob_type `pixels` branch.
- [ ] `pusht/env.py`: unused shapes (L, Z, square, I, small_tee, plus, box), `human` render mode,
  never-passed constructor args (`render_action`, `block_cog`, `damping`, `with_target`,
  `init_value`, `relative`), dead attributes.
- [ ] `four_room/env.py`: `expert` / `expert_explore` / `static` distractor policies,
  `distractor_action_noise`, periodic `teleport_mode`, `target_min_steps`, `render_target`,
  `set_distractor_policy`, `init_value`.
- [ ] `four_room/explore_policy.py`: `random` mixture entry and `ratio` alias.
- [ ] `locomaze_env.py` / `antmaze_policy.py`: `neutral_info_from_xy`, `ogbench_impls_path`,
  Humanoid-only `policy_prefix` / `env_label`; `navigate` dataset_type (medium).
- [ ] `envs/utils.py`: `get_mouse_pos`, `from_pygame`, `DrawOptions.draw_dot`, `pymunk_to_shapely`
  (drops `shapely`); `perturb_camera_angle` goes with the cube camera variation.
- [ ] `stable_worldmodel/utils.py`: `flatten_dict`, `record_video_from_dataset`, `pretraining`.
- [ ] `wrapper.py`: `EnsureImageShape`, `EnsureGoalInfoWrapper`, `VariationWrapper` non-`same` modes,
  `MegaWrapper(separate_goal)`.
- [ ] `spaces.py`: minor dead branches (`set_init_value`, Dict fallbacks); low value.

### Safe dead code: planning
- [ ] `World.evaluate`.
- [ ] `num_subgoals` > 1 in hierarchical planning (1 everywhere).
- [x] `gd.py`: `_rollout_final_actions` for the flat solver (medium).
- [x] `policy.py`: fold `AutoCostModel` / `_load_model_with_attribute` into the hierarchical load
  path (medium).

### Safe dead code: training and models
- [x] `main_hjepa.py`: `rand_str` run id (medium).
- [x] `models/module.py`: plain `Block` / `c is None` path and 4-D `Embedder` input (medium).
- [x] `data.py`: defensive checks (medium).
- [x] `utils.py`: per-epoch `*_epoch_N_object.ckpt` dumps (medium).
- [x] `hjepa_forward` no-grad logging extras (`mse`, `l1`, `dim_mean/std/min/max`) (medium).

### Safe dead code: probing (probing itself is kept)
- [ ] Per-dimension / summary CSVs and their logging, `manifest.yaml`, `eval_metrics_history.yaml`,
  `heads.ckpt` (duplicates `decoder_levelN.ckpt`), `policy_train_config.yaml`.

### Safe dead code: scripts
- [ ] `load_eval_config` `config_path` branch: only the generators' `--config-path` flags still pass
  it (moved here from the planning section).
- [ ] `generate_dataset_eval_trajs.py`: `random` and `_lift` modes, upright/moving filters (9 flags
  incl. legacy aliases), videos, `--max-attempts`, `--pickup-height-delta`, `--config-path`,
  positional overrides, `--num-episodes`. Check the output stays bit-identical.
- [ ] `generate_maze_expert_grid_eval_tasks.py`: `--goal-cell-set vertex`, `--dataset-name`,
  `--policy-dataset-type navigate`, `--policy-noise`, `--min/max-steps`, `--max-attempts`, videos,
  `--config-path`, `cfg.policy`-as-config branch; `_apply_configured_merges` (medium, check first).
- [ ] `generate_fourroom_eval_tasks.py`: videos, render sanity check, `--max-attempts`, `--max-steps`
  None branches, `--cross-n-rooms 3`, `_base_options`, multi-root config search; writing only `_d1`
  (medium); fold `fourroom_tp35_d0to5.yaml` into `d1.yaml` (medium).
- [ ] `scripts/data/convert_pusht_noise_to_h5.py` (unreferenced). Keep `fourroom_distractor_distance.py`.
- [ ] `collect_fourroom_distractors.py`: `num_traj` branch, `fixed_length_episodes: false` branch,
  `_is_set` assert, positivity checks.

### Needs a bit-exact check (changes seeding order or reset code)
- [ ] Cube / Push-T variation sub-spaces no config uses (colors, sizes, camera, light, shape spaces)
  and the matching `modify_mjcf_model` code.
- [ ] `PointMazeEnv` visual-variation machinery (fold the rest into `LocomazeEnv`).
- [ ] `options['state']` reset paths in Cube and Push-T.
- [ ] In-training `OnlineProbe` callbacks (probe init consumes the global torch RNG).

### Dependencies (`pyproject.toml`)
- [ ] Never imported: `tabulate`, `gdown`, `typer`, `minigrid`, `stable_baselines3`,
  `hydra-submitit-launcher` (+ `shapely` after the helper goes).
- [ ] Dev-only: `decord`, `mkdocs*`, `twine`, `pytest-cov`.
- [ ] Imported but undeclared: `scikit-learn`, `matplotlib`, possibly `imageio` (check transitive).

### Open (your call)
- [ ] Keep `planning_compute.json` + `_collect_solve_records` / `solve_records` (lets the
  "<100 TFLOPs" claim be checked) and `evaluation_time`?
- [ ] Other diagnostic planning outputs: `seeds`, `wall_clock_to_success`,
  `loaded_eval_original_lengths`, `steps_to_success[_success_only]`.
- [ ] Merge `FourRoomEnv` into `FourRoomDistractorsEnv` (cosmetic).
- [ ] Final probing/decoding rejects window_size != 1 (`_validate_dense_assumptions`), so a
  window_size > 1 training run raises after its planning eval unless
  `final_probing_decoding_eval.enabled=false`. Support it, or document it?

