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
9. Release pretrained checkpoints? If yes: add a README section, and keep the legacy module aliases
   (`register_legacy_checkpoint_module_aliases`) and the `lejepa_training_normalizer_v1` format tag
   that the `*_object.ckpt` / `normalizer.pt` files depend on.

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

## Kept on purpose

- precompute_levels=true path: the four paper LeWM runs trained with it (key absent -> default true).
- latent action queue (queue_size): GradientSolver's upper-level action prior and clipping.
- action_encoder.patch_embed: level 1 uses the Embedder default (True); levels 2+ set false.
- multi-step rollout loss, window_size > 1 path + seq_encoder, pixel/proprio loss components and the
  legacy flat loss format, intermediate_cost_weight, quick_debug + DebugArtifactCleanupCallback (per
  request).
- ground-truth goal-cost monotonicity metrics (`gt_cost_monotonicity_levelN` in metrics.yaml).

## Unused features (candidates for removal)

- [ ] ground-truth goal-cost monotonicity metrics (diagnostic only)
- [ ] num_subgoals > 1 in hierarchical planning (1 everywhere)
- [ ] action_cost_space other than `pooled`
- [ ] non-batch eval path (eval.batch_eval true everywhere)
- [ ] held-out task options (eval.exclude_eval_trajs_path / exclude_min_start_gap)
- [ ] World.evaluate / World.record_video (only evaluate_from_dataset and record_dataset are used)
- [ ] pre-existing unused imports (data.py, models/jepa.py, generate_maze_expert_grid_eval_tasks.py)
