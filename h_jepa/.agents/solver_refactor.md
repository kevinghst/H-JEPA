# Hierarchical solver refactor

Cleanups for `stable_worldmodel/solver/hierarchical_solver.py` (review of 2026-09-28), ranked by
leverage + confidence. Line numbers refer to the file after organization items 1, 3, 4 and 6.

Verification: Cube planning evals (deterministic, executed actions must match exactly) plus a smoke
run on another env.

Done: Tier 1 (dead/defensive code, 7 items); Cube l2/l3/l4/l3_project executed actions and solve
records bit-identical before/after, Ant l3, Push-T l3 and FourRoom l4_project smoke evals ran.
Organization item 1 (`solve()` loop reordered: skip check first, three goal modes in the
docstring, `horizon_one_upper_goal_level` -> `skipped_goal_level`) is done; Cube actions and solve
records bit-identical, Ant l3 and FourRoom l4_project smoke evals ran.
Organization items 3 and 4 are done: file ordered top-down (builder, `HierarchicalSolver` with
`solve` before its helpers, then the level adapter with `get_cost` first), `_flatten_samples` /
`_restore_samples` replaced by inline `flatten(0, 1)` / `unflatten(0, (b, s))`; Cube actions and
solve records bit-identical, Ant l3 and FourRoom l4_project smoke evals ran.
Organization item 6 is done: `_HierarchicalLevelModel` -> `LevelCostModel`, `_encode_with_upper`
-> `_project_to_goal_space`, `_resolved_horizon_schedules` -> `_horizon_schedules`,
`_build_hierarchical_solver` -> `build_hierarchical_solver` (also in `planning_eval.py` and
`ARCHITECTURE.md`), `import hydra` at the top (the data scripts already import it), new module
docstring; Cube actions and solve records bit-identical, Ant l3 and FourRoom l4_project smoke
evals ran.
Decided to keep: `num_subgoals`, the `pre_decay_horizon` fallback, the three goal flags passed
through `info_dict`.

## Organization

Plan: 5 after the shape check; 2 only with a window 1-3 equivalence test.

- [ ] 2. **Compute the window padding once in `_project_to_goal_space`.** `_prepare_upper_state_stream`
  and `_prepare_upper_action_stream` each recompute the same `count` from `dense` and
  `window_size`, which is why `dense`, `state_len` and `temporal_stride` are threaded through
  `_project_actions_with_upper`. Compute `count` and `stride` once at the top and pass them down;
  removes three parameters from two helpers. Low-medium impact, medium risk: it only changes the
  window > 1 path, which no paper config uses, so the Cube check can't catch a mistake. Do it only
  with a scratch test that runs the old and new `_prepare_upper_*` / `_project_actions_with_upper`
  on random tensors for windows 1-3 and requires identical outputs.
- [ ] 5. **Clean up the input handling in `_encode_all_levels`** (`:286-310`). Medium confidence:
  print the shapes the policy passes before removing anything.
  - `_level1_encode_info` only picks `goal`/`goal_proprio` versus `pixels`/`proprio`; inline it as
    a two-line mapping.
  - The variable `goal_pixels` is also used for the observation, which is misleading.
  - The `as_tensor` calls and the `ndim >= 6` / `ndim >= 4` "sample axis" slicing look defensive.

## Leave alone

- The dense vs. causal window padding (`_prepare_upper_*`): `release_todo.md` says window size > 1
  must keep working.
- The action cost path: `action_cost_weight: 0.2` is set in 6 configs.
- The `'lower'`/`'upper'` option for `horizon_one_goal_cost_space`: both are used.
- `this_level_anchored = True` in the `skipped_goal_level` branch of `solve()`: it is read
  when that level then solves (`upper_anchored = this_level_anchored`).
- `_action_cost`, `get_cost` and the horizon-schedule code: already clear.
