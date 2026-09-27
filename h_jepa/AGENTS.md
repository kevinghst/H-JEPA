# AGENTS.md

## Goal

This repo (`/mnt/vast/home/kevin/H-JEPA`, GitHub `kevinghst/H-JEPA`) is the public code release
for the H-JEPA paper (paper repo: `/mnt/vast/home/kevin/hjepa_paper`). It has to reproduce:

- the bottom row of the depth figure (`planning/depth_planning_combined`): flat LeWM, H-JEPA and
  HWM with 2/3/4 levels on Visual AntMaze, FourRoom Distractors, OGBench Cube and Push-T, with
  planners under 100 TFLOPs/episode;
- the level-1 projected-cost columns of `tab:cost-ladder` / `tab:cost-ladder-appendix`.

Anything not needed for these results does not belong here. The research history (sweeps, analysis
skills, paper-figure pipeline, experiment records) lives in the development repo
`/mnt/vast/home/kevin/stable-wm-lejepa` (branch `hjepa_6_24`); this release was pruned from its
`code_release` branch. Open cleanup items are tracked in
`/mnt/vast/home/kevin/stable-wm-lejepa/lejepa_code/codex/tasks/code_release_todo.md`.

## Style

Readers are researchers reproducing the paper. Favor simple, readable code over generality:

- Minimal, targeted changes; readable repetition over abstraction; no helpers or layers unless they
  clearly help.
- No defensive checks, fallbacks, or compatibility shims for cases that cannot happen here.
- No comments unless something is genuinely non-obvious; no docstrings unless asked.
- Match the surrounding code; snake_case functions, PascalCase classes; flat control flow.
- Removing a feature: remove its code path and its config keys together.
- Do not change numerics silently. When touching training or planning code, verify against the
  previous behavior (a seeded training step, or a Cube planning eval, which is deterministic;
  Push-T/Ant/FourRoom evals are not bit-reproducible run to run).

## Layout

```
stable_worldmodel/   vendored library: envs (pusht, four_room, ogbench), HDF5 dataset, solvers, policy, world
scripts/data/        dataset collection (AntMaze, FourRoom) + Push-T utilities, configs in scripts/data/config/
h_jepa/              main code; run everything from here
  main_hjepa.py      training (ends with planning eval + final probing/decoding eval)
  eval.py            standalone planning eval
  main_probing_decoding_eval.py
  config/train/      <env>_{lewm,hjepa_l2..4,hwm_l2..4}.yaml on base/<env>.yaml
  config/eval/       <env>_{flat,l2,l3,l4,l2_project,l3_project,l4_project}.yaml (standalone YAMLs, no defaults list)
  config/probing/    <env>.yaml
  scripts/           eval-task generators, train_all.sh, eval_depth.sh, eval_cost_ladder.sh
  ARCHITECTURE.md    training, hierarchy and planning internals
```

`<env>` is `ant`, `fourroom`, `cube`, `pusht`. `README.md` at the repo root is the user-facing guide.

## Running

- Use the `stable_wm` conda env with `STABLEWM_HOME=/mnt/vast/home/kevin/stable-wm-lejepa/datasets`
  (datasets, experts, checkpoints under `ckpts/`).
- That env has the development repo's `stable_worldmodel` installed in editable mode, so always run
  with `PYTHONPATH=/mnt/vast/home/kevin/H-JEPA:/mnt/vast/home/kevin/H-JEPA/h_jepa` to import this
  repo's copy.
- Eval tasks live in `h_jepa/assets/eval_trajs/` (git-ignored; locally a symlink to the development
  repo's assets).
- Paper checkpoints (for checking evals) are under `$STABLEWM_HOME/ckpts/`; the depth-figure models
  are e.g. `ant/9-6-1/1/seed42` (H-JEPA 3 levels) and `ogb/9-10-2/0/seed42` (Cube 3 levels).

## Skills

`.agents/skills/` (symlinked as `.claude/skills/`): `debug-run` (fast smoke tests of train/eval/probing
commands), `slurm-submit` (one-off SLURM jobs), `paper-writing` (edits to the paper repo).

## Rules

- Commit only when asked, to the current branch; never push without being asked.
- Never delete files outside this repo (datasets, checkpoints, the development repo).
