---
name: release-check
description: Verify that a change to the H-JEPA release keeps the paper results reproducible. Use after editing training code, planning code, the vendored stable_worldmodel library, or any release config; when removing or simplifying a feature; or when the user asks whether a config/eval still matches the paper runs.
---

# Release Check

The release must stay equivalent to the runs behind the paper. Four checks, from cheapest to most
expensive. Run from `$REPO/h_jepa` with

```bash
export PYTHONPATH=$REPO:$REPO/h_jepa
# HJEPA_HOME must point at the data root holding ckpts/ and eval assets
S=.agents/skills/release-check/scripts
```

Which paper run backs each config, and why: `.agents/provenance.md`.

## 1. Configs match the paper runs (seconds, CPU)

```bash
python $S/verify_train.py          # 28 training configs x 3 seeds vs each run's saved config.yaml
python $S/verify_eval.py           # 28 planning configs vs the eval_config.yaml they came from
```

Expected: every H-JEPA/HWM config and every planning config prints `IDENTICAL`. The four LeWM
configs have known behavior-neutral differences: `level1.lr` (equals `optimizer.lr`),
`trainer.precision` (`bf16` is Lightning's alias of `bf16-mixed`), dataloader workers/prefetch,
`save_every_n_epochs`, `level1.probes.enabled` (online probes are diagnostics), Ant's
`val_total_transitions` (validation only), FourRoom's `max_train_batches_total: null`. Keys removed on purpose during the cleanup are
listed in the scripts' `REMOVED` patterns; when you remove another config key, add it there.

## 2. Training step is unchanged (a minute, GPU)

```bash
python $S/fwd_test.py /tmp/after.pt
python $S/fwd_test.py --compare .agents/skills/release-check/reference/fwd_losses.pt /tmp/after.pt
```

Builds five models (Ant H-JEPA 3, FourRoom H-JEPA 4, Push-T HWM 2, Cube LeWM, Cube H-JEPA 2) with a
fixed seed and runs one `hjepa_forward` on a fixed random batch. Every loss term and parameter count
must be identical to the reference. The reference was regenerated on 2026-10-02 with the
prediction loss in fp32 (`F.mse_loss` under bf16 autocast); the paper's `lejepa_code` computed it in
bf16, so its prediction-loss terms differ from the reference by bf16 rounding. If a change
is meant to alter training, say so and regenerate the reference only after the user agrees.

Real training is not seeded before model construction (`spt.Manager` seeds afterwards), so two real
training runs never match; compare with this test, not with training logs.

## 3. Planning is unchanged (minutes, GPU)

Only Cube evals are deterministic run to run. Push-T, Ant and FourRoom evals are not (unseeded env
resets, FourRoom distractor motion): the same code gives different trajectories, even different
success, so never diff those. Reference values (paper checkpoints, 3 episodes):

| command | success |
|---|---|
| `eval.py --config-name cube_l3 eval.num_eval=3 policy=$HJEPA_HOME/ckpts/ogb/9-10-2/0/seed42/model_object.ckpt` | [T, F, F] |
| `eval.py --config-name cube_flat eval.num_eval=3 policy=$HJEPA_HOME/ckpts/ogb/7-28-1/ogb_level1_seed42__stage1/ogb_level1_seed42__stage1_object.ckpt` | [F, F, F] |
| `eval.py --config-name cube_l3_project eval.num_eval=3 policy=$HJEPA_HOME/ckpts/ogb/9-12-8/0/seed42/model_object.ckpt` | [T, F, F], steps 22 |

Add `output.dir=$(mktemp -d)`. These cover flat, hierarchical (3 levels with the top one skipped)
and projected-cost (4-level model) planning. Compare two metrics files with
`python $S/cmp_metrics.py a/metrics.yaml b/metrics.yaml`.

## 4. Paper numbers (hours, SLURM)

The reproduction targets are the reference tables in the root `README.md`. `scripts/eval_depth.sh`
and `scripts/eval_cost_ladder.sh` evaluate trained models; to check the planners against the paper
checkpoints instead, map each `<env>_<model>` to its paper checkpoint with `runs.checkpoint(env,
model, seed)`. Checked so far: `fourroom_l3` on `fourroom_distractors/9-11-2/4/seed42`, 50 episodes:
98% (paper 98%). Submit through the `slurm-submit` skill; results are noisy across runs (see 3).

## Eval tasks

Regenerate a task file with the command in the README and compare with
`python $S/cmp_tasks.py new.pt h_jepa/assets/eval_trajs/<...>.pt`. Cube and FourRoom regenerate
bit-identically, Push-T identically apart from an extra unused `block_ori` column. Ant does not
(different expert rollouts; whether to ship the original Ant file is open, see `.agents/release_todo.md`).

## Reporting

State which checks ran and their outcome, and list any difference that is not in the expected set
above. A difference in check 1 or 2 is a regression unless the change was meant to alter behavior.
