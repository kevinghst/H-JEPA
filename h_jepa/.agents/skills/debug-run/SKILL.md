---
name: debug-run
description: Run fast H-JEPA smoke tests that preserve the actual command's eval setting. Use when the user gives a `python main_hjepa.py` training command, `python eval.py` planning eval command, or `python main_probing_decoding_eval.py` probing command and wants it shrunk with quick-debug overrides, run locally, and diagnosed before a full run.
---

# Debug Run

Shrink a real command into a quick smoke test and run it locally. Keep it faithful to the real
run: same config, policy, eval tasks and eval callbacks, only smaller. Fidelity applies to model,
data and eval behavior, not to side effects: always redirect W&B and outputs to disposable
locations. Append the debug overrides after the user's command so Hydra's last-value-wins makes
them take precedence.

Run everything from `/mnt/vast/home/kevin/H-JEPA/h_jepa` with

```bash
export PYTHONPATH=/mnt/vast/home/kevin/H-JEPA:/mnt/vast/home/kevin/H-JEPA/h_jepa
```

(the conda env has the development repo's `stable_worldmodel` installed; this makes Python import
this repo's copy).

## main_hjepa.py

1. Create a unique run id, e.g. `debug_smoke_$(date +%Y%m%d_%H%M%S)`.
2. Append the training overrides:

```bash
quick_debug=true
wandb.enabled=false
subdir=<env>/<debug_run_id>
trainer.max_epochs=1
+trainer.limit_train_batches=2
+trainer.limit_val_batches=0
loader.batch_size=16
num_workers=2
loader.persistent_workers=false
```

3. Shrink the training data. Dataset caching (`keys_to_cache` loaded at build time) dominates
   smoke-test cost and is not reduced by `limit_train_batches`.
   - FourRoom, Cube, Push-T: train on the small val file instead:
     `data.dataset.name='${data.dataset.val_name}' ++data.dataset.total_transitions=null`
   - Ant: its val file is also large (6 GB), so subsample the train file instead:
     `++data.dataset.total_transitions=40000`

4. Planning eval. Release configs run it once at the end of training
   (`planning_eval.run_on_train_end=true`). Keep it unless the real command disables it, and shrink
   it through `planning_eval.overrides` (the callback loads `config/eval/<name>.yaml` itself, so
   plain `eval.*` overrides do not reach it). Use `++` so it works whether or not the key exists:

```bash
++planning_eval.overrides.eval.num_eval=1
```

   Optionally shrink the solver, matching its shape in the eval config: flat configs
   (`<env>_flat`) use `solver.n_steps` / `solver.num_samples`; hierarchical configs nest them under
   `solver.solvers.levelN` (one entry per level). The wrong shape injects an invalid kwarg
   (`GradientSolver.__init__() got an unexpected keyword argument 'solvers'`).

```bash
++planning_eval.overrides.solver.solvers.level1.n_steps=10
++planning_eval.overrides.solver.solvers.level1.num_samples=32
# ... one pair per level in the eval config
```

   Do not shrink `eval.eval_budget`: the world asserts `max_episode_steps >= goal_offset_steps`,
   and `max_episode_steps` is set to `2 * eval_budget`.

5. Final probing/decoding eval. Keep it unless the real command disables it, and shrink it:

```bash
+final_probing_decoding_eval.overrides.epochs=1
+final_probing_decoding_eval.overrides.limit_train_batches=2
+final_probing_decoding_eval.overrides.limit_eval_batches=1
+final_probing_decoding_eval.overrides.loader.batch_size=8
+final_probing_decoding_eval.overrides.loader.num_workers=0
+final_probing_decoding_eval.overrides.loader.persistent_workers=false
+final_probing_decoding_eval.overrides.save_artifacts=false
```

After the run, delete `$HJEPA_HOME/ckpts/<env>/<debug_run_id>` unless it is needed to diagnose a
failure.

## eval.py

Keep the command's `policy`, config, eval tasks, dataset and callables; only shrink:

```bash
output.dir=$(mktemp -d)/planning_eval
eval.num_eval=1
```

and optionally the solver (flat: `solver.n_steps=10 solver.num_samples=32`; hierarchical:
`solver.solvers.levelN.n_steps=10 solver.solvers.levelN.num_samples=32` for each level). W&B is
off in the release eval configs; keep `wandb.enabled=false`. Delete the temporary directory after
the run unless it holds evidence for a failure.

## main_probing_decoding_eval.py

```bash
++output_dir=$(mktemp -d)/probing
epochs=1
++limit_train_batches=2
++limit_eval_batches=1
loader.batch_size=8
loader.num_workers=0
loader.persistent_workers=false
++save_artifacts=false
```

(`output_dir`, `limit_*` and `save_artifacts` are not in the probing configs, hence `++`.)

Keep every configured probe and decoder level; the point is to exercise the same heads.

## Running and reporting

Poll until the process exits. Do not stop at successful setup: a training smoke test is only done
when the end-of-training planning eval and the final probing/decoding eval (if enabled) finish.

Report:

- whether it completed, and the exact overrides appended;
- which stages ran (training, planning eval, probing/decoding eval) and which were intentionally
  left disabled because the real run disables them;
- the disposable output location and whether it was deleted;
- the first real traceback if it failed (ignore MuJoCo/EGL cleanup noise after the main
  exception).

If the failure looks like a real code bug, report it with the smallest likely fix and wait for
approval before editing code.
