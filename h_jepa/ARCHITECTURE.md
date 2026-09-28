# Hierarchical JEPA: Architecture Overview

This document covers the two entry points for working with the hierarchical world model:
- **Training**: `main_hjepa.py`
- **Planning eval**: `eval.py` → `planning_eval.py`

---

## 1. Dataloader

### Dataset

`swm.data.HDF5Dataset` loads pre-collected trajectories from HDF5 files. Each sample is a **clip** — a short window of consecutive frames sampled at a given frameskip.

The dataset is configured per-level. It returns a single level-1 stream long
enough for the top level, and the model constructs higher levels internally. For a
2-level model on PushT with level-2 stride 3:

| Level | Frameskip | Kernel size | Clip length (`num_steps`) |
|-------|-----------|-------------|---------------------------|
| 1     | 25 real steps per frame | 1 | 4 frames (history 3 + pred 1) |
| 2     | 75 real steps per frame (= 25 × stride 3) | 1 or more | 4 frames |

The dataset span is computed from the highest loaded level by recursively
expanding `num_steps` through each upper level's `frameskip`/`stride` and
`window_size`. This ensures state chunks at higher levels are fully populated;
action chunks may still be zero-padded by the model.

### Batch keys

The dataloader initially outputs only level-1 keys for each column in
`keys_to_load`:

```
pixels_level1   : (B, T1, C, H, W)   — raw RGB, ImageNet-normalized, resized to img_size
action_level1   : (B, T1, act_dim * frameskip1)   — actions stacked over frameskip
proprio_level1  : (B, T1, proprio_dim)  — z-score normalized (if loaded)
state_level1    : (B, T1, state_dim)    — z-score normalized (if loaded)
```

`T1` may be longer than the level-1 training target length because it must
contain enough lower-level states to construct all higher-level windows. After
model-side chunking, `encode_hierarchical_per_level_inputs` adds higher-level
state targets. If `levelN.window_size > 1`, non-action targets are chunked:

```
pixels_levelN   : (B, T, K, C, H, W)
proprio_levelN  : (B, T, K, proprio_dim)
state_levelN    : (B, T, K, state_dim)
```

where `K = levelN.window_size`. The learned state/action embeddings remain unchunked at the higher level:

```
embed_N  : (B, T, D)
action_N : (B, T, A)
```

**Image preprocessing**: `ToImage` + `Resize(img_size)` + ImageNet mean/std normalization, applied to all `pixels_*` columns.

**Action stacking (level 1)**: because the dataset skips 5 real steps between observations, the `frameskip` actions taken in between are concatenated, so the level-1 action vector has dimension `frameskip × raw_action_dim`.

**Action pooling (level 2+)**: an `action_pooler` (a small sequence encoder) compresses a chunk of lower-level action embeddings into a fixed-size vector. In model-side chunking, action chunks have length `levelN.stride`, independent of `levelN.window_size`.

**Non-pixel columns** are z-score normalized using per-column mean/std computed on the training set.

---

## 2. HJEPA: Model and Training Forward Pass

### Model structure

`HJEPA` is a thin `nn.ModuleList` container for N independent `JEPA` modules. Each `JEPA` has:

- **Encoder**: maps observations → latent vector. The encoder owns its projection head as `encoder.projector`.
  - Level 1 without proprio: ViT applied to raw pixels → CLS token → `encoder.projector` → `embed_dim`.
  - Level 1 with proprio: `encoder.pixel_encoder` maps pixels to `pixel_embed`, `encoder.proprio_encoder` maps proprio to `proprio_embed`, then `encoder.projector(concat(pixel_embed, proprio_embed))` produces `embed`.
  - Level 2+: a latent encoder applied to lower-level embeddings. With `window_size=1`, this can be a pointwise latent MLP. With `window_size>1`, it can be a sequence encoder that pools a lower-level temporal window into one abstract embedding. If proprio is enabled at that level, the pixel/proprio streams consume `pixel_embed_{level-1}` and `proprio_embed_{level-1}`.
- **Action Encoder**: small MLP that maps actions → action embedding.
- **Action Pooler** (level 2+): sequence encoder compressing a chunk of level-1 action embeddings into one vector.
- **Predictor**: `ProjectedPredictor`, where the base `ARPredictor` Transformer predicts in its hidden output space and `predictor.projector` maps the result to `embed_dim`.

### How the hierarchy is defined

The hierarchy is **temporal**: each level operates at a coarser timescale.

```
Level 1: frameskip = dataset.level1.frameskip  (set in dataset config)
Level N: frameskip = level(N-1).frameskip × levelN.stride
Level N state window length = levelN.window_size
```

Level 2's encoder takes as input the **embedding produced by level 1's encoder** — not raw pixels. The hierarchy is compositional:

```
s1_t = Enc1(o_t)
s2_T = Enc2(s1_{T·stride : T·stride + window_size})
```

To get a level-2 embedding, the model first runs the level-1 encoder, then feeds the result into the level-2 encoder.

For action alignment, the first lower-level action in upper chunk `T` starts at the last lower-level state in that state chunk:

```
state chunk  : s[start], ..., s[start + window_size - 1]
action chunk : a[start + window_size - 1], ..., length levelN.stride
```

State chunks are required to be full windows in the current training path. Action chunks may be zero-padded near the end.

### `encode_hierarchical`

The primitive encoding function. Given a single input dict (with one `pixels` key and one `action` key), it runs the full encoder chain from level 1 up through level N in sequence:

```python
# level 1: encode raw pixels
embed_1 = JEPA[1].encode(pixels)

# level 2: encode level-1 embeddings
embed_2 = JEPA[2].encode(embed_1)

# ... and so on up to levels_to_encode
```

Each level's `encode` call usually passes the previous level's `embed_0` output as the next level's observation. If a higher level has proprio enabled, it instead passes the previous level's `pixel_embed_0` and `proprio_embed_0` streams into that level's fusion encoder.

**Returns**: the input dict augmented with `embed_1..embed_N` (shape `(B, T, D)` each) and `action_1..action_N` (shape `(B, T, A)` each). Proprio-enabled levels also return `pixel_embed_N` and `proprio_embed_N`. With model-side chunking, it may also carry chunked state targets such as `pixels_levelN`, `state_levelN`, and `proprio_levelN`. If `return_last_only=True`, returns only the last level's model outputs, including stream embeddings when present.

This function is used during **planning**, where the observation and goal are each a single pixel stream that gets lifted through all levels.

### `encode_hierarchical_per_level_inputs`

During training, the dataset returns only a long `level1` stream.
`encode_hierarchical_per_level_inputs` calls `encode_hierarchical` once with
`return_last_only=False`; higher-level `JEPA.encode` calls do the constant-stride
chunking/subsampling. Lower levels may be longer than their training target
length while constructing upper levels, so before returning the output, each
level's state and action variables are randomly cropped to:

```
levelN.wm.history_size + levelN.wm.rollout_n
```

The same random time indices are used for all variables at that level. For example, with `level2.stride=3`, `level2.window_size=3`, and target length 4, level 1 is loaded long enough to produce four full level-2 windows, but returned `pixels_level1`, `action_level1`, `embed_1`, and `action_1` are cropped back to length 4.

**Sparse level-1 encoding** (`sparse_level1_encode=True`, the default): the crop
offsets are drawn *before* encoding, and the level-1 encoder runs only on the
frames that some kept step at some level actually reads (each level's crop, plus
the windows those steps expand to at the levels below). The results are scattered
into zero-filled full-length `embed_1` / `pixel_embed_1` / `proprio_embed_1`, so
the higher levels' chunking code is unchanged and the zeros only ever occupy
positions the crop discards. Actions are still encoded on the full stream. This
is numerically identical to encoding every frame (see
`unit_tests/test_sparse_level1_encode.py`) and pays off when `window_size <
stride` leaves gaps between windows: e.g. a 4-level stride-2 window-1 model
encodes ~12 of its 25 level-1 frames. When `window_size >= stride` every frame is
needed and the path degenerates to the dense one.

### `hjepa_forward` (the training step)

After encoding all levels, for each trainable level:

**1. Prediction**

For `rollout_n=1`, the training path keeps the causal teacher-forced one-step objective:

```
context  = emb[:, :history_size]          # (B, H, D)
act_ctx  = act_emb[:, :history_size]      # (B, H, A)
target   = emb[:, 1 : history_size + 1]   # (B, H, D)

pred = JEPA.predict(context, act_ctx)     # (B, H, D)
```

For `rollout_n>1`, the model rolls out autoregressively from the full `history_size` context for `rollout_n` steps, using ground-truth future actions and predicted future embeddings. When `history_size > 1`, it also keeps the teacher-forced one-step loss as an anchor: `teacher_forcing_loss + rollout_loss_weight * rollout_loss`. When `history_size == 1`, the teacher-forced term duplicates the first rollout step, so `pred_loss = rollout_loss_weight * rollout_loss`.

**2. Losses**

| Loss | Description |
|------|-------------|
| `pred_loss` | Teacher-forced one-step MSE for `rollout_n == 1`; for longer rollouts, weighted rollout MSE plus the teacher-forced anchor only when `history_size > 1` |
| `sigreg_loss` | SIGReg regularization: encourages embedding diversity and prevents collapse |
| `teacher_forcing_loss` | One-step teacher-forced MSE anchor |
| `rollout_loss` | Autoregressive full-history rollout MSE, logged when `rollout_n > 1` |

Total loss per level: `pred_loss + sigreg_weight * sigreg_loss`.

Total loss: sum across all trainable levels.

**3. Probes**

`OnlineProbe` callbacks attach small MLPs to `embed_{level}` and `pred_embed_{level}` to predict non-pixel quantities (e.g., proprio, state). These are trained in parallel during training as a representation quality metric and do not affect the world model gradients.

If a higher-level target is chunked, probes use the last element in the chunk as the target. This matches the temporal alignment used by the upper-level embedding and avoids asking a single abstract state to reconstruct the whole local window.

---

## 3. HierarchicalSolver (Planning)

### Overview

`HierarchicalSolver` orchestrates planning across all hierarchy levels. The intuition is:

> The top level plans coarsely (where should I be in 15-step chunks?), and each lower level plans how to get there at finer granularity.

The solver relies on the JEPA models themselves for hierarchy semantics. In
particular, planning reads each upper level's `temporal_stride` and
`temporal_window_size` from the model; eval configs should not duplicate these
as planning fields.

### Setup

```
eval.py
  └─ run_planning_eval
       └─ _build_policy
            └─ _build_hierarchical_solver
                 └─ HierarchicalSolver
                      ├─ level_solvers[1]: GradientSolver  (fine, level-1 model)
                      └─ level_solvers[2]: GradientSolver  (coarse, level-2 model)
```

Each level gets its own gradient-descent solver wrapped in a `_HierarchicalLevelModel` adapter. The adapter:
- Runs **rollout** in the level's own latent space (using `JEPA.rollout`).
- Computes **state cost** in the *upper* level's space: it encodes the predicted level-N latents through level-(N+1)'s `JEPA.encode(..., chunk_temporal_inputs=True)` before comparing against the subgoal targets.
- Optionally computes **action cost** in the upper level's action-embedding space, controlled by `hierarchical_plan_config.levelN.action_cost_weight` (default `0.0`).

### Planning loop (`HierarchicalSolver.solve`)

Planning proceeds **top-down**, one level at a time:

```
# Precompute embeddings of current obs and goal at all levels
obs_embeddings  = encode_all_levels(obs_pixels)   # embed_1, embed_2, ...
goal_embeddings = encode_all_levels(goal_pixels)

for level in [N, N-1, ..., 1]:
    obs_emb  = obs_embeddings[embed_{level}]
    goal_emb = (top level)    goal_embeddings[embed_{level}]
               (lower levels) subgoal latents from level above

    actions, subgoal_latents = level_solver.solve(obs_emb, goal_emb)

return level_1_actions
```

At the end, `outputs['actions']` contains the level-1 solver's chosen primitive actions.

### Observation and goal encoding

Before planning, `HierarchicalSolver` encodes the current observation and goal
through all hierarchy levels. For level 1 this is just the pixel encoder. For
level 2+, if the upper JEPA has `window_size > 1`, the solver builds a context
window before calling that upper encoder:

- If only one frame is available, repeat that frame until the window has length
  `window_size`.
- If a longer context stream is available, use the latest `window_size` lower
  embeddings so the encoded upper context ends at the most recent observation.
- Store only the last embedding at each level as `embed_{level}` for planning.

Example with one current observation `o_t`, `level2.stride=3`, and
`level2.window_size=3`:

```
level 1 input  : (o_t)
level 1 embed  : (e_t)
level 2 input  : (e_t, e_t, e_t)
level 2 embed  : E_t
```

This is a fallback for planning from a single observation. If the policy later
tracks real recent observations, those real embeddings can fill the higher-level
context window instead of repeating the current one.

### Key design decisions

**Subgoal passing**: the top-level solver predicts a trajectory of coarse latents `(B, T_coarse, D_top)`. A subset (`num_subgoals`) of these are extracted and passed as goals to the level below, effectively converting abstract "where I want to be" into concrete waypoints.

**Cross-level state cost**: level-N rollout produces latents in level-N space. To measure progress toward level-(N+1) subgoals, the adapter feeds the whole predicted lower-level stream into level-(N+1)'s `JEPA.encode` with `chunk_temporal_inputs=True`. The upper JEPA applies its own `temporal_stride` and `temporal_window_size`.

When `hierarchical_plan_config.resolve_horizon` is enabled, each level must use
an integer `horizon`. The solver precomputes one horizon per solver call. The
highest level decreases monotonically from its configured horizon to `1`; each
lower level stays at its configured horizon until the next upper level reaches
`1`, then decreases over that remaining suffix. Lower levels can set
`pre_decay_horizon` to override the horizon used before their decay segment
starts; when unset, this defaults to the next upper level's temporal stride.
When an upper level's
resolved planning horizon is `1`, that upper solve is skipped because its
predicted subgoal would not be used. By default, the next lower level then plans
directly to the actual goal embedding in its own latent space. Set
`horizon_one_goal_cost_space: upper` on the skipped upper level's plan config to
instead compare the lower rollout after projection through that upper encoder
against the actual upper-level goal embedding.

For `window_size=1`, this reduces to stride subsampling. With `stride=3`:

```
(s0), (s3), (s6), ...
```

For `window_size=3`, the adapter first prefix-repeats the current context state
so the first upper chunk contains no predictions:

```
lower rollout        : (s0, s1, s2, s3, s4, s5, s6)
prepared upper input : (s0, s0, s0, s1, s2, s3, s4, s5, s6)
upper chunks         : (s0,s0,s0), (s1,s2,s3), (s4,s5,s6)
```

`JEPA.criterion()` drops the first encoded timestep, so cost is applied only to
prediction chunks:

```
Enc(s1,s2,s3), Enc(s4,s5,s6), ...
```

**Cross-level action cost**: lower-level actions can also be projected into the
upper level's action embedding space and compared against the upper solver's
chosen macro-actions. This term is disabled by default:

```
hierarchical_plan_config:
  level1:
    action_cost_weight: 0.0
```

When enabled, level-N candidate actions are first encoded with level N's
`action_encoder`, then pooled by level N+1's `action_pooler`, and this pooled
macro-action is compared directly against the upper solver's optimized
macro-action.

For `stride=3`, `window_size=1`:

```
upper plan actions : A0, A1
lower actions      : a0, a1, a2, a3, a4, a5
pooled action cost : mse(A0, pool_upper(enc_lower(a0:a2)))
                   + mse(A1, pool_upper(enc_lower(a3:a5)))
```

For `window_size>1`, action embeddings are padded at the front to stay aligned
with the prefix-repeated state stream. The final projected upper action is
dropped because it points beyond the last encoded upper state.

**Goal anchoring**: if the top-level planner's last predicted state is known to converge toward the true goal, the last subgoal is replaced with the ground-truth goal embedding to prevent drift propagation down the chain.

**Temporal span**: the number of real environment steps spanned by one prediction step at each level:

```
span(1) = action_block
span(N) = span(N-1) × horizon(N-1)
```

For the offline PushT example: `span(1) = 5`, `span(2) = 5 × 15 = 75` real steps.

### Solvers

| Solver | Strategy |
|--------|----------|
| `GradientSolver` | Backpropagate through the differentiable rollout to optimize actions directly |

Each level can use a different solver and budget. The hierarchical eval configs
use separate level-specific settings for horizon, `num_subgoals`, `cost_last_n`,
`intermediate_cost_weight`, and solver-specific optimization parameters.
