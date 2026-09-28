
"""Hierarchical solver scaffold for multi-level world models."""

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .solver import (
    num_planning_calls,
    planning_call_index,
    decreasing_horizon_schedule,
)


class _HierarchicalLevelModel(nn.Module):
    """Adapter that keeps rollout in local space and computes cost in parent space.

    For non-top levels, predicted latents are first encoded with the JEPA encoder
    from the level above before comparing against ``goal_embed_0``. When the goal
    lives more than one level up (every level in between was skipped at horizon 1
    with ``horizon_one_goal_cost_space='upper'``), the encoders are chained level
    by level up to ``goal_embed_0_level``.
    """

    def __init__(
        self,
        model: Any,
        upper_models: list[Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.upper_models = list(upper_models or [])
        self.upper_model = self.upper_models[0] if self.upper_models else None
        self.level = self.model.level

    def rollout(self, info_dict: dict, action_candidates: torch.Tensor) -> dict:
        return self.model.rollout(info_dict, action_candidates)

    def get_latent_action_queue(self) -> torch.Tensor | None:
        """Forward latent-action queue access to the wrapped level model."""
        return getattr(self.model, 'get_latent_action_queue', lambda: None)()

    @staticmethod
    def _prefix_repeat_first(x: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0:
            return x

        prefix_shape = (*x.shape[:-2], count, x.shape[-1])
        prefix = x[..., :1, :].expand(prefix_shape)
        return torch.cat([prefix, x], dim=-2)

    @staticmethod
    def _prefix_zeros(x: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0:
            return x

        prefix = x.new_zeros(*x.shape[:-2], count, x.shape[-1])
        return torch.cat([prefix, x], dim=-2)

    @staticmethod
    def _pad_zeros_to_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
        pad_len = int(target_len) - x.shape[-2]
        if pad_len < 0:
            raise ValueError(
                f"Cannot pad sequence of length {x.shape[-2]} to shorter length {target_len}."
            )
        if pad_len == 0:
            return x

        suffix = x.new_zeros(*x.shape[:-2], pad_len, x.shape[-1])
        return torch.cat([x, suffix], dim=-2)

    @staticmethod
    def _flatten_samples(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
        if x.ndim == 3:
            return x, None
        if x.ndim == 4:
            b, s, t, d = x.shape
            return x.reshape(b * s, t, d), (b, s)

        raise ValueError(
            f"Expected a rank-3 or rank-4 sequence, got shape {tuple(x.shape)}"
        )

    @staticmethod
    def _restore_samples(
        x: torch.Tensor,
        sample_shape: tuple[int, int] | None,
    ) -> torch.Tensor:
        if sample_shape is None:
            return x

        b, s = sample_shape
        return x.reshape(b, s, x.shape[-2], x.shape[-1])

    def _prepare_upper_state_stream(
        self, pred_emb: torch.Tensor, *, dense: bool = False
    ) -> torch.Tensor:
        # Non-dense (real-stride rollout): causal windows -> front-pad by kernel-1.
        # Dense (stride-1 goal projection): pad only enough to form one full window
        # when the stream is shorter than the kernel, then slide -> num windows =
        # max(1, N - kernel + 1). Both reduce to zero padding at window_size=1, so
        # a window_size=1 model is bit-identical to the pre-existing behavior.
        window_size = int(getattr(self.upper_model, "temporal_window_size", 1))
        count = (
            max(0, window_size - pred_emb.shape[-2]) if dense else window_size - 1
        )
        return self._prefix_repeat_first(pred_emb, count)

    def _prepare_upper_action_stream(
        self,
        action_candidates: torch.Tensor,
        *,
        target_len: int,
        dense: bool = False,
        state_len: int | None = None,
    ) -> torch.Tensor:
        # Mirror the state-stream front-pad (see _prepare_upper_state_stream) so
        # actions stay aligned to the upper states; actions pad with zeros.
        window_size = int(getattr(self.upper_model, "temporal_window_size", 1))
        count = (
            max(0, window_size - int(state_len)) if dense else window_size - 1
        )
        action_emb = self.model.action_embed(action_candidates)
        action_emb = self._prefix_zeros(action_emb, count)
        return self._pad_zeros_to_length(action_emb, target_len)

    def _project_actions_with_upper(
        self,
        prepared_pred: torch.Tensor,
        action_candidates: torch.Tensor,
        *,
        temporal_stride: int | None = None,
        dense: bool = False,
        state_len: int | None = None,
    ) -> torch.Tensor:
        input_key = f'embed_{self.level}'
        flat_pred, sample_shape = self._flatten_samples(prepared_pred)
        prepared_action = self._prepare_upper_action_stream(
            action_candidates,
            target_len=prepared_pred.shape[-2],
            dense=dense,
            state_len=state_len,
        )
        flat_action, action_sample_shape = self._flatten_samples(prepared_action)
        if action_sample_shape != sample_shape:
            raise ValueError(
                "State/action sample dimensions do not match for upper action projection."
            )

        chunked = self.upper_model._chunk_temporal_info(
            {input_key: flat_pred, "action": flat_action},
            key=input_key,
            stride=int(
                getattr(self.upper_model, "temporal_stride", 1)
                if temporal_stride is None
                else temporal_stride
            ),
            window_size=int(getattr(self.upper_model, "temporal_window_size", 1)),
        )

        action_mask = chunked.get("action_mask")
        if action_mask is None:
            pooled = self.upper_model.action_encoder(chunked["action"])
        else:
            pooled = self.upper_model.action_encoder(chunked["action"], action_mask)

        # The final pooled action points beyond the final encoded upper state.
        return self._restore_samples(pooled, sample_shape)[..., :-1, :]

    def _encode_with_upper(
        self,
        pred_emb: torch.Tensor,
        action_candidates: torch.Tensor | None = None,
        *,
        use_upper_space: bool = True,
        dense_upper_goal_projection: bool = False,
        goal_level: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode predicted embeddings through the upper-level JEPA encoder(s)."""
        if self.upper_model is None or not use_upper_space:
            return pred_emb, None
        if goal_level is None:
            goal_level = self.level + 1

        temporal_stride = None
        if dense_upper_goal_projection:
            # Encode the level-1 prediction stream into the upper space at stride 1
            # (one upper latent per sliding kernel-window) to score every predicted
            # step against the upper goal. Windowed prep supports window_size > 1.
            temporal_stride = 1

        prepared_pred = self._prepare_upper_state_stream(
            pred_emb, dense=bool(dense_upper_goal_projection)
        )
        flat_pred, sample_shape = self._flatten_samples(prepared_pred)

        input_key = f'embed_{self.level}'
        upper_out = self.upper_model.encode(
            {input_key: flat_pred},
            key=input_key,
            chunk_temporal_inputs=True,
            temporal_stride=temporal_stride,
        )
        for upper_level in range(self.level + 2, goal_level + 1):
            upper_out = self._encode_next_level(upper_out, upper_level, temporal_stride)

        pred_for_cost = self._restore_samples(upper_out["embed_0"], sample_shape)
        action_for_cost = None
        if action_candidates is not None:
            action_for_cost = self._project_actions_with_upper(
                prepared_pred,
                action_candidates,
                temporal_stride=temporal_stride,
                dense=bool(dense_upper_goal_projection),
                state_len=pred_emb.shape[-2],
            )

        return pred_for_cost, action_for_cost

    def _encode_next_level(
        self,
        lower_out: dict[str, torch.Tensor],
        upper_level: int,
        temporal_stride: int | None,
    ) -> dict[str, torch.Tensor]:
        """Encode one level's dense (stride-1) output stream through the level above it."""
        jepa = self.upper_models[upper_level - self.level - 1]
        input_key = f'embed_{upper_level - 1}'
        return jepa.encode(
            {input_key: lower_out['embed_0']},
            key=input_key,
            chunk_temporal_inputs=True,
            temporal_stride=temporal_stride,
        )

    def _action_cost(
        self,
        pred_action: torch.Tensor | None,
        info_dict: dict,
    ) -> torch.Tensor | None:
        target_action = info_dict.get("goal_action")
        if pred_action is None or target_action is None:
            return None

        cost_last_n = max(1, int(info_dict.get("cost_last_n", 1)))
        cost_window = min(cost_last_n, pred_action.shape[-2], target_action.shape[-2])
        pred_action = pred_action[..., -cost_window:, :]
        target_action = target_action[..., -cost_window:, :].expand_as(pred_action)

        cost = F.mse_loss(
            pred_action,
            target_action.detach(),
            reduction="none",
        )
        return cost.sum(dim=tuple(range(2, cost.ndim)))

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        for k, v in list(info_dict.items()):
            if torch.is_tensor(v):
                info_dict[k] = v.to(device)

        info_dict = self.rollout(info_dict, action_candidates)

        pred_emb = info_dict["predicted_embed_0"]
        use_upper_space = bool(info_dict.get('goal_embed_0_in_upper_space', True))
        goal_level = int(info_dict.get('goal_embed_0_level', self.level + 1))
        action_weight = float(info_dict.get("action_cost_weight", 0.0))
        pred_for_cost, action_for_cost = self._encode_with_upper(
            pred_emb,
            action_candidates if action_weight != 0.0 else None,
            use_upper_space=use_upper_space,
            dense_upper_goal_projection=bool(
                info_dict.get("dense_upper_goal_projection", False)
            ),
            goal_level=goal_level,
        )

        cost_info = dict(info_dict)
        cost_info["predicted_embed_0"] = pred_for_cost
        cost = self.model.criterion(cost_info)
        if action_weight != 0.0:
            action_cost = self._action_cost(action_for_cost, cost_info)
            if action_cost is not None:
                cost = cost + action_weight * action_cost
        return cost


class HierarchicalSolver:
    """Hierarchical wrapper that orchestrates one child solver per model level.

    Args:
        model: Hierarchical world model (e.g., HJEPA) exposing per-level modules.
        level_solvers: Mapping from 1-based level index to initialized solver.
    """

    def __init__(
        self,
        level_solvers: Mapping[int, Any],
        model: Any,
    ) -> None:
        self.model = model

        self._n_envs: int | None = None
        self._action_dim: int | None = None
        self._config: Any = None
        self._action_space: gym.Space | None = None

        self.level_models = self._extract_level_models(model)
        if len(self.level_models) == 0:
            raise ValueError('Hierarchical model must expose at least one level.')

        expected_levels = set(range(1, len(self.level_models) + 1))
        got_levels = set(level_solvers.keys())
        if got_levels != expected_levels:
            raise ValueError(
                'level_solvers keys must match hierarchical model levels: '
                f'expected {sorted(expected_levels)}, got {sorted(got_levels)}.'
            )

        self.level_solvers = dict(level_solvers)

    @staticmethod
    def _extract_level_models(model: Any) -> list[Any]:
        """Extract a list of per-level models from a hierarchical container."""
        return [model.get_level(level) for level in range(1, int(model.num_levels) + 1)]

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        """Configure all child solvers with shared environment settings."""
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))

        for level in sorted(self.level_solvers):
            self.level_solvers[level].configure(
                action_space=action_space,
                n_envs=n_envs,
                config=self._level_plan_config(level),
                level=level,
            )

    def _base_plan_config(self) -> Any:
        """Return config that defines the action-space execution granularity."""
        if self._config is None:
            raise RuntimeError('HierarchicalSolver.configure must be called first.')

        if hasattr(self._config, 'level1'):
            level1_cfg = getattr(self._config, 'level1')
            if level1_cfg is None:
                raise ValueError('Hierarchical config must define level1 plan config.')
            return level1_cfg

        return self._config

    def _level_plan_config(self, level: int) -> Any:
        """Return planning config for a specific hierarchy level."""
        if self._config is None:
            raise RuntimeError('HierarchicalSolver.configure must be called first.')

        level_key = f'level{level}'
        if hasattr(self._config, level_key):
            level_cfg = getattr(self._config, level_key)
            if level_cfg is None:
                raise ValueError(f'Hierarchical config must define {level_key}.')
            return level_cfg

        return self._config

    def _replanning_interval(self) -> int:
        """Return the environment-step interval between solver calls."""
        return int(self._config.receding_horizon) * int(
            self._level_plan_config(1).action_block
        )

    @staticmethod
    def _configured_int_horizon(configured_horizon: Any, *, level: int) -> int:
        try:
            horizon = int(configured_horizon)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'Expected integer horizon, '
                f'got {type(configured_horizon).__name__} for level {level}.'
            ) from exc
        if horizon <= 0:
            raise ValueError(
                f'Planning horizon must be positive for level {level}, got {horizon}.'
            )
        return horizon

    def _pre_decay_horizon(self, level: int, *, fallback: int) -> int:
        level_cfg = self._level_plan_config(level)
        configured = getattr(level_cfg, 'pre_decay_horizon', None)
        if configured is None:
            configured = getattr(self.level_models[level], 'temporal_stride', fallback)

        horizon = int(configured)
        if horizon <= 0:
            raise ValueError(
                f'pre_decay_horizon must be positive for level {level}, got {horizon}.'
            )
        return horizon

    def _resolved_horizon_schedules(
        self,
        *,
        eval_budget: int | None,
    ) -> dict[int, list[int]]:
        """Build recursive per-level horizon schedules for one evaluation."""
        if eval_budget is None:
            raise ValueError(
                'Hierarchical planning requires eval_budget to build the '
                'horizon schedules.'
            )

        calls = num_planning_calls(
            eval_budget=int(eval_budget),
            replanning_interval=self._replanning_interval(),
        )
        schedules: dict[int, list[int]] = {}
        highest_level = len(self.level_models)

        for level in range(highest_level, 0, -1):
            level_cfg = self._level_plan_config(level)
            horizon = self._configured_int_horizon(
                level_cfg.horizon,
                level=level,
            )
            if level == highest_level:
                schedules[level] = decreasing_horizon_schedule(
                    num_calls=calls,
                    configured_horizon=horizon,
                )
                continue

            upper_schedule = schedules[level + 1]
            pre_decay_horizon = self._pre_decay_horizon(level, fallback=horizon)
            try:
                tail_start = upper_schedule.index(1)
            except ValueError:
                schedules[level] = [pre_decay_horizon] * calls
                continue

            tail = decreasing_horizon_schedule(
                num_calls=calls - tail_start,
                configured_horizon=horizon,
            )
            schedules[level] = [pre_decay_horizon] * tail_start + tail

        return schedules

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        if self._n_envs is None:
            raise RuntimeError('HierarchicalSolver.configure must be called first.')
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        if self._action_dim is None or self._config is None:
            raise RuntimeError('HierarchicalSolver.configure must be called first.')
        base_cfg = self._base_plan_config()
        return self._action_dim * base_cfg.action_block

    @property
    def horizon(self) -> int:
        """Planning horizon in timesteps."""
        if self._config is None:
            raise RuntimeError('HierarchicalSolver.configure must be called first.')
        base_cfg = self._base_plan_config()
        return base_cfg.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """Make solver callable, forwarding to solve()."""
        return self.solve(*args, **kwargs)

    @staticmethod
    def _level_input_key(level: int) -> str:
        return 'pixels' if level == 1 else f'embed_{level - 1}'

    @staticmethod
    def _latest_context_window(x: torch.Tensor, window_size: int) -> torch.Tensor:
        window_size = int(window_size)
        if window_size <= 1:
            return x[:, -1:]

        if x.shape[1] >= window_size:
            return x[:, -window_size:]

        prefix = x[:, :1].expand(-1, window_size - x.shape[1], *x.shape[2:])
        return torch.cat([prefix, x], dim=1)

    def _encode_all_levels(
        self,
        info_dict: dict,
        key: str = 'pixels',
    ) -> dict[str, torch.Tensor]:
        """Encode one pixel-like stream through all hierarchy levels.

        Returns a dict with HJEPA-style keys: embed_1..embed_N.
        Actions are intentionally ignored.
        """
        encoded = self._level1_encode_info(info_dict, key)
        if encoded.get('pixels') is None:
            raise KeyError(f"Missing '{key}' key in info_dict for hierarchical planning.")

        goal_pixels = encoded['pixels']
        if not torch.is_tensor(goal_pixels):
            goal_pixels = torch.as_tensor(goal_pixels)

        model_device = next(self.level_models[0].parameters()).device
        goal_pixels = goal_pixels.to(model_device)

        # Match JEPA get_cost behavior if a sample axis is present.
        if goal_pixels.ndim >= 6:
            goal_pixels = goal_pixels[:, 0]

        encoded['pixels'] = goal_pixels
        for info_key, value in list(encoded.items()):
            if info_key == 'pixels':
                continue
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            value = value.to(model_device)
            if value.ndim >= 4:
                value = value[:, 0]
            encoded[info_key] = value

        outputs: dict[str, torch.Tensor] = {}

        # These observation/goal embeddings are fixed conditioning inputs for planning.
        # Detach them so repeated GD steps do not try to backprop through the same
        # encoder graph multiple times.
        with torch.no_grad():
            for level in range(1, len(self.level_models) + 1):
                jepa = self.level_models[level - 1]
                input_key = self._level_input_key(level)
                if input_key not in encoded:
                    raise KeyError(
                        f"Missing '{input_key}' for level {level} encoding."
                    )

                if level > 1:
                    encoded[input_key] = self._latest_context_window(
                        encoded[input_key],
                        getattr(jepa, "temporal_window_size", 1),
                    )

                level_out = jepa.encode(
                    encoded,
                    key=input_key,
                    chunk_temporal_inputs=level > 1,
                )

                encoded[f'embed_{level}'] = level_out['embed_0']
                outputs[f'embed_{level}'] = level_out['embed_0'][:, -1:].detach()

        return outputs

    def _level1_encode_info(self, info_dict: dict, key: str) -> dict[str, Any]:
        if key == 'goal':
            encoded = {'pixels': info_dict.get('goal')}
            if 'goal_proprio' in info_dict:
                encoded['proprio'] = info_dict['goal_proprio']
        else:
            encoded = {'pixels': info_dict.get(key)}
            if 'proprio' in info_dict:
                encoded['proprio'] = info_dict['proprio']

        return encoded

    def _level_action_targets(
        self,
        level: int,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return upper solver actions in the pooled macro-action space."""
        jepa = self.level_models[level - 1]
        device = next(jepa.parameters()).device
        return actions.to(device).detach()

    def solve(
        self,
        info_dict: dict,
        init_action: torch.Tensor | None = None,
        steps_taken: int | None = None,
        eval_budget: int | None = None,
    ) -> dict:
        """Plan top-down: each level's predicted latents become the goals of the level below."""
        subgoal_latents = None
        subgoal_actions = None
        horizon_one_upper_goal_level = None
        # Tracks whether the level above was anchored to the true goal at its last step.
        # True for the top level (trivially), and propagates down only when the substitution
        # is performed — so lower levels only anchor when the entire chain above is consistent.
        upper_anchored = True
        horizon_schedules = self._resolved_horizon_schedules(eval_budget=eval_budget)
        call_idx = planning_call_index(
            replanning_interval=self._replanning_interval(),
            steps_taken=steps_taken,
        )
        obs_embeddings = self._encode_all_levels(info_dict=info_dict, key='pixels')
        goal_embeddings = self._encode_all_levels(info_dict=info_dict, key='goal')

        for level in range(len(self.level_models), 0, -1):
            level_info = dict(info_dict)

            # Reuse precomputed observation embedding for this specific level.
            obs_key = f'embed_{level}'
            level_info['embed_0'] = obs_embeddings[obs_key]

            if subgoal_latents is not None:
                # set predicted latents as goal for current level
                next_goal = subgoal_latents
                next_action = subgoal_actions
                next_action_targets = None
                this_level_anchored = False
                if torch.is_tensor(next_goal) and next_goal.ndim >= 3:
                    upper_cfg = self._level_plan_config(level + 1)
                    num_subgoals = int(getattr(upper_cfg, 'num_subgoals', 1))
                    first_pred_idx = 1 if next_goal.shape[1] > 1 else 0
                    first_action_idx = max(0, first_pred_idx - 1)
                    includes_last = (first_pred_idx + num_subgoals >= next_goal.shape[1])
                    next_goal = next_goal[:, first_pred_idx:first_pred_idx + num_subgoals].clone()
                    if torch.is_tensor(next_action):
                        next_action = next_action[
                            :,
                            first_action_idx:first_action_idx + num_subgoals,
                        ].clone()
                        if next_action.shape[1] != next_goal.shape[1]:
                            raise ValueError(
                                f"Level {level}: got {next_action.shape[1]} action "
                                f"subgoals for {next_goal.shape[1]} state subgoals."
                            )
                        next_action_targets = self._level_action_targets(
                            level + 1,
                            next_action,
                        )
                    if includes_last and upper_anchored:
                        # The upper planner's last prediction should converge to its goal,
                        # so replace it with the ground truth goal embedding directly.
                        # Only valid if the entire chain above is anchored — otherwise the
                        # upper level was not optimizing toward goal_embeddings[level+1].
                        next_goal[:, -1] = goal_embeddings[f'embed_{level + 1}'].squeeze(1)
                        this_level_anchored = True
                level_info['goal_embed_0'] = next_goal
                if next_action_targets is not None:
                    level_info['goal_action'] = next_action_targets
                level_info['goal_embed_0_in_upper_space'] = True
                horizon_one_upper_goal_level = None
            elif horizon_one_upper_goal_level is not None:
                goal_level = horizon_one_upper_goal_level
                level_info['goal_embed_0'] = goal_embeddings[f'embed_{goal_level}']
                level_info['goal_embed_0_level'] = goal_level
                level_info['goal_embed_0_in_upper_space'] = True
                level_info['dense_upper_goal_projection'] = True
                this_level_anchored = True
            else:
                goal_key = f'embed_{level}'
                level_info['goal_embed_0'] = goal_embeddings[goal_key]
                level_info['goal_embed_0_in_upper_space'] = False
                this_level_anchored = True

            level_solver = self.level_solvers[level]
            level_cfg = self._level_plan_config(level)
            level_info['action_cost_weight'] = float(
                getattr(level_cfg, 'action_cost_weight', 0.0)
            )
            level_schedule = horizon_schedules[level]
            if call_idx >= len(level_schedule):
                raise ValueError(
                    'Planning call index exceeds resolved horizon schedule: '
                    f'got call {call_idx} for schedule length '
                    f'{len(level_schedule)} at level {level}.'
                )
            planning_horizon = level_schedule[call_idx]

            if level > 1 and planning_horizon == 1:
                # A one-step upper plan produces no useful subgoal for the lower
                # level, so skip that solve and pass through an actual goal.
                goal_cost_space = str(
                    getattr(level_cfg, 'horizon_one_goal_cost_space', 'lower')
                ).strip().lower()
                if goal_cost_space not in {'lower', 'upper'}:
                    raise ValueError(
                        "horizon_one_goal_cost_space must be 'lower' or 'upper', "
                        f"got {goal_cost_space!r} for level {level}."
                    )
                # 'upper' keeps the goal of the highest skipped level so a chain of
                # skipped levels projects the lower plan all the way up to it.
                horizon_one_upper_goal_level = (
                    (horizon_one_upper_goal_level or level)
                    if goal_cost_space == 'upper' else None
                )
                subgoal_latents = None
                subgoal_actions = None
                continue

            # num_subgoals, upper JEPA stride, and planning_horizon are coupled:
            # after stride-subsampling the level-N rollout (T+1 frames → ceil((T+1)/stride))
            # and dropping the context frame, exactly ceil((T+1)/stride)-1 frames remain to
            # compare against the subgoals. If num_subgoals doesn't match, criterion's
            # expand_as either crashes or — if cost_last_n accidentally equals num_subgoals —
            # silently compares the wrong temporal positions.
            goal_emb = level_info.get('goal_embed_0')
            if (
                level_info.get('goal_embed_0_in_upper_space', False)
                and torch.is_tensor(goal_emb)
                and goal_emb.shape[-2] > 1
            ):
                _upper_model = self.level_solvers[level].model.upper_model
                _stride = int(getattr(_upper_model, 'temporal_stride', 1))
                _expected = (planning_horizon + _stride) // _stride - 1
                assert goal_emb.shape[-2] == _expected, (
                    f'Level {level}: {goal_emb.shape[-2]} subgoals but upper temporal_stride={_stride} '
                    f'with planning_horizon={planning_horizon} expects {_expected}'
                )

            result = level_solver.solve(
                info_dict=level_info,
                init_action=init_action,
                planning_horizon=planning_horizon,
                steps_taken=steps_taken,
                eval_budget=eval_budget,
            )
            upper_anchored = this_level_anchored
            horizon_one_upper_goal_level = None
            subgoal_latents = result['predictions']['predicted_embed_0']
            subgoal_actions = result.get('actions')

        return {'actions': result['actions']}


def _build_hierarchical_solver(cfg: Any, model: Any) -> HierarchicalSolver:
    """Build one GradientSolver per level from cfg.solver.solvers.levelN."""
    import hydra

    level_models = HierarchicalSolver._extract_level_models(model)
    level_solvers = {}
    for level, level_model in enumerate(level_models, start=1):
        adapted_model = _HierarchicalLevelModel(
            model=level_model,
            upper_models=level_models[level:],
        )
        level_solvers[level] = hydra.utils.instantiate(
            cfg.solver.solvers[f'level{level}'], model=adapted_model
        )

    return HierarchicalSolver(level_solvers=level_solvers, model=model)
