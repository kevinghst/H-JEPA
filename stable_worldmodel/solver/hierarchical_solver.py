
"""Hierarchical planner: one gradient solver per level of a hierarchical world model, planned top-down."""

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .solver import (
    num_planning_calls,
    planning_call_index,
    decreasing_horizon_schedule,
)


class LevelCostModel(nn.Module):
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
        upper_models: list[Any],
    ) -> None:
        super().__init__()
        self.model = model
        self.upper_models = upper_models
        self.level = self.model.level

    def rollout(self, info_dict: dict, action_candidates: torch.Tensor) -> dict:
        return self.model.rollout(info_dict, action_candidates)

    def get_latent_action_queue(self) -> torch.Tensor | None:
        """Forward latent-action queue access to the wrapped level model."""
        return self.model.get_latent_action_queue()

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
        pad_len = target_len - x.shape[-2]
        if pad_len == 0:
            return x

        suffix = x.new_zeros(*x.shape[:-2], pad_len, x.shape[-1])
        return torch.cat([x, suffix], dim=-2)

    def _prepare_upper_state_stream(
        self, pred_emb: torch.Tensor, *, dense: bool = False
    ) -> torch.Tensor:
        # Non-dense (real-stride rollout): causal windows -> front-pad by kernel-1.
        # Dense (stride-1 goal projection): pad only enough to form one full window
        # when the stream is shorter than the kernel, then slide -> num windows =
        # max(1, N - kernel + 1). Both reduce to zero padding at window_size=1, so
        # a window_size=1 model is bit-identical to the pre-existing behavior.
        window_size = self.upper_models[0].temporal_window_size
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
        window_size = self.upper_models[0].temporal_window_size
        count = max(0, window_size - state_len) if dense else window_size - 1
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
        upper_model = self.upper_models[0]
        input_key = f'embed_{self.level}'
        b, s = prepared_pred.shape[:2]
        prepared_action = self._prepare_upper_action_stream(
            action_candidates,
            target_len=prepared_pred.shape[-2],
            dense=dense,
            state_len=state_len,
        )
        chunked = upper_model._chunk_temporal_info(
            {input_key: prepared_pred.flatten(0, 1), "action": prepared_action.flatten(0, 1)},
            key=input_key,
            stride=upper_model.temporal_stride if temporal_stride is None else temporal_stride,
            window_size=upper_model.temporal_window_size,
        )
        pooled = upper_model.action_encoder(chunked["action"], chunked["action_mask"])

        # The final pooled action points beyond the final encoded upper state.
        return pooled.unflatten(0, (b, s))[..., :-1, :]

    def _project_to_goal_space(
        self,
        pred_emb: torch.Tensor,
        action_candidates: torch.Tensor | None = None,
        *,
        use_upper_space: bool,
        dense_upper_goal_projection: bool,
        goal_level: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode predicted embeddings through the upper-level JEPA encoder(s)."""
        if not self.upper_models or not use_upper_space:
            return pred_emb, None

        temporal_stride = None
        if dense_upper_goal_projection:
            # Encode the level-1 prediction stream into the upper space at stride 1
            # (one upper latent per sliding kernel-window) to score every predicted
            # step against the upper goal. Windowed prep supports window_size > 1.
            temporal_stride = 1

        prepared_pred = self._prepare_upper_state_stream(
            pred_emb, dense=bool(dense_upper_goal_projection)
        )
        b, s = prepared_pred.shape[:2]
        input_key = f'embed_{self.level}'
        upper_out = self.upper_models[0].encode(
            {input_key: prepared_pred.flatten(0, 1)},
            key=input_key,
            chunk_temporal_inputs=True,
            temporal_stride=temporal_stride,
        )
        # Chain the dense (stride-1) output stream through each further level.
        for upper_level in range(self.level + 2, goal_level + 1):
            input_key = f'embed_{upper_level - 1}'
            upper_out = self.upper_models[upper_level - self.level - 1].encode(
                {input_key: upper_out['embed_0']},
                key=input_key,
                chunk_temporal_inputs=True,
                temporal_stride=temporal_stride,
            )

        pred_for_cost = upper_out["embed_0"].unflatten(0, (b, s))
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
        info_dict = self.rollout(info_dict, action_candidates)

        pred_emb = info_dict["predicted_embed_0"]
        use_upper_space = bool(info_dict.get('goal_embed_0_in_upper_space', True))
        goal_level = int(info_dict.get('goal_embed_0_level', self.level + 1))
        action_weight = float(info_dict.get("action_cost_weight", 0.0))
        pred_for_cost, action_for_cost = self._project_to_goal_space(
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
        level_solvers: Mapping from 1-based level index to initialized solver.
        level_models: Per-level JEPA models of the hierarchical model, level 1 first.
    """

    def __init__(
        self,
        level_solvers: Mapping[int, Any],
        level_models: list[Any],
    ) -> None:
        self.level_solvers = dict(level_solvers)
        self.level_models = level_models

        self._n_envs: int | None = None
        self._action_dim: int | None = None
        self._config: Any = None

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        """Configure all child solvers with shared environment settings."""
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

    def _level_plan_config(self, level: int) -> Any:
        """Return planning config for a specific hierarchy level."""
        return getattr(self._config, f'level{level}')

    def _replanning_interval(self) -> int:
        """Return the environment-step interval between solver calls."""
        return int(self._config.receding_horizon) * int(
            self._level_plan_config(1).action_block
        )

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

    def _horizon_schedules(
        self,
        *,
        eval_budget: int,
    ) -> dict[int, list[int]]:
        """Build recursive per-level horizon schedules for one evaluation."""
        calls = num_planning_calls(
            eval_budget=int(eval_budget),
            replanning_interval=self._replanning_interval(),
        )
        schedules: dict[int, list[int]] = {}
        highest_level = len(self.level_models)

        for level in range(highest_level, 0, -1):
            level_cfg = self._level_plan_config(level)
            horizon = level_cfg.horizon
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
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        return self._action_dim * self._level_plan_config(1).action_block

    @property
    def horizon(self) -> int:
        """Planning horizon in timesteps."""
        return self._level_plan_config(1).horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """Make solver callable, forwarding to solve()."""
        return self.solve(*args, **kwargs)

    @staticmethod
    def _level_input_key(level: int) -> str:
        return 'pixels' if level == 1 else f'embed_{level - 1}'

    @staticmethod
    def _latest_context_window(x: torch.Tensor, window_size: int) -> torch.Tensor:
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
                        jepa.temporal_window_size,
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

    def solve(
        self,
        info_dict: dict,
        steps_taken: int,
        eval_budget: int,
    ) -> dict:
        """Plan top-down: each level's predicted latents become the goals of the level below.

        A level with planning horizon 1 (other than level 1) is skipped. Each planned level takes
        its goal from one of three sources:
          - the level above planned: its first predicted latent (and macro-action), with the cost
            measured in the upper level's space;
          - the level(s) above were skipped with ``horizon_one_goal_cost_space='upper'``: the true
            goal at the highest skipped level, with predictions projected up to it at stride 1;
          - otherwise: the true goal at this level.
        """
        subgoal_latents = None
        subgoal_actions = None
        skipped_goal_level = None
        # Tracks whether the level above was anchored to the true goal at its last step.
        # True for the top level (trivially), and propagates down only when the substitution
        # is performed — so lower levels only anchor when the entire chain above is consistent.
        upper_anchored = True
        horizon_schedules = self._horizon_schedules(eval_budget=eval_budget)
        call_idx = planning_call_index(
            replanning_interval=self._replanning_interval(),
            steps_taken=steps_taken,
        )
        obs_embeddings = self._encode_all_levels(info_dict=info_dict, key='pixels')
        goal_embeddings = self._encode_all_levels(info_dict=info_dict, key='goal')

        for level in range(len(self.level_models), 0, -1):
            level_cfg = self._level_plan_config(level)
            planning_horizon = horizon_schedules[level][call_idx]

            if level > 1 and planning_horizon == 1:
                # A one-step upper plan produces no useful subgoal for the lower
                # level, so skip that solve and pass through an actual goal.
                # 'upper' keeps the goal of the highest skipped level so a chain of
                # skipped levels projects the lower plan all the way up to it.
                skipped_goal_level = (
                    (skipped_goal_level or level)
                    if level_cfg.horizon_one_goal_cost_space == 'upper' else None
                )
                subgoal_latents = None
                subgoal_actions = None
                continue

            level_info = dict(info_dict)
            level_info['embed_0'] = obs_embeddings[f'embed_{level}']
            level_info['action_cost_weight'] = level_cfg.action_cost_weight

            if subgoal_latents is not None:
                # Goal: the upper plan's first predicted latent (+ macro-action).
                num_subgoals = self._level_plan_config(level + 1).num_subgoals
                first_pred_idx = 1 if subgoal_latents.shape[1] > 1 else 0
                first_action_idx = max(0, first_pred_idx - 1)
                includes_last = (first_pred_idx + num_subgoals >= subgoal_latents.shape[1])
                next_goal = subgoal_latents[:, first_pred_idx:first_pred_idx + num_subgoals].clone()
                next_action = subgoal_actions[
                    :,
                    first_action_idx:first_action_idx + num_subgoals,
                ].clone()
                if next_action.shape[1] != next_goal.shape[1]:
                    raise ValueError(
                        f"Level {level}: got {next_action.shape[1]} action "
                        f"subgoals for {next_goal.shape[1]} state subgoals."
                    )
                this_level_anchored = False
                if includes_last and upper_anchored:
                    # The upper planner's last prediction should converge to its goal,
                    # so replace it with the ground truth goal embedding directly.
                    # Only valid if the entire chain above is anchored — otherwise the
                    # upper level was not optimizing toward goal_embeddings[level+1].
                    next_goal[:, -1] = goal_embeddings[f'embed_{level + 1}'].squeeze(1)
                    this_level_anchored = True
                level_info['goal_embed_0'] = next_goal
                level_info['goal_action'] = next_action
                level_info['goal_embed_0_in_upper_space'] = True
            elif skipped_goal_level is not None:
                # Goal: the true goal at the highest skipped level.
                level_info['goal_embed_0'] = goal_embeddings[f'embed_{skipped_goal_level}']
                level_info['goal_embed_0_level'] = skipped_goal_level
                level_info['goal_embed_0_in_upper_space'] = True
                level_info['dense_upper_goal_projection'] = True
                this_level_anchored = True
            else:
                # Goal: the true goal at this level.
                level_info['goal_embed_0'] = goal_embeddings[f'embed_{level}']
                level_info['goal_embed_0_in_upper_space'] = False
                this_level_anchored = True

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
                _stride = self.level_solvers[level].model.upper_models[0].temporal_stride
                _expected = (planning_horizon + _stride) // _stride - 1
                assert goal_emb.shape[-2] == _expected, (
                    f'Level {level}: {goal_emb.shape[-2]} subgoals but upper temporal_stride={_stride} '
                    f'with planning_horizon={planning_horizon} expects {_expected}'
                )

            result = self.level_solvers[level].solve(
                info_dict=level_info,
                planning_horizon=planning_horizon,
                steps_taken=steps_taken,
                eval_budget=eval_budget,
            )
            upper_anchored = this_level_anchored
            skipped_goal_level = None
            subgoal_latents = result['predictions']['predicted_embed_0']
            subgoal_actions = result['actions']

        return {'actions': result['actions']}


def build_hierarchical_solver(cfg: Any, model: Any) -> HierarchicalSolver:
    """Build one GradientSolver per level from cfg.solver.solvers.levelN."""
    level_models = [model.get_level(level) for level in range(1, model.num_levels + 1)]
    level_solvers = {}
    for level, level_model in enumerate(level_models, start=1):
        adapted_model = LevelCostModel(
            model=level_model,
            upper_models=level_models[level:],
        )
        level_solvers[level] = hydra.utils.instantiate(
            cfg.solver.solvers[f'level{level}'], model=adapted_model
        )

    return HierarchicalSolver(level_solvers=level_solvers, level_models=level_models)
