import math
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import gymnasium as gym
import torch


class Costable(Protocol):
    """Protocol for world model cost functions."""

    def criterion(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Compute the cost criterion for action candidates.

        Args:
            info_dict: Dictionary containing environment state information.
            action_candidates: Tensor of proposed actions.

        Returns:
            A tensor of cost values for each action candidate.
        """
        ...

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Compute cost for given action candidates based on info dictionary.

        Args:
            info_dict: Dictionary containing environment state information.
            action_candidates: Tensor of proposed actions.

        Returns:
            A tensor of cost values for each action candidate.
        """
        ...


def resolve_cost_last_n(
    cost_last_n: int | Sequence[int],
    replanning_interval: int,
    steps_taken: int | None = None,
) -> int:
    """Return the cost_last_n for the current planning call.

    If cost_last_n is a list, indexes into it using the current call index
    (same schedule logic as horizon). Clamps to the last entry if exhausted.
    """
    if isinstance(cost_last_n, int):
        return max(1, cost_last_n)
    call_idx = 0 if steps_taken is None else int(steps_taken) // int(replanning_interval)
    call_idx = min(call_idx, len(cost_last_n) - 1)
    return max(1, int(cost_last_n[call_idx]))


def configured_planning_horizon(
    *,
    configured_horizon: int | Sequence[int],
    resolve_horizon: bool = False,
    replanning_interval: int,
    steps_taken: int | None = None,
    eval_budget: int | None = None,
) -> int:
    """Return the configured horizon for the current planning call."""
    if isinstance(configured_horizon, int):
        if configured_horizon <= 0:
            raise ValueError(
                f'Planning horizon must be positive, got {configured_horizon}.'
            )
        return configured_horizon

    if resolve_horizon:
        raise ValueError(
            'Expected integer horizon when resolve_horizon is enabled, '
            f'got {type(configured_horizon).__name__}.'
        )

    if not isinstance(configured_horizon, Sequence) or isinstance(
        configured_horizon, (str, bytes)
    ):
        raise ValueError(
            'Expected integer or list-valued horizon, '
            f'got {type(configured_horizon).__name__}.'
        )

    if replanning_interval <= 0:
        raise ValueError(
            f'Replanning interval must be positive, got {replanning_interval}.'
        )

    horizon_schedule = [int(horizon) for horizon in configured_horizon]
    if len(horizon_schedule) == 0:
        raise ValueError('Horizon schedule must contain at least one entry.')
    if any(horizon <= 0 for horizon in horizon_schedule):
        raise ValueError(
            f'All scheduled planning horizons must be positive, got {horizon_schedule}.'
        )

    if eval_budget is not None:
        expected_calls = math.ceil(int(eval_budget) / int(replanning_interval))
        if len(horizon_schedule) != expected_calls:
            raise ValueError(
                'Horizon schedule length must match the number of solver calls '
                'under eval_budget and replanning_interval: '
                f'got {len(horizon_schedule)} entries, expected {expected_calls}.'
            )

    call_idx = 0 if steps_taken is None else int(steps_taken) // int(replanning_interval)
    if call_idx >= len(horizon_schedule):
        raise ValueError(
            'Planning call index exceeds configured horizon schedule: '
            f'got call {call_idx} for schedule length {len(horizon_schedule)}.'
        )

    return horizon_schedule[call_idx]


def num_planning_calls(*, eval_budget: int, replanning_interval: int) -> int:
    """Return how many solver calls fit in the evaluation budget."""
    if eval_budget <= 0:
        raise ValueError(f'Eval budget must be positive, got {eval_budget}.')
    if replanning_interval <= 0:
        raise ValueError(
            f'Replanning interval must be positive, got {replanning_interval}.'
        )
    return math.ceil(int(eval_budget) / int(replanning_interval))


def planning_call_index(
    *,
    replanning_interval: int,
    steps_taken: int | None = None,
) -> int:
    """Return the current zero-based planning call index."""
    if replanning_interval <= 0:
        raise ValueError(
            f'Replanning interval must be positive, got {replanning_interval}.'
        )
    return 0 if steps_taken is None else int(steps_taken) // int(replanning_interval)


def decreasing_horizon_schedule(
    *,
    num_calls: int,
    configured_horizon: int,
) -> list[int]:
    """Build a monotone decreasing per-call horizon schedule.

    When there are enough calls to represent every horizon value, each value
    greater than 1 receives the same number of calls and horizon 1 receives any
    remainder. If there are fewer calls than horizon values, keep the configured
    horizon as the first call and horizon 1 as the final call.
    """
    num_calls = int(num_calls)
    configured_horizon = int(configured_horizon)
    if num_calls <= 0:
        raise ValueError(
            f'Number of planning calls must be positive, got {num_calls}.'
        )
    if configured_horizon <= 0:
        raise ValueError(
            f'Planning horizon must be positive, got {configured_horizon}.'
        )
    if configured_horizon == 1:
        return [1] * num_calls
    if num_calls == 1:
        return [configured_horizon]
    if num_calls < configured_horizon:
        span = configured_horizon - 1
        denom = num_calls - 1
        return [
            max(1, math.ceil(1 + span * (num_calls - 1 - idx) / denom))
            for idx in range(num_calls)
        ]

    horizon_values = list(range(configured_horizon, 0, -1))
    calls_per_horizon = num_calls // len(horizon_values)
    extra_calls = num_calls % len(horizon_values)
    counts = [calls_per_horizon] * len(horizon_values)
    for idx in range(len(horizon_values) - extra_calls, len(horizon_values)):
        counts[idx] += 1

    schedule = []
    for horizon, count in zip(horizon_values, counts, strict=True):
        schedule.extend([horizon] * count)
    return schedule


def get_planning_horizon(
    *,
    configured_horizon: int | Sequence[int],
    resolve_horizon: bool = False,
    replanning_interval: int,
    level_span: int,
    horizon: int | None = None,
    steps_taken: int | None = None,
    eval_budget: int | None = None,
) -> int:
    """Resolve the planning horizon for the current solver call."""
    if horizon is not None:
        planning_horizon = int(horizon)
        if planning_horizon <= 0:
            raise ValueError(
                f'Planning horizon must be positive, got {planning_horizon}.'
            )
        return planning_horizon

    base_horizon = configured_planning_horizon(
        configured_horizon=configured_horizon,
        resolve_horizon=resolve_horizon,
        replanning_interval=replanning_interval,
        steps_taken=steps_taken,
        eval_budget=eval_budget,
    )
    if not resolve_horizon:
        return base_horizon

    if eval_budget is None:
        return base_horizon

    schedule = decreasing_horizon_schedule(
        num_calls=num_planning_calls(
            eval_budget=int(eval_budget),
            replanning_interval=int(replanning_interval),
        ),
        configured_horizon=base_horizon,
    )
    call_idx = planning_call_index(
        replanning_interval=replanning_interval,
        steps_taken=steps_taken,
    )
    if call_idx >= len(schedule):
        raise ValueError(
            'Planning call index exceeds resolved horizon schedule: '
            f'got call {call_idx} for schedule length {len(schedule)}.'
        )
    return schedule[call_idx]


@runtime_checkable
class Solver(Protocol):
    """Protocol for model-based planning solvers."""

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        """Configure the solver with environment and planning specifications.

        Args:
            action_space: The action space of the environment.
            n_envs: Number of parallel environments.
            config: Planning configuration object.
        """
        ...

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        ...

    @property
    def n_envs(self) -> int:
        """Number of parallel environments being planned for."""
        ...

    @property
    def horizon(self) -> int:
        """Planning horizon length in timesteps."""
        ...

    def solve(
        self,
        info_dict: dict,
        init_action: torch.Tensor | None = None,
        horizon: int | None = None,
        steps_taken: int | None = None,
        eval_budget: int | None = None,
        level_span: int | None = None,
    ) -> dict:
        """Solve the planning optimization problem to find optimal actions.

        Args:
            info_dict: Dictionary containing environment state information.
            init_action: Optional initial action sequence to warm-start the solver.
            horizon: Optional per-call horizon override.
            steps_taken: Number of environment steps already consumed.
            eval_budget: Total environment-step budget for evaluation.
            level_span: Optional env-step span of one prediction step at this level.

        Returns:
            Dictionary containing optimized actions and other solver-specific info.
        """
        ...
