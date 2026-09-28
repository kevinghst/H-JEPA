import math
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
    configured_horizon: int,
    replanning_interval: int,
    steps_taken: int | None = None,
    eval_budget: int,
) -> int:
    """Return this call's horizon from the decreasing schedule over the eval budget."""
    schedule = decreasing_horizon_schedule(
        num_calls=num_planning_calls(
            eval_budget=int(eval_budget),
            replanning_interval=int(replanning_interval),
        ),
        configured_horizon=int(configured_horizon),
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
        steps_taken: int | None = None,
        eval_budget: int | None = None,
    ) -> dict:
        """Solve the planning optimization problem to find optimal actions.

        Args:
            info_dict: Dictionary containing environment state information.
            init_action: Optional initial action sequence to warm-start the solver.
            steps_taken: Number of environment steps already consumed.
            eval_budget: Total environment-step budget for evaluation.

        Returns:
            Dictionary containing optimized actions and other solver-specific info.
        """
        ...
