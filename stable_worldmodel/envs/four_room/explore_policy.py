from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi
from typing import Any

import numpy as np

from stable_worldmodel.policy import BasePolicy, RandomPolicy


@dataclass
class ExploreControllerConfig:
    speed_mean: float = 0.75
    speed_noise_std: float = 0.05
    speed_autocorr: float = 0.95
    speed_min: float = 0.05
    speed_max: float = 1.0
    turn_autocorr: float = 0.95
    turn_noise_std: float = 0.08
    target_heading_gain: float = 0.15
    switch_tau: float | None = 40.0
    switch_mode: str = 'poisson'
    max_turn_rate: float = 0.5
    max_action_norm: float = 1.0
    initial_turn_rate_std: float = 0.05


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + pi) % (2.0 * pi) - pi


class ExploreActionController:
    """Environment-unaware smooth stochastic action generator."""

    def __init__(
        self,
        num_actors: int,
        seed: int | None = None,
        **kwargs: Any,
    ):
        self.cfg = ExploreControllerConfig(**kwargs)
        self.rng = np.random.default_rng(seed)
        self.num_actors = int(num_actors)
        self.heading = np.zeros(self.num_actors, dtype=np.float32)
        self.target_heading = np.zeros(self.num_actors, dtype=np.float32)
        self.turn_rate = np.zeros(self.num_actors, dtype=np.float32)
        self.speed = np.zeros(self.num_actors, dtype=np.float32)
        self.steps_since_switch = np.zeros(self.num_actors, dtype=np.int64)
        self.reset()

    def set_seed(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        for idx in range(self.num_actors):
            self.reset_actor(idx)

    def reset_actor(self, idx: int) -> None:
        self.reset_actor_direction(idx)
        self.speed[idx] = np.clip(
            self.rng.normal(self.cfg.speed_mean, self.cfg.speed_noise_std),
            self.cfg.speed_min,
            self.cfg.speed_max,
        )

    def reset_actor_direction(self, idx: int) -> None:
        self.heading[idx] = self.rng.uniform(0.0, 2.0 * pi)
        self.target_heading[idx] = self.rng.uniform(0.0, 2.0 * pi)
        self.steps_since_switch[idx] = 0
        self.turn_rate[idx] = self.rng.normal(
            0.0, self.cfg.initial_turn_rate_std
        )

    def _switch_probability(self) -> float:
        tau = self.cfg.switch_tau
        if tau is None or not np.isfinite(tau) or tau <= 0:
            return 0.0
        return 1.0 - exp(-1.0 / float(tau))

    def _periodic_switch_mask(self, actor_indices: np.ndarray) -> np.ndarray:
        tau = self.cfg.switch_tau
        if tau is None or not np.isfinite(tau) or tau <= 0:
            return np.zeros(actor_indices.size, dtype=bool)

        interval = max(1, int(round(float(tau))))
        self.steps_since_switch[actor_indices] += 1
        switch_mask = self.steps_since_switch[actor_indices] >= interval
        if np.any(switch_mask):
            self.steps_since_switch[actor_indices[switch_mask]] = 0
        return switch_mask

    def _switch_mask(self, actor_indices: np.ndarray) -> np.ndarray:
        mode = str(self.cfg.switch_mode)
        if mode == 'poisson':
            return (
                self.rng.uniform(0.0, 1.0, size=actor_indices.size)
                < self._switch_probability()
            )
        if mode == 'periodic':
            return self._periodic_switch_mask(actor_indices)
        raise ValueError(
            "switch_mode must be one of {'poisson', 'periodic'}, "
            f"got {mode!r}."
        )

    def get_actions(
        self,
        actor_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        if actor_indices is None:
            actor_indices = np.arange(self.num_actors)
        actor_indices = np.asarray(actor_indices, dtype=np.int64)

        if actor_indices.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        switch_mask = self._switch_mask(actor_indices)
        if np.any(switch_mask):
            switch_indices = actor_indices[switch_mask]
            self.target_heading[switch_indices] = self.rng.uniform(
                0.0,
                2.0 * pi,
                size=switch_indices.size,
            )

        heading_error = _wrap_angle(
            self.target_heading[actor_indices] - self.heading[actor_indices]
        )
        turn_noise = self.rng.normal(
            0.0,
            self.cfg.turn_noise_std,
            size=actor_indices.size,
        )
        turn_rate = (
            self.cfg.turn_autocorr * self.turn_rate[actor_indices]
            + self.cfg.target_heading_gain * heading_error
            + turn_noise
        )
        turn_rate = np.clip(
            turn_rate,
            -self.cfg.max_turn_rate,
            self.cfg.max_turn_rate,
        )
        self.turn_rate[actor_indices] = turn_rate
        self.heading[actor_indices] = _wrap_angle(
            self.heading[actor_indices] + turn_rate
        )

        speed_noise = self.rng.normal(
            0.0,
            self.cfg.speed_noise_std,
            size=actor_indices.size,
        )
        speed = (
            self.cfg.speed_mean
            + self.cfg.speed_autocorr
            * (self.speed[actor_indices] - self.cfg.speed_mean)
            + speed_noise
        )
        self.speed[actor_indices] = np.clip(
            speed,
            self.cfg.speed_min,
            self.cfg.speed_max,
        )

        actions = np.stack(
            [
                np.cos(self.heading[actor_indices]),
                np.sin(self.heading[actor_indices]),
            ],
            axis=-1,
        )
        actions = actions * self.speed[actor_indices, None]

        norm = np.linalg.norm(actions, axis=-1, keepdims=True)
        max_norm = max(float(self.cfg.max_action_norm), 1e-6)
        actions = actions * np.minimum(1.0, max_norm / np.maximum(norm, 1e-6))
        return np.clip(actions, -1.0, 1.0).astype(np.float32)


class ExplorePolicy(BasePolicy):
    """Collection policy using the shared smooth stochastic action process."""

    def __init__(self, seed: int | None = None, **kwargs: Any):
        super().__init__()
        self.type = 'explore'
        self.seed = seed
        self.controller_kwargs = dict(kwargs)
        self.controller: ExploreActionController | None = None

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        if self.controller is not None:
            self.controller.set_seed(seed)

    def _num_envs(self) -> int:
        if hasattr(self.env, 'num_envs'):
            return int(self.env.num_envs)
        return 1

    def reset(self) -> None:
        self.controller = ExploreActionController(
            self._num_envs(),
            seed=self.seed,
            **self.controller_kwargs,
        )

    def reset_env(self, env_idx: int) -> None:
        if self.controller is None:
            self.reset()
        assert self.controller is not None
        self.controller.reset_actor(int(env_idx))

    def get_action(self, info_dict, **kwargs):
        if self.controller is None:
            self.reset()
        assert self.controller is not None

        actions = self.controller.get_actions()
        if hasattr(self.env, 'action_space'):
            low = np.asarray(self.env.action_space.low, dtype=np.float32)
            high = np.asarray(self.env.action_space.high, dtype=np.float32)
            actions = np.clip(actions, low, high)
        return actions.astype(np.float32)


class EpisodicPolicyMixture(BasePolicy):
    """Choose one ego collection policy per episode for four-room data."""

    def __init__(
        self,
        policies: dict[str, BasePolicy],
        proportions: dict[str, float],
        total_episodes: int | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.type = 'mixture'
        self.policies = policies
        self.proportions = proportions
        self.total_episodes = total_episodes
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.active_names: list[str] = []
        self._schedule: list[str] = []
        self._cursor = 0

    @classmethod
    def from_config(
        cls,
        entries,
        total_episodes: int | None = None,
        seed: int | None = None,
    ) -> 'EpisodicPolicyMixture':
        from .expert_policy import ExpertPolicy

        policies: dict[str, BasePolicy] = {}
        proportions: dict[str, float] = {}
        for entry in entries:
            name = str(entry['name'])
            params = dict(entry.get('params', {}) or {})
            proportion = float(
                entry.get('proportion', entry.get('ratio', 1.0))
            )
            if name == 'expert':
                policy = ExpertPolicy(**params)
            elif name == 'random':
                policy = RandomPolicy(**params)
            elif name == 'explore':
                policy = ExplorePolicy(**params)
            else:
                raise ValueError(f'Unsupported four-room policy {name!r}.')
            policies[name] = policy
            proportions[name] = proportion

        return cls(
            policies=policies,
            proportions=proportions,
            total_episodes=total_episodes,
            seed=seed,
        )

    def set_env(self, env) -> None:
        self.env = env
        for policy in self.policies.values():
            policy.set_env(env)

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        for offset, policy in enumerate(self.policies.values()):
            if hasattr(policy, 'set_seed'):
                policy.set_seed(None if seed is None else seed + offset + 1)

    def _num_envs(self) -> int:
        if hasattr(self.env, 'num_envs'):
            return int(self.env.num_envs)
        return 1

    def _build_schedule(self) -> None:
        names = list(self.proportions)
        weights = np.asarray([self.proportions[name] for name in names])
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError(
                'Policy mixture proportions must sum to a positive value.'
            )
        weights = weights / weights.sum()

        if self.total_episodes is None:
            self._schedule = []
            return

        total = int(self.total_episodes)
        counts = np.floor(weights * total).astype(np.int64)
        remainder = total - int(counts.sum())
        if remainder > 0:
            fractional = weights * total - counts
            for idx in np.argsort(fractional)[-remainder:]:
                counts[idx] += 1

        schedule = [
            name
            for name, count in zip(names, counts, strict=True)
            for _ in range(int(count))
        ]
        self.rng.shuffle(schedule)
        self._schedule = schedule
        self._cursor = 0

    def _sample_name(self) -> str:
        if self._cursor < len(self._schedule):
            name = self._schedule[self._cursor]
            self._cursor += 1
            return name

        names = list(self.proportions)
        weights = np.asarray([self.proportions[name] for name in names])
        weights = weights / weights.sum()
        return str(self.rng.choice(names, p=weights))

    def reset(self) -> None:
        for policy in self.policies.values():
            policy.reset()
        self._build_schedule()
        self.active_names = [
            self._sample_name() for _ in range(self._num_envs())
        ]
        for env_idx, name in enumerate(self.active_names):
            policy = self.policies[name]
            if hasattr(policy, 'reset_env'):
                policy.reset_env(env_idx)

    def reset_env(self, env_idx: int) -> None:
        env_idx = int(env_idx)
        while len(self.active_names) <= env_idx:
            self.active_names.append(self._sample_name())
        name = self._sample_name()
        self.active_names[env_idx] = name
        policy = self.policies[name]
        if hasattr(policy, 'reset_env'):
            policy.reset_env(env_idx)

    def get_action(self, info_dict, **kwargs):
        if not self.active_names:
            self.reset()

        candidate_actions = {
            name: policy.get_action(info_dict, **kwargs)
            for name, policy in self.policies.items()
        }
        first = next(iter(candidate_actions.values()))
        actions = np.asarray(first, dtype=np.float32).copy()

        for env_idx, name in enumerate(self.active_names):
            actions[env_idx] = candidate_actions[name][env_idx]

        if hasattr(self.env, 'action_space'):
            low = np.asarray(self.env.action_space.low, dtype=np.float32)
            high = np.asarray(self.env.action_space.high, dtype=np.float32)
            actions = np.clip(actions, low, high)
        return actions.astype(np.float32)


__all__ = [
    'EpisodicPolicyMixture',
    'ExploreActionController',
    'ExploreControllerConfig',
    'ExplorePolicy',
]
