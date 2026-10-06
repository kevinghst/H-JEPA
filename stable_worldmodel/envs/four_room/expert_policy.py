from __future__ import annotations

import numpy as np

from stable_worldmodel.policy import BasePolicy


class ExpertPolicy(BasePolicy):
    """Oracle policy for the continuous four-room environment."""

    def __init__(
        self,
        action_noise: float = 0.0,
        action_repeat_prob: float = 0.0,
        door_reach_tol: float | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = 'expert'
        self.action_noise = float(action_noise)
        self.action_repeat_prob = float(action_repeat_prob)
        self.door_reach_tol = door_reach_tol
        self._last_action = None
        self.set_seed(seed)

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_env(self, env) -> None:
        self.env = env

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'state' in info_dict, "'state' must be provided in info_dict"
        assert 'goal_state' in info_dict, (
            "'goal_state' must be provided in info_dict"
        )

        base_env = self.env.unwrapped
        if hasattr(base_env, 'envs'):
            envs = [e.unwrapped for e in base_env.envs]
            is_vectorized = True
        else:
            envs = [base_env]
            is_vectorized = False

        actions = np.zeros(self.env.action_space.shape, dtype=np.float32)

        for i, env in enumerate(envs):
            if is_vectorized:
                agent_pos = np.asarray(
                    info_dict['state'][i], dtype=np.float32
                ).squeeze()
                goal_pos = np.asarray(
                    info_dict['goal_state'][i], dtype=np.float32
                ).squeeze()
            else:
                agent_pos = np.asarray(
                    info_dict['state'], dtype=np.float32
                ).squeeze()
                goal_pos = np.asarray(
                    info_dict['goal_state'], dtype=np.float32
                ).squeeze()

            tol = (
                float(self.door_reach_tol)
                if self.door_reach_tol is not None
                else 10.5
            )
            waypoint = env.expert_waypoint(agent_pos, goal_pos, tol)
            direction = waypoint - agent_pos
            norm = float(np.linalg.norm(direction))
            if norm > 1e-8:
                direction = direction / norm
            else:
                direction = np.zeros_like(direction, dtype=np.float32)

            if is_vectorized:
                actions[i] = direction.astype(np.float32)
            else:
                actions = direction.astype(np.float32)

        if self.action_noise > 0:
            actions = actions + self.rng.normal(
                0.0, self.action_noise, size=actions.shape
            ).astype(np.float32)

        if self._last_action is not None and self.action_repeat_prob > 0.0:
            repeat_shape = (actions.shape[0],) if is_vectorized else ()
            repeat_mask = (
                self.rng.uniform(0.0, 1.0, size=repeat_shape)
                < self.action_repeat_prob
            )
            if is_vectorized:
                actions[repeat_mask] = self._last_action[repeat_mask]
            elif repeat_mask:
                actions = self._last_action

        actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
        self._last_action = actions.copy()
        return actions
