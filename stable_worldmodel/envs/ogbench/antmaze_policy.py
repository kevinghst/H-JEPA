"""Policies for collecting OGBench visual locomaze datasets."""

import glob
import os
import json
import sys
import types
from pathlib import Path

import numpy as np

from stable_worldmodel.policy import BasePolicy


def _find_wrapped_env_with_attr(env, attr_name):
    cur = env
    while cur is not None:
        if hasattr(cur, attr_name):
            return cur
        cur = getattr(cur, 'env', None)
    return None


def _ensure_ml_collections_shim():
    if 'ml_collections' in sys.modules:
        return

    class ConfigDict(dict):
        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = value

    module = types.ModuleType('ml_collections')
    module.ConfigDict = ConfigDict
    module.config_dict = types.SimpleNamespace(placeholder=lambda _: None)
    sys.modules['ml_collections'] = module


def _ensure_distrax_shim():
    if 'distrax' in sys.modules:
        return

    import jax
    import jax.numpy as jnp

    class MultivariateNormalDiag:
        def __init__(self, loc, scale_diag):
            self.loc = loc
            self.scale_diag = scale_diag

        def sample(self, seed=None):
            noise = jax.random.normal(seed, self.loc.shape)
            return self.loc + noise * self.scale_diag

        def mode(self):
            return self.loc

        def stddev(self):
            return self.scale_diag

    class Tanh:
        def forward(self, value):
            return jnp.tanh(value)

    class Block:
        def __init__(self, bijector, ndims=1):
            self.bijector = bijector
            self.ndims = ndims

        def forward(self, value):
            return self.bijector.forward(value)

    class Transformed:
        def __init__(self, distribution, bijector):
            self.distribution = distribution
            self.bijector = bijector
            self._distribution = distribution

        def sample(self, seed=None):
            return self.bijector.forward(self.distribution.sample(seed=seed))

        def mode(self):
            return self.bijector.forward(self.distribution.mode())

    class Categorical:
        def __init__(self, logits):
            self.logits = logits

    module = types.ModuleType('distrax')
    module.MultivariateNormalDiag = MultivariateNormalDiag
    module.Tanh = Tanh
    module.Block = Block
    module.Transformed = Transformed
    module.Categorical = Categorical
    sys.modules['distrax'] = module


class LocomazeExplorePolicy(BasePolicy):
    """Reproduce OGBench visual locomaze data-generation policies.

    ``explore`` samples random directions every ``sample_every`` steps.
    ``navigate`` and ``stitch`` use OGBench's maze oracle subgoal to choose a
    direction toward the current goal. All modes feed the agent's proprioceptive
    state plus the 2D direction into the restored SAC directional locomotion
    expert, add Gaussian action noise, and clip to ``[-1, 1]``.
    """

    action_aligned_info_keys = ('direction',)

    def __init__(
        self,
        dataset_type='explore',
        expert_name='ant',
        policy_prefix='antmaze',
        env_label='AntMaze',
        ogbench_impls_path=None,
        restore_path=None,
        restore_epoch=400000,
        sample_every=10,
        noise=None,
        seed=0,
        **kwargs,
    ):
        if restore_path is None:
            restore_path = os.path.join(
                os.environ['HJEPA_HOME'], f'ogbench_experts/{expert_name}'
            )
        super().__init__(**kwargs)
        if dataset_type not in {'explore', 'navigate', 'stitch'}:
            raise ValueError(
                f'Unsupported {env_label} dataset_type: {dataset_type}'
            )

        self.type = f'{policy_prefix}_{dataset_type}'
        self.dataset_type = dataset_type
        self.expert_name = expert_name
        self.env_label = env_label
        # Kept as an accepted argument for old configs. The expert code is now
        # vendored in stable_worldmodel.envs.ogbench.sac_expert.
        self.ogbench_impls_path = ogbench_impls_path
        self.restore_path = restore_path
        self.restore_epoch = restore_epoch
        self.sample_every = sample_every
        self.noise = 1.0 if noise is None and dataset_type == 'explore' else noise
        if self.noise is None:
            self.noise = 0.2
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.actor_fn = None

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_env(self, env):
        self.env = env
        self._wrapped_envs = list(self.env.unwrapped.envs)
        self._base_envs = [env.unwrapped for env in self._wrapped_envs]
        self._direction_envs = [
            _find_wrapped_env_with_attr(env, 'set_current_direction')
            for env in self._wrapped_envs
        ]
        self._antmaze_envs = [
            _find_wrapped_env_with_attr(env, 'set_navigation_goal')
            for env in self._wrapped_envs
        ]
        if any(env is None for env in self._direction_envs):
            raise RuntimeError(
                f'{self.__class__.__name__} requires envs with '
                'set_current_direction().'
            )
        if any(env is None for env in self._antmaze_envs):
            raise RuntimeError(
                f'{self.__class__.__name__} requires envs with '
                'set_navigation_goal().'
            )

        self._directions = np.full(
            (self.env.num_envs, 2), np.nan, dtype=np.float32
        )
        self._load_actor(self._base_envs[0])

    def _load_actor(self, example_env):
        _ensure_ml_collections_shim()
        _ensure_distrax_shim()
        from stable_worldmodel.envs.ogbench.sac_expert import (
            SACAgent,
            restore_agent,
            supply_rng,
        )

        candidates = glob.glob(str(Path(self.restore_path).expanduser()))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f'Expected exactly one OGBench {self.expert_name} expert '
                f'directory from restore_path={self.restore_path}, found '
                f'{candidates}. '
                'Download the OGBench expert policies (README §4.1).'
            )

        flags_path = Path(candidates[0]) / 'flags.json'
        if not flags_path.exists():
            raise FileNotFoundError(f'Missing OGBench expert config: {flags_path}')
        with flags_path.open() as f:
            agent_config = json.load(f)['agent']

        state_ob = np.asarray(example_env.get_ob(ob_type='states'))
        example_direction = np.zeros(2, dtype=state_ob.dtype)
        example_agent_ob = np.concatenate([state_ob[2:], example_direction])
        agent = SACAgent.create(
            self.seed,
            np.zeros_like(example_agent_ob),
            example_env.action_space.sample(),
            agent_config,
        )
        agent = restore_agent(agent, self.restore_path, self.restore_epoch)
        self.actor_fn = supply_rng(agent.sample_actions, rng=agent.rng)

    def _sample_direction(self):
        direction = self.rng.normal(size=2)
        norm = np.linalg.norm(direction)
        return (direction / (norm + 1e-6)).astype(np.float32)

    def _oracle_direction(self, env):
        subgoal_xy, _ = env.get_oracle_subgoal(env.get_xy(), env.cur_goal_xy)
        direction = subgoal_xy - env.get_xy()
        norm = np.linalg.norm(direction)
        return (direction / (norm + 1e-6)).astype(np.float32)

    def _update_navigate_goal(self, info_dict, env_idx):
        if self.dataset_type != 'navigate':
            return

        success = float(np.asarray(info_dict['success'][env_idx]).reshape(-1)[0])
        if success <= 0.0:
            return

        goal_ij = self._antmaze_envs[env_idx].sample_goal_ij()
        self._antmaze_envs[env_idx].set_navigation_goal(goal_ij=goal_ij)

    def _direction_for_env(self, info_dict, env_idx, env):
        if self.dataset_type == 'explore':
            step_idx = int(
                np.asarray(info_dict['step_idx'][env_idx]).reshape(-1)[0]
            )
            if step_idx == 0 or step_idx % self.sample_every == 0:
                self._directions[env_idx] = self._sample_direction()
            return self._directions[env_idx]

        self._update_navigate_goal(info_dict, env_idx)
        self._directions[env_idx] = self._oracle_direction(env)
        return self._directions[env_idx]

    def get_action(self, info_dict, **kwargs):
        if self.actor_fn is None:
            raise RuntimeError(f'{self.__class__.__name__} actor is not loaded.')

        actions = np.zeros(self.env.action_space.shape, dtype=np.float32)
        for i, env in enumerate(self._base_envs):
            direction = self._direction_for_env(info_dict, i, env)
            self._direction_envs[i].set_current_direction(direction)

            state_ob = np.asarray(env.get_ob(ob_type='states'))
            agent_ob = np.concatenate([state_ob[2:], direction])
            action = self.actor_fn(agent_ob, temperature=0)
            action = np.array(action, dtype=np.float32, copy=True)
            action += self.rng.normal(0, self.noise, action.shape).astype(
                np.float32
            )
            actions[i] = np.clip(action, -1, 1)

        return actions


class AntMazeExplorePolicy(LocomazeExplorePolicy):
    """Directional policy for visual AntMaze collection."""

    def __init__(self, **kwargs):
        kwargs.setdefault('expert_name', 'ant')
        kwargs.setdefault('policy_prefix', 'antmaze')
        kwargs.setdefault('env_label', 'AntMaze')
        kwargs.setdefault('restore_epoch', 400000)
        super().__init__(**kwargs)


class HumanoidMazeExplorePolicy(LocomazeExplorePolicy):
    """Directional policy for visual HumanoidMaze collection."""

    def __init__(self, **kwargs):
        kwargs.setdefault('expert_name', 'humanoid')
        kwargs.setdefault('policy_prefix', 'humanoidmaze')
        kwargs.setdefault('env_label', 'HumanoidMaze')
        kwargs.setdefault('restore_epoch', 40000000)
        super().__init__(**kwargs)
