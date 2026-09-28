from collections import deque
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any, Protocol
from collections.abc import Callable

import numpy as np
import torch
from loguru import logger as logging
from torchvision import tv_tensors

import stable_worldmodel as swm
from stable_worldmodel.solver import Solver


def _register_legacy_checkpoint_module_aliases() -> None:
    """Map legacy LEJEPA module paths to the reorganized model package."""
    alias_pairs = {
        "jepa": "models.jepa",
        "hjepa": "models.hjepa",
        "module": "models.module",
        "seq_encoder": "models.encoders.seq_encoder",
        "spt_backbone_utils": "models.encoders.vit",
        "unit_tests.spt_backbone_utils": "models.encoders.vit",
        "factories.build_encoder": "models.encoders.build_encoder",
    }

    for legacy_name, current_name in alias_pairs.items():
        if legacy_name in sys.modules:
            continue
        try:
            sys.modules[legacy_name] = importlib.import_module(current_name)
        except ModuleNotFoundError:
            # Keep stable_worldmodel usable even when the LEJEPA package is absent.
            continue


@dataclass(frozen=True)
class PlanConfig:
    """Configuration for the MPC planning loop.

    Attributes:
        horizon: Planning horizon; each call uses a decreasing schedule derived
            from it and the eval budget.
        receding_horizon: Number of steps to execute before re-planning.
        action_block: Number of times each action is repeated (frameskip).
        action_dim: Optional explicit per-step action dimension used by solvers.
        pre_decay_horizon: Optional lower-level horizon used before its
            resolved decay segment starts in hierarchical planning.
        horizon_one_goal_cost_space: For skipped upper levels with horizon 1,
            choose lower-level goal cost space: 'lower' keeps current behavior,
            'upper' projects lower predictions to the actual upper-level goal.
    """

    horizon: int
    receding_horizon: int
    action_block: int = 1
    action_dim: int | None = None
    pre_decay_horizon: int | None = None
    num_subgoals: int = 1
    cost_last_n: int = 1
    intermediate_cost_weight: float = 1.0
    action_cost_weight: float = 0.0
    horizon_one_goal_cost_space: str = 'lower'


@dataclass(frozen=True)
class HierarchicalPlanConfig:
    """Configuration container for hierarchical MPC planning.

    Attributes:
        receding_horizon: Number of high-level steps to execute before re-planning.
        levels: Per-level planning configs, ordered from level 1 upward. Also
            exposed as ``level1``, ``level2``, ... attributes.
    """

    receding_horizon: int
    levels: tuple[PlanConfig, ...] = ()

    def __getattr__(self, name: str) -> PlanConfig:
        index = name.removeprefix('level')
        if name.startswith('level') and index.isdigit() and 1 <= int(index) <= len(self.levels):
            return self.levels[int(index) - 1]
        raise AttributeError(name)


class Transformable(Protocol):
    """Protocol for reversible data transformations (e.g., normalizers, scalers)."""

    def transform(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Apply preprocessing to input data.

        Args:
            x: Input data as a numpy array.

        Returns:
            Preprocessed data as a numpy array.
        """
        ...

    def inverse_transform(
        self, x: np.ndarray
    ) -> np.ndarray:  # pragma: no cover
        """Reverse the preprocessing transformation.

        Args:
            x: Preprocessed data as a numpy array.

        Returns:
            Original data as a numpy array.
        """
        ...


class BasePolicy:
    """Base class for agent policies.

    Attributes:
        env: The environment the policy is associated with.
        type: A string identifier for the policy type.
    """

    env: Any
    type: str

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the base policy.

        Args:
            **kwargs: Additional configuration parameters.
        """
        self.env = None
        self.type = 'base'
        for arg, value in kwargs.items():
            setattr(self, arg, value)

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        """Get action from the policy given the observation.

        Args:
            obs: The current observation from the environment.
            **kwargs: Additional parameters for action selection.

        Returns:
            Selected action as a numpy array.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    def set_env(self, env: Any) -> None:
        """Associate this policy with an environment.

        Args:
            env: The environment to associate.
        """
        self.env = env

    def reset(self) -> None:
        """Reset any internal policy state before a new rollout."""
        return None

    def _prepare_info(self, info_dict: dict) -> dict[str, torch.Tensor]:
        """Pre-process and transform observations.

        Applies preprocessing (via `self.process`) and transformations (via `self.transform`)
        to observation data. Used by subclasses like WorldModelPolicy.

        Args:
            info_dict: Raw observation dictionary from the environment.

        Returns:
            A dictionary of processed tensors.

        Raises:
            ValueError: If an expected numpy array is missing for processing.
        """
        self._align_proprio_to_processor(info_dict)

        for k, v in info_dict.items():
            is_numpy = isinstance(v, (np.ndarray | np.generic))

            if hasattr(self, 'process') and k in self.process:
                if not is_numpy:
                    raise ValueError(
                        f"Expected numpy array for key '{k}' in process, got {type(v)}"
                    )

                # flatten extra dimensions if needed
                shape = v.shape
                if len(shape) > 2:
                    v = v.reshape(-1, *shape[2:])

                # process and reshape back
                v = self.process[k].transform(v)
                v = v.reshape(shape)

            # collapse env and time dimensions for transform (e, t, ...) -> (e * t, ...)
            # then restore after transform
            if hasattr(self, 'transform') and k in self.transform:
                shape = None
                if is_numpy or torch.is_tensor(v):
                    if v.ndim > 2:
                        shape = v.shape
                        v = v.reshape(-1, *shape[2:])
                if k.startswith('pixels') or k.startswith('goal'):
                    # permute channel first for transform
                    if is_numpy:
                        v = np.transpose(v, (0, 3, 1, 2))
                    else:
                        v = v.permute(0, 3, 1, 2)
                v = torch.stack(
                    [self.transform[k](tv_tensors.Image(x)) for x in v]
                )
                is_numpy = isinstance(v, (np.ndarray | np.generic))

                if shape is not None:
                    v = v.reshape(*shape[:2], *v.shape[1:])

            if is_numpy and v.dtype.kind not in 'USO':
                v = torch.from_numpy(v)

            info_dict[k] = v

        return info_dict

    @staticmethod
    def _processor_dim(processor: Transformable | None) -> int | None:
        mean = getattr(processor, 'mean_', None)
        if mean is None:
            return None
        mean = np.asarray(mean)
        if mean.ndim == 0:
            return None
        return int(mean.shape[-1])

    @staticmethod
    def _merge_qpos_qvel_for_dim(
        info_dict: dict,
        *,
        prefix: str,
        target_dim: int,
    ) -> np.ndarray | None:
        qpos = info_dict.get(f'{prefix}qpos')
        qvel = info_dict.get(f'{prefix}qvel')
        if qpos is None or qvel is None:
            return None
        if not isinstance(qpos, (np.ndarray | np.generic)) or not isinstance(
            qvel,
            (np.ndarray | np.generic),
        ):
            return None

        candidates = (
            np.concatenate([qpos, qvel], axis=-1),
            np.concatenate([qpos[..., 2:], qvel], axis=-1),
        )
        for candidate in candidates:
            if candidate.shape[-1] == target_dim:
                return candidate
        return None

    def _align_proprio_to_processor(self, info_dict: dict) -> None:
        if not hasattr(self, 'process'):
            return

        for key, prefix in (('proprio', ''), ('goal_proprio', 'goal_')):
            if key not in info_dict or key not in self.process:
                continue
            value = info_dict[key]
            if not isinstance(value, (np.ndarray | np.generic)):
                continue

            target_dim = self._processor_dim(self.process.get(key))
            if target_dim is None or value.shape[-1] == target_dim:
                continue

            merged = self._merge_qpos_qvel_for_dim(
                info_dict,
                prefix=prefix,
                target_dim=target_dim,
            )
            if merged is not None:
                info_dict[key] = merged
                continue

            if value.shape[-1] - 2 == target_dim:
                info_dict[key] = value[..., 2:]


class RandomPolicy(BasePolicy):
    """Policy that samples random actions from the action space."""

    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        """Initialize the random policy.

        Args:
            seed: Optional random seed for the action space.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.type = 'random'
        self.seed = seed

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        """Get a random action from the environment's action space.

        Args:
            obs: The current observation (ignored).
            **kwargs: Additional parameters (ignored).

        Returns:
            A randomly sampled action.
        """
        return self.env.action_space.sample()

    def set_seed(self, seed: int) -> None:
        """Set the random seed for action sampling.

        Args:
            seed: The seed value.
        """
        if self.env is not None:
            self.env.action_space.seed(seed)


class WorldModelPolicy(BasePolicy):
    """Policy using a world model and planning solver for action selection."""

    def __init__(
        self,
        solver: Solver,
        config: PlanConfig | HierarchicalPlanConfig,
        process: dict[str, Transformable] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]]
        | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the world model policy.

        Args:
            solver: The planning solver to use.
            config: MPC planning configuration.
            process: Dictionary of data preprocessors for specific keys.
            transform: Dictionary of tensor transformations (e.g., image transforms).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)

        self.type = 'world_model'
        self.cfg = config
        self.solver = solver
        self.process = process or {}
        self.transform = transform or {}
        self._action_buffer: deque[torch.Tensor] | None = None
        self.eval_budget: int | None = getattr(self, 'eval_budget', None)
        self._steps_taken = 0
        self._policy_call_count = 0

    @property
    def execution_plan_config(self) -> PlanConfig:
        """Return the low-level plan config used to execute environment actions."""
        if isinstance(self.cfg, PlanConfig):
            return self.cfg

        if self.cfg.level1 is None:
            raise ValueError(
                'HierarchicalPlanConfig.level1 must be provided for action execution.'
            )

        return self.cfg.level1

    @property
    def flatten_receding_horizon(self) -> int:
        """Receding horizon in environment steps (with frameskip)."""
        plan_cfg = self.execution_plan_config
        return self.cfg.receding_horizon * plan_cfg.action_block

    def set_env(self, env: Any) -> None:
        """Configure the policy and solver for the given environment.

        Args:
            env: The environment to associate with the policy.
        """
        self.env = env
        n_envs = getattr(env, 'num_envs', 1)
        self.solver.configure(
            action_space=env.action_space, n_envs=n_envs, config=self.cfg
        )
        self.reset()

        assert isinstance(self.solver, Solver), (
            'Solver must implement the Solver protocol'
        )

    def reset(self) -> None:
        """Clear action buffers and step counters for a new rollout."""
        self._action_buffer = deque(maxlen=self.flatten_receding_horizon)
        self._steps_taken = 0
        self._policy_call_count = 0

    def _remaining_eval_steps(self) -> int | None:
        """Return the remaining environment-step budget, if one is set."""
        if self.eval_budget is None:
            return None
        return max(0, self.eval_budget - self._steps_taken)

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        """Get action via planning with the world model.

        Args:
            info_dict: Current state information from the environment.
            **kwargs: Additional parameters for planning.

        Returns:
            The selected action(s) as a numpy array.
        """
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'pixels' in info_dict, "'pixels' must be provided in info_dict"
        assert 'goal' in info_dict, "'goal' must be provided in info_dict"

        info_dict = self._prepare_info(info_dict)

        # need to replan if action buffer is empty
        if len(self._action_buffer) == 0:
            remaining_steps = self._remaining_eval_steps()
            if remaining_steps is not None and remaining_steps <= 0:
                raise RuntimeError('Evaluation step budget exhausted.')

            self._policy_call_count += 1
            print(f'policy call number: {self._policy_call_count}')
            outputs = self.solver(
                info_dict,
                init_action=None,
                steps_taken=self._steps_taken,
                eval_budget=self.eval_budget,
            )

            actions = outputs['actions']  # (num_envs, horizon, action_dim)
            keep_horizon = min(self.cfg.receding_horizon, actions.shape[1])
            plan = actions[:, :keep_horizon]

            # frameskip back to timestep
            plan = plan.reshape(
                self.env.num_envs,
                keep_horizon * self.execution_plan_config.action_block,
                -1,
            )

            self._action_buffer.extend(plan.transpose(0, 1))

        action = self._action_buffer.popleft()
        action = action.reshape(*self.env.action_space.shape)
        action = action.numpy()

        # post-process action
        if 'action' in self.process:
            action = self.process['action'].inverse_transform(action)

        self._steps_taken += 1
        return action  # (num_envs, action_dim)


def _load_model_with_attribute(run_name, attribute_name, cache_dir=None):
    """Helper function to load a model checkpoint and find a module with the specified attribute.

    Args:
        run_name: Path or name of the model run
        attribute_name: Name of the attribute to look for in the module (e.g., 'get_action', 'get_cost')
        cache_dir: Optional cache directory path

    Returns:
        The module with the specified attribute

    Raises:
        RuntimeError: If no module with the specified attribute is found
    """
    run_path = Path(run_name).expanduser()
    if run_path.suffix == '.ckpt':
        path = run_path
        if not path.exists() and not path.is_absolute():
            path = Path(cache_dir or swm.data.utils.get_cache_dir(), run_name)

        assert path.exists(), (
            f'Checkpoint path does not exist: {path}. Launch pretraining first.'
        )
    else:
        if not run_path.exists():
            run_path = Path(cache_dir or swm.data.utils.get_cache_dir(), run_name)

        if run_path.is_dir():
            ckpt_files = list(run_path.glob('*_object.ckpt'))
            ckpt_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
            path = ckpt_files[0]
            logging.info(f'Loading model from checkpoint: {path}')
        else:
            path = Path(f'{run_path}_object.ckpt')
            assert path.exists(), (
                f'Checkpoint path does not exist: {path}. Launch pretraining first.'
            )

    _register_legacy_checkpoint_module_aliases()
    spt_module = torch.load(path, weights_only=False, map_location='cpu')

    def scan_module(module):
        if hasattr(module, attribute_name):
            if isinstance(module, torch.nn.Module):
                module = module.eval()
            return module
        for child in module.children():
            result = scan_module(child)
            if result is not None:
                return result
        return None

    result = scan_module(spt_module)
    if result is not None:
        return result

    raise RuntimeError(
        f"No module with '{attribute_name}' found in the loaded world model."
    )


def AutoCostModel(
    run_name: str, cache_dir: str | Path | None = None
) -> torch.nn.Module:
    """Load a model checkpoint and return the module with a `get_cost` method.

    Automatically scans the checkpoint for a module implementing a cost function
    (i.e., has a `get_cost` method) for use with planning solvers.

    Args:
        run_name: Path or name of the model run/checkpoint.
        cache_dir: Optional cache directory path. Defaults to STABLEWM_HOME.

    Returns:
        The module with a `get_cost` method, set to eval mode.

    Raises:
        RuntimeError: If no module with `get_cost` is found in the checkpoint.
    """
    return _load_model_with_attribute(run_name, 'get_cost', cache_dir)


# Alias for backward compatibility and type hinting
Policy = BasePolicy
