"""World environment manager for vectorized Gymnasium environments."""

import hashlib
import json
import os
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import hdf5plugin
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from loguru import logger as logging
from rich import print
from tqdm import tqdm

from stable_worldmodel.data.utils import get_cache_dir
from stable_worldmodel.policy import Policy

from .wrapper import MegaWrapper, SyncWorld, VariationWrapper


EVAL_STATE_MODES = ('dataset_full', 'xy_neutral')
ORIGINAL_LENGTH_KEY = '_original_length'
ORIGINAL_LENGTHS_KEY = '_original_lengths'
VALID_LAST_INDEX_KEY = '_valid_last_index'


def _make_env(env_name, max_episode_steps, wrappers, **kwargs):
    """Create a gymnasium environment with specified wrappers.

    Factory function for creating environments within a vectorized setup.
    Creates the base environment and applies wrappers in order.

    Args:
        env_name: Name of the gymnasium environment to create.
        max_episode_steps: Maximum steps per episode before truncation.
        wrappers: List of wrapper functions/classes to apply. Each wrapper
            should accept an environment and return a wrapped environment.
        **kwargs: Additional keyword arguments passed to gym.make.

    Returns:
        The wrapped gymnasium environment.

    Example:
        >>> from functools import partial
        >>> wrappers = [partial(MegaWrapper, image_shape=(64, 64))]
        >>> env = _make_env("CartPole-v1", max_episode_steps=500, wrappers=wrappers)
    """
    env = gym.make(env_name, max_episode_steps=max_episode_steps, **kwargs)
    for wrapper in wrappers:
        env = wrapper(env)
    return env



def _find_env_method(env: gym.Env, method_name: str) -> Callable | None:
    method_owner = env
    while method_owner is not None and not hasattr(method_owner, method_name):
        next_env = getattr(method_owner, 'env', None)
        if next_env is method_owner:
            break
        method_owner = next_env

    if method_owner is not None and hasattr(method_owner, method_name):
        return getattr(method_owner, method_name)

    env_unwrapped = env.unwrapped
    if hasattr(env_unwrapped, method_name):
        return getattr(env_unwrapped, method_name)

    return None


def _validate_eval_state_mode(mode: str, mode_name: str) -> None:
    if mode not in EVAL_STATE_MODES:
        raise ValueError(
            f'{mode_name} must be one of {EVAL_STATE_MODES}, got {mode!r}.'
        )


def _episode_sequence_length(
    episode: dict[str, Any],
    columns: Sequence[str],
) -> int | None:
    keys = ['pixels', *[key for key in columns if key != 'pixels']]
    for key in keys:
        value = episode.get(key)
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return int(value.shape[0])
    return None


def _is_sequence_value(value: Any) -> bool:
    if torch.is_tensor(value):
        return value.ndim > 0
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return False


def _pad_time_sequence(value: Any, target_length: int, *, zero_pad: bool) -> Any:
    current_length = int(value.shape[0])
    pad_length = target_length - current_length
    if pad_length <= 0:
        return value

    if torch.is_tensor(value):
        if zero_pad:
            padding = torch.zeros(
                (pad_length, *value.shape[1:]),
                dtype=value.dtype,
                device=value.device,
            )
        else:
            padding = value[-1:].expand(pad_length, *value.shape[1:]).clone()
        return torch.cat([value, padding], dim=0)

    if zero_pad:
        padding = np.zeros((pad_length, *value.shape[1:]), dtype=value.dtype)
    else:
        padding = np.repeat(value[-1:], pad_length, axis=0)
    return np.concatenate([value, padding], axis=0)


def _pad_loaded_eval_trajectories(
    data: Sequence[dict[str, Any]],
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    lengths = [_episode_sequence_length(episode, columns) for episode in data]
    if any(length is None for length in lengths):
        raise ValueError('Could not infer lengths for all loaded eval trajectories.')

    original_episode_lengths = [int(length) for length in lengths if length is not None]
    if not original_episode_lengths:
        return list(data)

    max_episode_length = max(original_episode_lengths)
    if max_episode_length <= 0:
        raise ValueError('Loaded eval trajectories must have positive length.')

    target_lengths: dict[str, int] = {}
    for episode in data:
        for key, value in episode.items():
            if key.startswith('_') or not _is_sequence_value(value):
                continue
            target_lengths[key] = max(
                target_lengths.get(key, 0),
                int(value.shape[0]),
            )

    padded = []
    for episode, original_length in zip(data, original_episode_lengths):
        padded_episode = dict(episode)
        key_lengths = {
            key: int(value.shape[0])
            for key, value in episode.items()
            if not key.startswith('_') and _is_sequence_value(value)
        }
        padded_episode[ORIGINAL_LENGTH_KEY] = original_length
        padded_episode[ORIGINAL_LENGTHS_KEY] = key_lengths
        padded_episode[VALID_LAST_INDEX_KEY] = original_length - 1

        for key, value in list(padded_episode.items()):
            target_length = target_lengths.get(key)
            if key.startswith('_') or target_length is None:
                continue
            if not _is_sequence_value(value) or int(value.shape[0]) >= target_length:
                continue
            padded_episode[key] = _pad_time_sequence(
                value,
                target_length,
                zero_pad=key == 'action',
            )
        padded.append(padded_episode)

    if len(set(original_episode_lengths)) > 1:
        logging.info(
            'Padded loaded eval trajectories to length {} from original lengths {}.',
            max_episode_length,
            sorted(set(original_episode_lengths)),
        )

    return padded


class World:
    """High-level manager for vectorized Gymnasium environments.

    Manages a set of synchronized vectorized environments with automatic
    preprocessing (resizing, frame stacking, goal conditioning).

    Args:
        env_name: Name of the Gymnasium environment to create.
        num_envs: Number of parallel environments.
        image_shape: Target shape for image observations (H, W).
        seed: Random seed for reproducibility.
        history_size: Number of frames to stack.
        frame_skip: Number of frames to skip per step.
        max_episode_steps: Maximum steps per episode before truncation.
        verbose: Verbosity level (0: silent, >0: info).
        **kwargs: Additional keyword arguments passed to `gym.make_vec`.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        image_shape: tuple[int, int],
        seed: int = 2349867,
        history_size: int = 1,
        frame_skip: int = 1,
        max_episode_steps: int = 100,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        wrappers = [
            partial(
                MegaWrapper,
                image_shape=image_shape,
                history_size=history_size,
                frame_skip=frame_skip,
            ),
        ]

        env_fn = partial(
            _make_env, env_name, max_episode_steps, wrappers, **kwargs
        )
        env_fns = [env_fn for _ in range(num_envs)]
        self.envs: VectorEnv = VariationWrapper(SyncWorld(env_fns))
        self.envs.unwrapped.autoreset_mode = gym.vector.AutoresetMode.DISABLED

        self._history_size = history_size
        self.policy: Policy | None = None
        self.states: dict | None = None
        self.infos: dict = {}
        self.rewards: np.ndarray | None = None
        self.terminateds: np.ndarray | None = None
        self.truncateds: np.ndarray | None = None

        if verbose > 0:
            logging.info(f'🌍🌍🌍 World {env_name} initialized 🌍🌍🌍')

            logging.info('🕹️ 🕹️ 🕹️ Action space 🕹️ 🕹️ 🕹️')
            logging.info(f'{self.envs.action_space}')

            logging.info('👁️ 👁️ 👁️ Observation space 👁️ 👁️ 👁️')
            logging.info(f'{str(self.envs.observation_space)}')

            if self.envs.variation_space is not None:
                logging.info('⚗️ ⚗️ ⚗️ Variation space ⚗️ ⚗️ ⚗️')
                print(self.envs.single_variation_space.to_str())
            else:
                logging.warning('No variation space provided!')

        self.seed = seed

    @property
    def num_envs(self) -> int:
        """Number of parallel environment instances."""
        return self.envs.num_envs

    def close(self, **kwargs: Any) -> None:
        """Close all environments and clean up resources."""
        return self.envs.close(**kwargs)

    def step(self) -> None:
        """Advance all environments by one step using the current policy."""
        # note: reset happens before because of auto-reset, should fix that
        if self.policy is None:
            raise RuntimeError('No policy set. Call set_policy() first.')

        actions = self.policy.get_action(self.infos)
        (
            self.states,
            self.rewards,
            self.terminateds,
            self.truncateds,
            self.infos,
        ) = self.envs.step(actions)

    def reset(
        self,
        seed: int | list[int] | None = None,
        options: dict | None = None,
    ) -> None:
        """Reset all environments to initial states.

        Args:
            seed: Random seed(s) for the environments.
            options: Additional options passed to the environment reset.
        """
        if self.policy is not None:
            self.policy.reset()
        self.states, self.infos = self.envs.reset(seed=seed, options=options)

    def set_policy(self, policy: Policy) -> None:
        """Attach a policy to the world.

        Args:
            policy: The policy instance to use for determining actions.
        """
        self.policy = policy
        self.policy.set_env(self.envs)

        if hasattr(self.policy, 'seed') and self.policy.seed is not None:
            self.policy.set_seed(self.policy.seed)

    def record_dataset(
        self,
        dataset_name: str,
        episodes: int = 10,
        seed: int | None = None,
        cache_dir: os.PathLike | str | None = None,
        options: dict | None = None,
    ) -> None:
        """Records episodes from the environment into an HDF5 dataset.

        Args:
            dataset_name: Name of the dataset file (without extension).
            episodes: Total number of episodes to record.
            seed: Initial random seed.
            cache_dir: Directory to save the dataset. Defaults to standard cache.
            options: Reset options passed to environments.

        Raises:
            NotImplementedError: If history_size > 1.
        """
        if self._history_size > 1:
            raise NotImplementedError(
                'Frame history > 1 not supported for dataset recording.'
            )

        path = Path(cache_dir or get_cache_dir()) / f'{dataset_name}.h5'
        path.parent.mkdir(parents=True, exist_ok=True)

        self.terminateds = np.zeros(self.num_envs, dtype=bool)
        self.truncateds = np.zeros(self.num_envs, dtype=bool)

        episode_buffers = [defaultdict(list) for _ in range(self.num_envs)]

        h5_kwargs = {
            'name': str(path),
            'mode': 'a' if path.exists() else 'w',
            'libver': 'latest',
        }

        if not path.exists():  # creation only args
            h5_kwargs.update(
                {'fs_strategy': 'page', 'fs_page_size': 4 * 1024 * 1024}
            )

        with h5py.File(**h5_kwargs) as f:
            f.swmr_mode = True  # avoid issue when killed

            if 'ep_len' in f:
                n_ep_recorded = f['ep_len'].shape[0]
                global_step_ptr = (
                    f['ep_offset'][-1] + f['ep_len'][-1]
                    if n_ep_recorded > 0
                    else 0
                )
                initialized = True
                seed = None if seed is None else (seed + n_ep_recorded)
                logging.info(
                    f'Resuming: {n_ep_recorded} episodes already on disk.'
                )
            else:
                n_ep_recorded = 0
                global_step_ptr = 0
                initialized = False

            self.reset(seed, options=options)
            seed = None if seed is None else (seed + self.num_envs)
            self._dump_step_data(episode_buffers)  # record initial state

            with tqdm(
                total=episodes, initial=n_ep_recorded, desc='Recording'
            ) as pbar:
                while n_ep_recorded < episodes:
                    self.step()
                    self._dump_step_data(episode_buffers)

                    for i in range(self.num_envs):
                        if self.terminateds[i] or self.truncateds[i]:
                            finished_ep = self._handle_done_ep(
                                episode_buffers, i, n_ep_recorded
                            )

                            # lazy dataset initialization
                            if not initialized:
                                self._init_h5_datasets(f, finished_ep)
                                initialized = True

                            # contiguous writing
                            steps_written = self._write_episode(
                                f, finished_ep, global_step_ptr
                            )
                            global_step_ptr += steps_written
                            n_ep_recorded += 1
                            pbar.update(1)

                            f.flush()  # flush metadata to avoid corruption

                            if n_ep_recorded >= episodes:
                                break

                            # reset terminated env and record initial state
                            n_seed = (
                                None
                                if seed is None
                                else (seed + n_ep_recorded)
                            )
                            self._reset_single_env(i, n_seed, options)
                            self._dump_step_data(episode_buffers, env_idx=i)

        logging.info(f'Recording complete. Total frames: {global_step_ptr}')

    def _init_h5_datasets(
        self, f: h5py.File, sample_episode: dict[str, list[Any]]
    ) -> None:
        """Initialize resizable HDF5 datasets based on the first episode.

        Args:
            f: The open HDF5 file handle.
            sample_episode: A dictionary containing data from a single episode,
                used to determine shapes and dtypes.
        """
        for key, data_list in sample_episode.items():
            if key in ['ep_len', 'ep_idx', 'policy']:
                continue

            key = key.replace('/', '_')  # sanitize keys for h5

            # determine array shape and dtype from sample data
            sample_data = np.array(data_list[0])
            shape = (0,) + sample_data.shape
            maxshape = (None,) + sample_data.shape

            # determine chunk size and compression
            if sample_data.ndim >= 2:
                chunks = (100,) + sample_data.shape
                compression = hdf5plugin.Blosc(
                    cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
                )

            else:
                chunks = (1000,) + sample_data.shape
                compression = None

            dtype = sample_data.dtype
            if np.issubdtype(dtype, np.str_) or np.issubdtype(
                dtype, np.bytes_
            ):
                dtype = h5py.string_dtype()

            f.create_dataset(
                key,
                shape=shape,
                maxshape=maxshape,
                dtype=dtype,
                chunks=chunks,
                compression=compression,
            )

        # index metadata
        f.create_dataset(
            'ep_offset', shape=(0,), maxshape=(None,), dtype=np.int64
        )
        f.create_dataset(
            'ep_len', shape=(0,), maxshape=(None,), dtype=np.int32
        )

        # per-step episode index
        f.create_dataset(
            'ep_idx',
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(1000,),
        )

    def _reset_single_env(
        self,
        env_idx: int,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        """Reset a single environment and update infos dict.

        Args:
            env_idx: Index of the environment to reset.
            seed: Random seed for this specific environment.
            options: Reset options.
        """
        self.envs.unwrapped._autoreset_envs = np.zeros(self.num_envs)
        _, infos = self.envs.envs[env_idx].reset(seed=seed, options=options)

        for k, v in infos.items():
            if k in self.infos:
                self.infos[k][env_idx] = v

        if self.policy is not None and hasattr(self.policy, 'reset_env'):
            self.policy.reset_env(env_idx)

    def _apply_xy_neutral_eval_state(
        self,
        init_step: dict[str, np.ndarray],
        goal_step: dict[str, np.ndarray],
        start_state_mode: str,
        goal_state_mode: str,
    ) -> None:
        _validate_eval_state_mode(start_state_mode, 'start_state_mode')
        _validate_eval_state_mode(goal_state_mode, 'goal_state_mode')

        if (
            start_state_mode == 'dataset_full'
            and goal_state_mode == 'dataset_full'
        ):
            return

        if start_state_mode == 'xy_neutral' and 'xy' not in init_step:
            raise ValueError(
                "start_state_mode='xy_neutral' requires dataset column 'xy'."
            )
        if goal_state_mode == 'xy_neutral' and 'goal_xy' not in goal_step:
            raise ValueError(
                "goal_state_mode='xy_neutral' requires dataset column 'xy'."
            )

        for env_idx, env in enumerate(self.envs.unwrapped.envs):
            xy_info_fn = _find_env_method(env, 'settled_info_from_xy')
            if xy_info_fn is None:
                xy_info_fn = _find_env_method(env, 'neutral_info_from_xy')
            if xy_info_fn is None:
                raise ValueError(
                    'xy_neutral eval state mode requires an environment with '
                    'settled_info_from_xy() or neutral_info_from_xy().'
                )

            if start_state_mode == 'xy_neutral':
                info = xy_info_fn(
                    init_step['xy'][env_idx],
                    render_pixels='pixels' in init_step,
                )
                self._write_neutral_step_info(init_step, env_idx, info, '')

            if goal_state_mode == 'xy_neutral':
                info = xy_info_fn(
                    goal_step['goal_xy'][env_idx],
                    render_pixels='goal' in goal_step,
                )
                self._write_neutral_step_info(goal_step, env_idx, info, 'goal_')

    @staticmethod
    def _proprio_from_qpos_qvel(
        info: dict[str, np.ndarray],
        target_dim: int,
    ) -> np.ndarray | None:
        qpos = info.get('qpos')
        qvel = info.get('qvel')
        if qpos is None or qvel is None:
            return None

        candidates = (
            np.concatenate([qpos, qvel], axis=-1),
            np.concatenate([qpos[..., 2:], qvel], axis=-1),
        )
        for candidate in candidates:
            if candidate.shape[-1] == target_dim:
                return candidate
        return None

    @classmethod
    def _match_proprio_dim(
        cls,
        value: np.ndarray,
        target_dim: int,
        info: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if value.shape[-1] == target_dim:
            return value

        if info is not None:
            candidate = cls._proprio_from_qpos_qvel(info, target_dim)
            if candidate is not None:
                while candidate.ndim > value.ndim:
                    candidate = candidate[:, -1]
                return candidate

        if value.shape[-1] - 2 == target_dim:
            return value[..., 2:]

        return value

    @classmethod
    def _write_neutral_step_info(
        cls,
        step: dict[str, np.ndarray],
        env_idx: int,
        neutral_info: dict[str, np.ndarray],
        prefix: str,
    ) -> None:
        key_pairs = (
            ('xy', 'xy'),
            ('qpos', 'qpos'),
            ('qvel', 'qvel'),
            ('prev_qpos', 'qpos'),
            ('prev_qvel', 'qvel'),
            ('proprio', 'proprio'),
            ('pixels', 'pixels'),
        )

        for step_key, info_key in key_pairs:
            target_key = (
                'goal' if prefix == 'goal_' and step_key == 'pixels'
                else f'{prefix}{step_key}'
            )
            if target_key in step and info_key in neutral_info:
                value = neutral_info[info_key]
                if step_key == 'proprio':
                    value = cls._match_proprio_dim(
                        value,
                        step[target_key].shape[-1],
                        neutral_info,
                    )
                step[target_key][env_idx] = value

    def _handle_done_ep(
        self,
        tmp_buffer: list[dict[str, list[Any]]],
        env_idx: int,
        n_ep_recorded: int,
    ) -> dict[str, list[Any]]:
        """Prepare the episode buffer for writing.

        Args:
            tmp_buffer: List of dictionaries accumulating step data per env.
            env_idx: Index of the environment that finished an episode.
            n_ep_recorded: Number of episodes recorded so far.

        Returns:
            A dictionary containing the complete episode data.
        """
        ep_buffer = tmp_buffer[env_idx]
        ep_len = len(ep_buffer['step_idx'])

        # left-shift actions to align with observations i.e. (o_t, a_t).
        # Some policies expose action-side metadata, such as AntMaze explore
        # directions, that should be shifted with actions.
        action_aligned_keys = ['action']
        if self.policy is not None:
            action_aligned_keys.extend(
                getattr(self.policy, 'action_aligned_info_keys', ())
            )
        for key in action_aligned_keys:
            if key in ep_buffer:
                values = ep_buffer[key]
                if len(values) == ep_len:
                    nan = values.pop(0)
                    values.append(nan)
                elif len(values) == ep_len - 1 and values:
                    values.append(np.full_like(values[-1], np.nan))

        # Extract a copy and clear the temporary buffer
        out = {k: list(v) for k, v in ep_buffer.items()}
        ep_buffer.clear()
        self.terminateds[env_idx] = False
        self.truncateds[env_idx] = False

        # Add episode index to all steps
        out['ep_idx'] = [n_ep_recorded] * ep_len

        return out

    def _write_episode(
        self, f: h5py.File, ep_data: dict[str, list[Any]], global_ptr: int
    ) -> int:
        """Write a single contiguous episode to the HDF5 file.

        Args:
            f: The open HDF5 file handle.
            ep_data: The episode data dictionary.
            global_ptr: The global step index where this episode starts.

        Returns:
            The length of the episode written.
        """
        ep_len = len(ep_data['step_idx'])

        # append data to each dataset
        for key in ep_data:
            h5_key = key.replace('/', '_')  # sanitize keys for h5
            if h5_key in ['ep_len', 'policy']:
                continue

            ds = f[h5_key]
            curr_size = ds.shape[0]
            ds.resize(curr_size + ep_len, axis=0)
            ds[curr_size:] = np.array(ep_data[key])

        # update metadata
        meta_idx = f['ep_offset'].shape[0]
        f['ep_offset'].resize(meta_idx + 1, axis=0)
        f['ep_len'].resize(meta_idx + 1, axis=0)

        f['ep_offset'][meta_idx] = global_ptr
        f['ep_len'][meta_idx] = ep_len

        return ep_len

    def _dump_step_data(
        self,
        tmp_buffer: list[dict[str, list[Any]]],
        env_idx: int | None = None,
    ) -> None:
        """Append current step data to temporary episode buffers.

        Args:
            tmp_buffer: List of dictionaries accumulating step data.
            env_idx: Optional index to dump data for a single environment.
                If None, dumps for all environments.
        """
        env_indices = range(self.num_envs) if env_idx is None else [env_idx]

        for col, data in self.infos.items():
            if col.startswith('_'):
                continue

            # normalize data shape and type
            if isinstance(data, np.ndarray):
                data = (
                    np.squeeze(data, axis=1)
                    if data.ndim > 1 and data.shape[1] == 1
                    else data
                )
                if data.dtype == object:
                    data = np.concatenate(data).tolist()

            # append to buffers
            for i in env_indices:
                env_data = (
                    data[i].copy()
                    if isinstance(data[i], np.ndarray)
                    else data[i]
                )
                tmp_buffer[i][col].append(env_data)

    def evaluate(
        self,
        episodes: int = 10,
        eval_keys: list[str] | None = None,
        seed: int | None = None,
        options: dict | None = None,
        dump_every: int = -1,
    ) -> dict:
        """Evaluate the current policy over multiple episodes.

        Args:
            episodes: Number of episodes to evaluate.
            eval_keys: List of keys in `infos` to collect and return.
            seed: Random seed for evaluation.
            options: Reset options.
            dump_every: Interval to save intermediate results (for long evals).

        Returns:
            Dictionary containing success rates, seeds, and collected keys.
        """
        options = options or {}

        results: dict = {
            'episode_count': 0,
            'success_rate': 0.0,
            'episode_successes': np.zeros(episodes),
            'seeds': np.zeros(episodes, dtype=np.int32),
        }

        if eval_keys:
            for key in eval_keys:
                results[key] = np.zeros(episodes)

        self.terminateds = np.zeros(self.num_envs)
        self.truncateds = np.zeros(self.num_envs)

        episode_idx = np.arange(self.num_envs)
        self.reset(seed=seed, options=options)
        root_seed = seed + self.num_envs if seed is not None else None

        eval_done = False

        # determine "unique" hash for this eval run
        config = {
            'episodes': episodes,
            'eval_keys': tuple(sorted(eval_keys)) if eval_keys else None,
            'seed': seed,
            'options': tuple(sorted(options.items())) if options else None,
            'dump_every': dump_every,
        }

        config_str = json.dumps(config, sort_keys=True)
        run_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        run_tmp_path = Path(f'eval_tmp_{run_hash}.npy')

        # load back intermediate results if file exists
        if run_tmp_path.exists():
            tmp_results = np.load(run_tmp_path, allow_pickle=True).item()
            results.update(tmp_results)

            ep_count = results['episode_count']
            episode_idx = np.arange(ep_count, ep_count + self.num_envs)

            # reset seed where we left off
            last_seed = seed + ep_count if seed is not None else None
            self.reset(seed=last_seed, options=options)

            logging.success(
                f'Found existing eval tmp file {run_tmp_path}, resuming from episode {ep_count}/{episodes}'
            )

        while True:
            self.step()

            # start new episode for done envs
            for i in range(self.num_envs):
                if self.terminateds[i] or self.truncateds[i]:
                    # record eval info
                    ep_idx = episode_idx[i]
                    results['episode_successes'][ep_idx] = self.terminateds[i]
                    results['seeds'][ep_idx] = self.envs.envs[
                        i
                    ].unwrapped.np_random_seed

                    if eval_keys:
                        for key in eval_keys:
                            assert key in self.infos, (
                                f'key {key} not found in infos'
                            )
                            results[key][ep_idx] = self.infos[key][i]

                    # determine new episode idx
                    # re-reset env with seed and options (no supported by auto-reset)
                    new_seed = (
                        root_seed + results['episode_count']
                        if seed is not None
                        else None
                    )
                    next_ep_idx = episode_idx.max() + 1
                    episode_idx[i] = next_ep_idx
                    results['episode_count'] += 1

                    # break if enough episodes evaluated
                    if results['episode_count'] >= episodes:
                        eval_done = True
                        if run_tmp_path.exists():
                            logging.info(
                                f'Eval done, deleting tmp file {run_tmp_path}'
                            )
                            os.remove(run_tmp_path)
                        break

                    # dump temporary results in a file
                    if dump_every > 0 and (
                        results['episode_count'] % dump_every == 0
                    ):
                        np.save(run_tmp_path, results)
                        logging.success(
                            f'Dumped intermediate eval results to {run_tmp_path} ({results["episode_count"]}/{episodes})'
                        )
                    self.envs.unwrapped._autoreset_envs = np.zeros(
                        (self.num_envs,)
                    )
                    _, infos = self.envs.envs[i].reset(
                        seed=new_seed, options=options
                    )

                    for k, v in infos.items():
                        if k not in self.infos:
                            continue
                        # Convert to array and extract scalar to preserve dtype
                        self.infos[k][i] = np.asarray(v)

            if eval_done:
                break

        # compute success rate
        results['success_rate'] = (
            float(np.sum(results['episode_successes'])) / episodes * 100.0
        )

        assert results['episode_count'] == episodes, (
            f'episode_count {results["episode_count"]} != episodes {episodes}'
        )

        assert np.unique(results['seeds']).shape[0] == episodes, (
            'Some episode seeds are identical!'
        )

        return results

    def evaluate_from_dataset(
        self,
        dataset: Any | None,
        episodes_idx: Sequence[int] | None,
        start_steps: Sequence[int] | None,
        goal_offset_steps: int,
        eval_budget: int,
        callables: list[dict] | None = None,
        dump_eval_trajs_path: str | Path | None = None,
        load_eval_trajs_path: str | Path | dict | None = None,
        process: dict[str, Any] | None = None,
        start_state_mode: str = 'dataset_full',
        goal_state_mode: str = 'dataset_full',
    ) -> dict:
        """Evaluate the policy starting from states sampled from a dataset.

        Args:
            dataset: The source dataset to sample initial states/goals from.
            episodes_idx: Indices of episodes to sample from.
            start_steps: Step indices within those episodes to start from.
            goal_offset_steps: Number of steps ahead to look for the goal.
            eval_budget: Maximum steps allowed for the agent to reach the goal.
            callables: Optional list of method calls to setup the env.

        Returns:
            Dictionary containing success rates and other metrics.

        Raises:
            ValueError: If input sequence lengths mismatch or don't match num_envs.
        """
        assert (
            self.envs.envs[0].spec.max_episode_steps is None
            or self.envs.envs[0].spec.max_episode_steps >= goal_offset_steps
        ), 'env max_episode_steps must be greater than eval_budget'

        if load_eval_trajs_path is not None:
            # a dict is a payload preloaded by the caller (chunked eval)
            payload = (
                load_eval_trajs_path
                if isinstance(load_eval_trajs_path, dict)
                else torch.load(
                    Path(load_eval_trajs_path), map_location='cpu', weights_only=False
                )
            )
            data = payload['data']
            columns = payload['columns']
            ep_idx_arr = np.array(payload.get('episodes_idx', []), dtype=np.int64)
            if ep_idx_arr.size == 0:
                ep_idx_arr = np.arange(len(data), dtype=np.int64)

            if len(data) < self.num_envs:
                raise ValueError(
                    f'Loaded eval trajectories contain {len(data)} episodes, '
                    f'fewer than num_envs={self.num_envs}. Decrease eval.num_eval.'
                )

            if len(data) > self.num_envs:
                logging.info(
                    f'Loaded {len(data)} episodes; using the first {self.num_envs} '
                    'to match num_envs.'
                )
                data = data[:self.num_envs]
                ep_idx_arr = ep_idx_arr[:self.num_envs]

            data = _pad_loaded_eval_trajectories(data, columns)
        else:
            if dataset is None:
                raise ValueError('dataset must be provided when not loading eval trajectories')
            if episodes_idx is None or start_steps is None:
                raise ValueError('episodes_idx and start_steps must be provided when not loading eval trajectories')

            ep_idx_arr = np.array(episodes_idx)
            start_steps_arr = np.array(start_steps)
            end_steps = start_steps_arr + goal_offset_steps

            if not (len(ep_idx_arr) == len(start_steps_arr)):
                raise ValueError(
                    'episodes_idx and start_steps must have the same length'
                )

            if len(ep_idx_arr) != self.num_envs:
                raise ValueError(
                    'Number of episodes to evaluate must match number of envs'
                )

            data = dataset.load_chunk(ep_idx_arr, start_steps_arr, end_steps)
            columns = dataset.column_names

            if dump_eval_trajs_path is not None:
                dump_path = Path(dump_eval_trajs_path)
                dump_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    'data': data,
                    'columns': list(columns),
                    'episodes_idx': ep_idx_arr.tolist(),
                    'start_steps': start_steps_arr.tolist(),
                    'goal_offset_steps': int(goal_offset_steps),
                    'eval_budget': int(eval_budget),
                    'process': process,
                }
                torch.save(payload, dump_path)
                logging.success(f'Dumped eval trajectories to {dump_path}')
                return {
                    'dumped_eval_trajs_path': str(dump_path),
                    'num_episodes': int(len(ep_idx_arr)),
                }

        # keep relevant part of the chunk
        init_step_per_env: dict[str, list[Any]] = defaultdict(list)
        goal_step_per_env: dict[str, list[Any]] = defaultdict(list)

        for i, ep in enumerate(data):
            for col in columns:
                if col.startswith('goal'):
                    continue
                if col.startswith('pixels'):
                    # permute channel to be last
                    ep[col] = ep[col].permute(0, 2, 3, 1)

                if not isinstance(ep[col], (torch.Tensor | np.ndarray)):
                    continue

                init_data = ep[col][0]
                goal_data = ep[col][-1]

                # TODO handle that better
                if not isinstance(init_data, (np.ndarray | torch.Tensor)):
                    logging.warning(
                        f'Data type {type(init_data)} for column {col} not supported, yet skipping conversion'
                    )
                    continue

                init_data = (
                    init_data.numpy()
                    if isinstance(init_data, torch.Tensor)
                    else init_data
                )
                goal_data = (
                    goal_data.numpy()
                    if isinstance(goal_data, torch.Tensor)
                    else goal_data
                )

                init_step_per_env[col].append(init_data)
                goal_step_per_env[col].append(goal_data)

        init_step = {
            k: np.stack(v) for k, v in deepcopy(init_step_per_env).items()
        }

        if 'step_idx' in init_step:
            init_step['dataset_step_idx'] = init_step['step_idx'].copy()
            init_step['step_idx'] = np.zeros_like(init_step['step_idx'])

        goal_step = {}
        for key, value in goal_step_per_env.items():
            key = 'goal' if key == 'pixels' else f'goal_{key}'
            goal_step[key] = np.stack(value)

        # get dataset info
        seeds = init_step.get('seed')
        # get dataset variation
        vkey = 'variation.'
        variations_dict = {
            k.removeprefix(vkey): v
            for k, v in init_step.items()
            if k.startswith(vkey)
        }

        options = [{} for _ in range(self.num_envs)]

        if len(variations_dict) > 0:
            for i in range(self.num_envs):
                options[i]['variation'] = list(variations_dict.keys())
                options[i]['variation_values'] = {
                    k: v[i] for k, v in variations_dict.items()
                }

        self.reset(seed=seeds, options=options)  # set seeds for all envs
        self._apply_xy_neutral_eval_state(
            init_step,
            goal_step,
            start_state_mode,
            goal_state_mode,
        )
        init_step.update(deepcopy(goal_step))

        # apply callable list (e.g used for set initial position if not access to seed)
        callables = callables or []
        for i, env in enumerate(self.envs.unwrapped.envs):
            env_unwrapped = env.unwrapped

            set_closed_door = _find_env_method(env, '_set_closed_door_idx')
            if set_closed_door is not None and 'closed_door_idx' in init_step:
                set_closed_door(deepcopy(init_step['closed_door_idx'][i]))

            for spec in callables:
                method_name = spec['method']
                method = _find_env_method(env, method_name)
                if method is None:
                    logging.warning(
                        f'Env {env} has no method {method_name}, skipping callable'
                    )
                    continue

                args = spec.get('args', spec)

                # prepare args
                prepared_args = {}
                for args_name, args_data in args.items():
                    value = args_data.get('value', None)
                    is_in_datset = args_data.get('in_dataset', True)

                    if is_in_datset:
                        if value not in init_step:
                            logging.warning(
                                f'Col {value} not found in dataset, skipping callable for env {env_unwrapped}'
                            )
                            continue
                        prepared_args[args_name] = deepcopy(
                            init_step[value][i]
                        )
                    else:
                        prepared_args[args_name] = args_data.get('value')

                # call method with prepared args
                method(**prepared_args)

            set_goal_info = _find_env_method(env, 'set_current_goal_info')
            if set_goal_info is not None:
                goal_info = {}
                for key in (
                    'goal',
                    'goal_xy',
                    'goal_qpos',
                    'goal_qvel',
                    'goal_proprio',
                ):
                    if key in init_step:
                        goal_info[key] = deepcopy(init_step[key][i])
                if goal_info:
                    set_goal_info(**goal_info)

        results: dict = {
            'success_rate': 0.0,
            'episode_successes': np.zeros(len(ep_idx_arr)),
            'seeds': seeds,
        }

        # expend all data to the right shape (x, y, (original_shape))
        shape_prefix = self.infos['pixels'].shape[:2]

        # TODO get the data from the previous step in the dataset for history
        init_step = {
            k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:])
            for k, v in init_step.items()
        }
        goal_step = {
            k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:])
            for k, v in goal_step.items()
        }

        # update the reset with our new init and goal infos
        self.infos.update(deepcopy(init_step))
        self.infos.update(deepcopy(goal_step))

        if 'goal' in goal_step and 'goal' in self.infos:
            assert np.allclose(self.infos['goal'], goal_step['goal']), (
                'Goal info does not match'
            )

        original_lengths = np.array(
            [
                int(ep.get(ORIGINAL_LENGTH_KEY, ep['pixels'].shape[0]))
                for ep in data
            ],
            dtype=np.int64,
        )
        wall_clock_to_success = np.full(self.num_envs, np.nan, dtype=np.float64)
        steps_to_success = np.full(self.num_envs, np.nan, dtype=np.float64)

        eval_start_time = time.perf_counter()
        for i in range(eval_budget):
            self.infos.update(deepcopy(goal_step))
            self.step()
            newly_successful = np.logical_and(
                self.terminateds, np.isnan(wall_clock_to_success)
            )
            wall_clock_to_success[newly_successful] = (
                time.perf_counter() - eval_start_time
            )
            steps_to_success[newly_successful] = i + 1

            results['episode_successes'] = np.logical_or(
                results['episode_successes'], self.terminateds
            )
            # for auto-reset
            self.envs.unwrapped._autoreset_envs = np.zeros((self.num_envs,))

        n_episodes = len(ep_idx_arr)

        # compute success rate
        results['success_rate'] = (
            float(np.sum(results['episode_successes'])) / n_episodes * 100.0
        )
        results['wall_clock_to_success'] = wall_clock_to_success
        results['steps_to_success'] = steps_to_success
        results['loaded_eval_original_lengths'] = original_lengths
        successful_steps = steps_to_success[np.isfinite(steps_to_success)]
        results['steps_to_success_success_only'] = (
            float(np.mean(successful_steps))
            if successful_steps.size > 0
            else float('nan')
        )

        if results['seeds'] is not None:
            assert np.unique(results['seeds']).shape[0] == n_episodes, (
                'Some episode seeds are identical!'
            )

        return results
