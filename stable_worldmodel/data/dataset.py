"""Dataset classes for episode-based reinforcement learning data."""

import logging
from collections.abc import Callable
from pathlib import Path
import re
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from omegaconf import OmegaConf

from stable_worldmodel.data.utils import get_cache_dir


class Dataset:
    """Base class for episode-based datasets.

    Args:
        lengths: Array of episode lengths.
        offsets: Array of episode start offsets in the data.
        frameskip: Number of frames to skip between samples.
        num_steps: Number of steps per sample.
        transform: Optional transform to apply to loaded data.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        offsets: np.ndarray,
        # frameskip: int = 1,
        # num_steps: int = 1,
        level1: dict | None = None,
        level2: dict | None = None,
        precompute_levels: bool = True,
        transform: Callable[[dict], dict] | None = None,
        **level_kwargs,
    ) -> None:
        self.lengths = lengths
        self.offsets = offsets
        self.precompute_levels = bool(precompute_levels)

        def _normalize_level(level_cfg: dict | None, level_name: str) -> dict | None:
            if level_cfg is None:
                return None

            # Convert to a plain dict so we can attach derived fields without
            # mutating Hydra/OmegaConf struct configs.
            if OmegaConf.is_config(level_cfg):
                normalized = OmegaConf.to_container(level_cfg, resolve=True)
            else:
                normalized = dict(level_cfg)

            if not isinstance(normalized, dict):
                raise TypeError(f"{level_name} must be a mapping")

            normalized["frameskip"] = int(normalized["frameskip"])
            normalized["num_steps"] = int(normalized["num_steps"])
            normalized["window_size"] = int(normalized.get("window_size", normalized.get("kernel_size", 1)))
            return normalized

        level_inputs = {"level1": level1, "level2": level2}
        for level_name, level_cfg in level_kwargs.items():
            match = re.fullmatch(r"level([1-9]\d*)", level_name)
            if match is None:
                raise TypeError(f"Unexpected dataset argument: {level_name}")
            level_inputs[level_name] = level_cfg

        levels_by_idx = {}
        for level_name, level_cfg in level_inputs.items():
            if level_cfg is None:
                continue
            level_idx = int(level_name.removeprefix("level"))
            levels_by_idx[level_idx] = _normalize_level(level_cfg, level_name)

        if not levels_by_idx:
            raise ValueError("At least level1 must be configured")

        max_level = max(levels_by_idx)
        missing_levels = [
            level_idx for level_idx in range(1, max_level + 1)
            if level_idx not in levels_by_idx
        ]
        if missing_levels:
            raise ValueError(f"Missing dataset level configs: {missing_levels}")

        levels = [levels_by_idx[level_idx] for level_idx in range(1, max_level + 1)]
        self.level1 = None
        self.level2 = None
        for level_idx, level in enumerate(levels, start=1):
            setattr(self, f"level{level_idx}", level)

        self.levels = len(levels)
        self.last_level = levels[-1]
        self.level_configs = levels

        for i, level in enumerate(levels):
            if i:
                level['effective_frameskip'] = level['frameskip'] * levels[i-1]['effective_frameskip']
            else:
                level['effective_frameskip'] = level['frameskip']

        def _required_level1_frames(level_idx: int) -> int:
            required = int(levels[level_idx - 1]['num_steps'])
            for upper_idx in range(level_idx - 1, 0, -1):
                upper_level = levels[upper_idx]
                required = (
                    (required - 1) * int(upper_level['frameskip'])
                    + int(upper_level['window_size'])
                )
            return required

        required_level1_frames = max(
            _required_level1_frames(level_idx)
            for level_idx in range(1, self.levels + 1)
        )
        self.last_level['span'] = (
            required_level1_frames * self.level1['frameskip']
        )

        self.clip_indices = [
            (ep, start)
            for ep, length in enumerate(lengths)
            if length >= self.last_level['span']
            for start in range(length - self.last_level['span'] + 1)
        ]
            
        self.transform = transform

        # self.frameskip = frameskip
        # self.num_steps = num_steps
        # self.span = num_steps * frameskip
        # self.clip_indices = [
        #     (ep, start)
        #     for ep, length in enumerate(lengths)
        #     if length >= self.span
        #     for start in range(length - self.span + 1)
        # ]

    @property
    def column_names(self) -> list[str]:
        raise NotImplementedError

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.clip_indices[idx]
        # steps = self._load_slice(ep_idx, start, start + self.span)
        # if 'action' in steps:
        #     steps['action'] = steps['action'].reshape(self.num_steps, -1)
        
        steps = self._load_slice_with_levels(ep_idx, start)
        
        return steps

    def load_chunk(
        self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> list[dict]:
        chunk = []
        for ep, s, e in zip(episodes_idx, start, end):
            steps = self._load_slice(ep, s, e, frameskip=self.level1['frameskip'])
            if 'action' in steps:
                steps['action'] = steps['action'].reshape(
                    (e - s) // self.level1['frameskip'], -1
                )
            chunk.append(steps)
        return chunk

    def load_episode(self, episode_idx: int) -> dict:
        """Load full episode by index."""
        return self._load_slice(episode_idx, 0, self.lengths[episode_idx])

    def get_col_data(self, col: str) -> np.ndarray:
        raise NotImplementedError

    def get_dim(self, col: str) -> int:
        raise NotImplementedError

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        raise NotImplementedError

    def merge_col(
        self,
        source: list[str | dict[str, Any]] | str | dict[str, Any],
        target: str,
        dim: int = -1,
    ) -> None:
        raise NotImplementedError


class HDF5Dataset(Dataset):
    """Dataset loading from HDF5 file.

    Reads data from a single .h5 file containing all episode data.
    Uses SWMR mode for robust reading while writing.

    Args:
        name: Name of the dataset (filename without extension).
        frameskip: Number of frames to skip between samples.
        num_steps: Number of steps per sample sequence.
        transform: Optional data transform callable.
        keys_to_load: Specific keys to load (defaults to all except metadata).
        keys_to_cache: Keys to load entirely into memory for faster access.
        cache_dir: Directory containing the dataset file.
    """

    def __init__(
        self,
        name: str,
        # frameskip: int = 1,
        # num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        keys_to_merge: dict[str, list[str] | str] | None = None,
        cache_dir: str | Path | None = None,
        level1: dict | None = None,
        level2: dict | None = None,
        precompute_levels: bool = True,
        **level_kwargs,
    ) -> None:
        self.h5_path = Path(cache_dir or get_cache_dir(), f'{name}.h5')
        self.h5_file: h5py.File | None = None
        self._cache: dict[str, np.ndarray] = {}

        with h5py.File(self.h5_path, 'r') as f:
            lengths, offsets = f['ep_len'][:], f['ep_offset'][:]
            self._keys = keys_to_load or [
                k for k in f.keys() if k not in ('ep_len', 'ep_offset')
            ]

            for key in keys_to_cache or []:
                self._cache[key] = f[key][:]
                logging.info(f"Cached '{key}' from '{self.h5_path}'")

        super().__init__(
            lengths=lengths,
            offsets=offsets,
            level1=level1,
            level2=level2,
            precompute_levels=precompute_levels,
            transform=transform,
            **level_kwargs,
        )

        if keys_to_merge:
            for target, source in keys_to_merge.items():
                self.merge_col(source, target)

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def _open(self) -> None:
        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.h5_path, 'r', swmr=True, rdcc_nbytes=256 * 1024 * 1024
            )

    def _build_level2_action_chunks(
        self,
        level1_action_steps: torch.Tensor,
        action_dim: int,
    ) -> torch.Tensor:
        chunk_len = int(self.level2['frameskip'])
        num_chunks = int(self.level2['num_steps'])
        chunks = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_len
            end = start + chunk_len
            chunk = level1_action_steps[start:end]

            if chunk.shape[0] < chunk_len:
                pad = torch.zeros(
                    (chunk_len - chunk.shape[0], self.level1['frameskip'], action_dim),
                    dtype=chunk.dtype,
                    device=chunk.device,
                )
                chunk = torch.cat((chunk, pad), dim=0)

            chunks.append(chunk)

        return torch.stack(chunks)

    def _load_slice_with_levels(self, ep_dix: int, start: int) -> dict:
        # if self.levels == 1:
        #     return self._load_slice(ep_dix, start, start + self.level1['span'])
        # else:
        #     steps = self._load_slice(ep_dix, start, start + self.level2['span'])
        #     steps['level1'] = self._load_slice(
        #         ep_dix, start, start + self.level1['span']
        #     )
        #     return steps
        
        # load the largest span with level 1 granularity, then subsample for level 2
        
        output = {}
        
        total_level1_steps = self._load_slice(
            ep_idx=ep_dix, 
            start=start, 
            end=start + self.last_level['span'],
            frameskip=self.level1['frameskip'],
            apply_transform=False,
        )

        assert total_level1_steps['action'].shape[0] % self.level1['frameskip'] == 0
        total_level1_frames = total_level1_steps['action'].shape[0] // self.level1['frameskip']
        action_dim = total_level1_steps['action'].shape[-1]
        level1_action_steps = total_level1_steps['action'].reshape(
            total_level1_frames, self.level1['frameskip'], action_dim
        )
        total_level1_steps['action'] = level1_action_steps

        if not self.precompute_levels:
            # Return the full level-1 sequence covering the largest span and let
            # the model derive higher levels from it.
            level1_steps = dict(total_level1_steps)

            if self.transform:
                level1_steps = self.transform(level1_steps)

            if 'action' in level1_steps:
                level1_steps['action'] = level1_steps['action'].reshape(
                    level1_steps['action'].shape[0], -1
                )

            for col, steps in level1_steps.items():
                output[f'{col}_level1'] = steps

            return output

        # construct level 1 by sampling a chunk from the total_level1_steps
        if self.level1 is not None:
            total_level1_len = total_level1_frames
            start = torch.randint(0, total_level1_len - self.level1['num_steps'] + 1, (1,)).item()  # inclusive start
            level1_steps = {}
            for col in self._keys:
                if col == 'action':
                    chunk = level1_action_steps[start:start + self.level1['num_steps']]
                else:
                    x = total_level1_steps[col]
                    chunk = x[start:start + self.level1['num_steps']]
                level1_steps[col] = chunk

            if self.transform:
                level1_steps = self.transform(level1_steps)

            if 'action' in level1_steps:
                level1_steps['action'] = level1_steps['action'].reshape(
                    level1_steps['action'].shape[0], -1
                )

            for col, chunk in level1_steps.items():
                output[f'{col}_level1'] = chunk
            
        # At this point, pixels_level1 (4, 3, 224, 224). action_level1 (4, 10)
            
        # construct level 2 by subsampling from total_level1_steps
        if self.level2 is not None:
            level2_steps = {}
            for col in self._keys:
                if col == 'action':
                    # Keep primitive action dim through transform, then flatten back.
                    chunk = self._build_level2_action_chunks(
                        level1_action_steps,
                        action_dim,
                    )
                else:
                    x = total_level1_steps[col]
                    # keep every frameskip-th level-1 frame
                    chunk = x[:: self.level2['frameskip']][:self.level2['num_steps']]
                level2_steps[col] = chunk

            if self.transform:
                level2_steps = self.transform(level2_steps)

            if 'action' in level2_steps:
                a = level2_steps['action']
                level2_steps['action'] = a.reshape(a.shape[0], a.shape[1], -1)

            for col, chunk in level2_steps.items():
                output[f'{col}_level2'] = chunk
                                                
        return output
        

    def _load_slice(
        self,
        ep_idx: int,
        start: int,
        end: int,
        frameskip: int,
        apply_transform: bool = True,
    ) -> dict:
        self._open()
        g_start, g_end = (
            self.offsets[ep_idx] + start,
            self.offsets[ep_idx] + end,
        )
        steps = {}
        for col in self._keys:
            src = self._cache if col in self._cache else self.h5_file
            data = src[col][g_start:g_end]
            if col != 'action':
                data = data[:: frameskip]

            if data.dtype == np.object_ or data.dtype.kind in ('S', 'U'):
                val = data[0] if len(data) > 0 else b''
                steps[col] = val.decode() if isinstance(val, bytes) else val
            else:
                steps[col] = torch.from_numpy(data)
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    steps[col] = steps[col].permute(0, 3, 1, 2)

        if apply_transform and self.transform:
            return self.transform(steps)
        return steps

    def _get_col(self, col: str) -> np.ndarray:
        if col in self._cache:
            return self._cache[col]
        self._open()
        return self.h5_file[col][:]

    def get_col_data(self, col: str) -> np.ndarray:
        return self._get_col(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        self._open()
        data = {}
        for col in self._keys:
            src = self._cache if col in self._cache else self.h5_file
            data[col] = src[col][row_idx]
        return data

    def _load_merge_sources(
        self,
        source: list[str | dict[str, Any]] | str | dict[str, Any],
        dim: int,
    ) -> list[tuple[str, np.ndarray]]:
        self._open()
        if OmegaConf.is_config(source):
            source = OmegaConf.to_container(source, resolve=True)

        if isinstance(source, str):
            matches = [k for k in self.h5_file.keys() if re.match(source, k)]
            if not matches:
                raise KeyError(f"No columns matched merge source {source!r}")
            return [(key, self._get_col(key)) for key in matches]

        if isinstance(source, dict):
            key = source.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError("Merge source dicts must define a non-empty 'key'.")

            data = self._get_col(key)
            start = source.get("start", None)
            end = source.get("end", None)
            if start is not None or end is not None:
                axis = dim if dim >= 0 else data.ndim + dim
                if axis < 0 or axis >= data.ndim:
                    raise ValueError(
                        f"Cannot slice merge source {key!r} along dim={dim}; "
                        f"source has {data.ndim} dimensions."
                    )
                slices = [slice(None)] * data.ndim
                slices[axis] = slice(
                    None if start is None else int(start),
                    None if end is None else int(end),
                )
                data = data[tuple(slices)]

            label = key
            if start is not None or end is not None:
                start_label = "" if start is None else str(start)
                end_label = "" if end is None else str(end)
                label = f"{key}[{start_label}:{end_label}]"
            return [(label, data)]

        sources = []
        for item in source:
            sources.extend(self._load_merge_sources(item, dim))
        return sources

    def merge_col(
        self,
        source: list[str | dict[str, Any]] | str | dict[str, Any],
        target: str,
        dim: int = -1,
    ) -> None:
        sources = self._load_merge_sources(source, dim)

        merged = np.concatenate([data for _, data in sources], axis=dim)
        self._cache[target] = merged
        if target not in self._keys:
            self._keys.append(target)
        labels = [label for label, _ in sources]
        logging.info(f"Merged columns {labels} into '{target}' and cached it")

    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return np.prod(data.shape[1:]).item() if data.ndim > 1 else 1
