"""Dataset classes for episode-based reinforcement learning data."""

import logging
from pathlib import Path
import re
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from omegaconf import OmegaConf

from stable_worldmodel.data.utils import get_cache_dir


class HDF5Dataset:
    """Episode dataset loaded from a single HDF5 file.

    Args:
        name: Name of the dataset (filename without extension).
        keys_to_load: Specific keys to load (defaults to all except metadata).
        keys_to_cache: Keys to load entirely into memory for faster access.
        keys_to_merge: Target column -> source columns to concatenate and cache.
        cache_dir: Directory containing the dataset file.
        level1, level2, ...: Per-level clip configs (frameskip, num_steps, window_size).
    """

    def __init__(
        self,
        name: str,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        keys_to_merge: dict[str, list[str] | str] | None = None,
        cache_dir: str | Path | None = None,
        level1: dict | None = None,
        **level_kwargs,
    ) -> None:
        self.h5_path = Path(cache_dir or get_cache_dir(), f'{name}.h5')
        self.h5_file: h5py.File | None = None
        self._cache: dict[str, np.ndarray] = {}

        with h5py.File(self.h5_path, 'r') as f:
            self.lengths, self.offsets = f['ep_len'][:], f['ep_offset'][:]
            self._keys = keys_to_load or [
                k for k in f.keys() if k not in ('ep_len', 'ep_offset')
            ]

            for key in keys_to_cache or []:
                self._cache[key] = f[key][:]
                logging.info(f"Cached '{key}' from '{self.h5_path}'")

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
            normalized["window_size"] = int(normalized.get("window_size", 1))
            return normalized

        level_inputs = {"level1": level1}
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
            for ep, length in enumerate(self.lengths)
            if length >= self.last_level['span']
            for start in range(length - self.last_level['span'] + 1)
        ]

        self.transform = None

        if keys_to_merge:
            for target, source in keys_to_merge.items():
                self.merge_col(source, target)

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.clip_indices[idx]
        return self._load_slice_with_levels(ep_idx, start)

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

    def _open(self) -> None:
        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.h5_path, 'r', swmr=True, rdcc_nbytes=256 * 1024 * 1024
            )

    def _load_slice_with_levels(self, ep_dix: int, start: int) -> dict:
        # Return the full level-1 sequence covering the largest span; the model
        # derives the higher levels from it.
        level1_steps = self._load_slice(
            ep_idx=ep_dix,
            start=start,
            end=start + self.last_level['span'],
            frameskip=self.level1['frameskip'],
            apply_transform=False,
        )

        assert level1_steps['action'].shape[0] % self.level1['frameskip'] == 0
        level1_frames = level1_steps['action'].shape[0] // self.level1['frameskip']
        action_dim = level1_steps['action'].shape[-1]
        level1_steps['action'] = level1_steps['action'].reshape(
            level1_frames, self.level1['frameskip'], action_dim
        )

        if self.transform:
            level1_steps = self.transform(level1_steps)

        if 'action' in level1_steps:
            level1_steps['action'] = level1_steps['action'].reshape(
                level1_steps['action'].shape[0], -1
            )

        return {f'{col}_level1': steps for col, steps in level1_steps.items()}

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
