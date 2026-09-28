from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import stable_worldmodel as swm
from stable_pretraining import data as dt
import torch


NORMALIZER_ARTIFACT_FILENAME = "normalizer.pt"


class MixedHDF5Dataset:
    """Virtual HDF5 dataset made by concatenating selected source episodes."""

    def __init__(
        self,
        datasets: list,
        selected_episodes: list[np.ndarray],
    ):
        if not datasets:
            raise ValueError("MixedHDF5Dataset requires at least one source dataset.")
        if len(datasets) != len(selected_episodes):
            raise ValueError("datasets and selected_episodes must have the same length.")

        self.datasets = list(datasets)
        self.selected_episodes = [
            np.asarray(episodes, dtype=np.int64) for episodes in selected_episodes
        ]
        self.cumulative_sizes = np.cumsum([len(dataset) for dataset in self.datasets])
        self._transform = None
        self.lengths = np.concatenate(
            [
                dataset.lengths[episodes]
                for dataset, episodes in zip(self.datasets, self.selected_episodes)
            ]
        )
        self.offsets = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.cumsum(self.lengths[:-1], dtype=np.int64),
            )
        )

        first_columns = list(self.datasets[0].column_names)
        for dataset in self.datasets[1:]:
            if list(dataset.column_names) != first_columns:
                raise ValueError(
                    "All mixed HDF5 sources must expose the same columns. "
                    f"Expected {first_columns}, got {list(dataset.column_names)}."
                )

    @property
    def column_names(self) -> list[str]:
        return list(self.datasets[0].column_names)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform) -> None:
        self._transform = transform
        for dataset in self.datasets:
            dataset.transform = transform

    def __len__(self) -> int:
        return int(self.cumulative_sizes[-1])

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        source_idx = int(np.searchsorted(self.cumulative_sizes, idx, side="right"))
        previous_size = (
            0 if source_idx == 0 else int(self.cumulative_sizes[source_idx - 1])
        )
        return self.datasets[source_idx][idx - previous_size]

    def get_col_data(self, col: str) -> np.ndarray:
        chunks = []
        for dataset, episodes in zip(self.datasets, self.selected_episodes):
            data = dataset.get_col_data(col)
            chunks.extend(
                data[
                    dataset.offsets[episode] :
                    dataset.offsets[episode] + dataset.lengths[episode]
                ]
                for episode in episodes
            )
        if not chunks:
            raise ValueError(f"Column {col!r} has no selected rows.")
        return np.concatenate(
            chunks,
            axis=0,
        )

    def get_col_stats(self, col: str) -> tuple[np.ndarray, np.ndarray]:
        total_count = 0
        total_sum = None
        total_sumsq = None

        for dataset, episodes in zip(self.datasets, self.selected_episodes):
            source_data = np.asarray(dataset.get_col_data(col))
            for episode in episodes:
                start = dataset.offsets[episode]
                end = start + dataset.lengths[episode]
                data = source_data[start:end]
                if data.ndim == 0:
                    raise ValueError(
                        f"Column {col!r} is scalar, expected row-aligned data."
                    )

                flat = data.reshape(data.shape[0], -1)
                valid_mask = ~np.isnan(flat).any(axis=1)
                if not np.any(valid_mask):
                    continue

                valid = data[valid_mask].astype(np.float64, copy=False)
                count = valid.shape[0]
                source_sum = valid.sum(axis=0, keepdims=True)
                source_sumsq = np.square(valid).sum(axis=0, keepdims=True)

                total_count += count
                if total_sum is None:
                    total_sum = source_sum
                    total_sumsq = source_sumsq
                else:
                    total_sum = total_sum + source_sum
                    total_sumsq = total_sumsq + source_sumsq

        if total_count == 0 or total_sum is None or total_sumsq is None:
            raise ValueError(f"Column {col!r} has no finite rows in mixed dataset.")

        mean = total_sum / total_count
        if total_count > 1:
            variance = (total_sumsq - np.square(total_sum) / total_count) / (
                total_count - 1
            )
        else:
            variance = np.zeros_like(mean)
        std = np.sqrt(np.maximum(variance, 0.0))
        return mean, std

    def get_dim(self, col: str) -> int:
        dims = [dataset.get_dim(col) for dataset in self.datasets]
        if len(set(dims)) != 1:
            raise ValueError(f"Mixed sources disagree on dimension for {col!r}: {dims}")
        return dims[0]


def build_hdf5_dataset(dataset_cfg, cache_dir=None):
    if OmegaConf.is_config(dataset_cfg):
        cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    else:
        cfg = dict(dataset_cfg)
    cfg.pop("val_name", None)
    sources = cfg.pop("sources", None)

    if sources is None:
        cfg.pop("mix_mode", None)
        cfg.pop("subset_unit", None)
        subset_seed = int(cfg.pop("subset_seed", 0))
        total_transitions = cfg.pop("total_transitions", None)
        dataset = swm.data.HDF5Dataset(**cfg, cache_dir=cache_dir)
        if total_transitions is not None:
            episodes = _select_episodes_for_transition_budget(
                dataset, int(total_transitions), np.random.default_rng(subset_seed)
            )
            selected = set(episodes.tolist())
            dataset.clip_indices = [
                (episode, start)
                for episode, start in dataset.clip_indices
                if episode in selected
            ]
        return dataset

    mix_mode = cfg.pop("mix_mode", "subset")
    if mix_mode != "subset":
        raise ValueError(f"Unsupported mixed dataset mode: {mix_mode!r}")

    subset_unit = cfg.pop("subset_unit", "episode")
    if subset_unit != "episode":
        raise ValueError(f"Unsupported mixed dataset subset_unit: {subset_unit!r}")

    total_transitions = cfg.pop("total_transitions", None)
    if total_transitions is None:
        raise ValueError("Mixed subset datasets must define total_transitions.")
    total_transitions = int(total_transitions)
    if total_transitions <= 0:
        raise ValueError("total_transitions must be positive.")

    subset_seed = int(cfg.pop("subset_seed", 0))
    rng = np.random.default_rng(subset_seed)

    cfg.pop("name", None)
    source_datasets = []
    source_proportions = []

    for source in sources:
        if OmegaConf.is_config(source):
            source_cfg = OmegaConf.to_container(source, resolve=True)
        else:
            source_cfg = dict(source)
        if "proportion" not in source_cfg:
            raise ValueError("Each mixed dataset source must define a proportion.")
        proportion = float(source_cfg.pop("proportion"))
        if not np.isfinite(proportion) or proportion <= 0:
            raise ValueError("Each source proportion must be positive and finite.")
        name = source_cfg.get("name")
        if not name:
            raise ValueError("Each mixed dataset source must define a name.")

        child_cfg = dict(cfg)
        child_cfg.update(source_cfg)
        source_datasets.append(
            swm.data.HDF5Dataset(**child_cfg, cache_dir=cache_dir)
        )
        source_proportions.append(proportion)

    selected_episodes = _select_mixed_episode_subsets(
        source_datasets,
        source_proportions,
        total_transitions,
        rng,
    )
    for dataset, episodes in zip(source_datasets, selected_episodes):
        selected = set(episodes.tolist())
        dataset.clip_indices = [
            (episode, start)
            for episode, start in dataset.clip_indices
            if episode in selected
        ]

    return MixedHDF5Dataset(source_datasets, selected_episodes)


def _select_mixed_episode_subsets(
    datasets: list,
    proportions: list[float],
    total_transitions: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    proportions_array = np.asarray(proportions, dtype=np.float64)
    proportions_array = proportions_array / proportions_array.sum()
    targets = np.floor(proportions_array * total_transitions).astype(np.int64)
    targets[-1] += total_transitions - int(targets.sum())

    selected = []
    for dataset, target in zip(datasets, targets):
        selected.append(_select_episodes_for_transition_budget(dataset, int(target), rng))
    return selected


def _select_episodes_for_transition_budget(
    dataset,
    target_transitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    valid_episodes = np.flatnonzero(dataset.lengths >= dataset.last_level["span"])
    if valid_episodes.size == 0:
        raise ValueError(f"No valid episodes in {dataset.h5_path}.")

    available_transitions = int(dataset.lengths[valid_episodes].sum())
    if target_transitions > available_transitions:
        raise ValueError(
            f"Requested {target_transitions} transitions from {dataset.h5_path}, "
            f"but only {available_transitions} valid episode transitions are available."
        )

    shuffled = rng.permutation(valid_episodes)
    chosen = []
    total = 0

    for episode in shuffled:
        length = int(dataset.lengths[episode])
        previous_total = total
        chosen.append(int(episode))
        total += length

        if total >= target_transitions:
            if (
                len(chosen) > 1
                and abs(target_transitions - previous_total)
                < abs(target_transitions - total)
            ):
                chosen.pop()
            break

    if not chosen:
        chosen.append(int(shuffled[0]))

    return np.asarray(sorted(chosen), dtype=np.int64)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    if hasattr(dataset, "get_col_stats"):
        mean_np, std_np = dataset.get_col_stats(source)
        mean = torch.from_numpy(np.array(mean_np)).float()
        std = torch.from_numpy(np.array(std_np)).float()
    else:
        col_data = dataset.get_col_data(source)
        data = torch.from_numpy(np.array(col_data))
        data = data[~torch.isnan(data).any(dim=1)]
        mean = data.mean(0, keepdim=True).clone()
        std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        x = (x - mean) / std
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)

    return normalizer


def get_column_normalizer_from_artifact(
    artifact: dict, source: str, target: str, fill_nan: bool = True
):
    """Get a column normalizer from saved training-set stats.

    With fill_nan=False, NaN rows are preserved so callers can mask them (used
    for probe targets whose padded/inactive slots are stored as NaN).
    """
    stats = artifact.get("stats", {})
    if source not in stats:
        available = ", ".join(sorted(stats.keys()))
        raise KeyError(
            f"Normalizer artifact is missing column {source!r}. "
            f"Available columns: {available}"
        )

    col_stats = stats[source]
    mean = torch.from_numpy(np.asarray(col_stats["mean"])).float()
    std = torch.from_numpy(np.asarray(col_stats["std"])).float()

    def norm_fn(x):
        x = (x - mean) / std
        if fill_nan:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.float()

    return dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)


def normalizer_columns_from_dataset_cfg(dataset_cfg) -> list[str]:
    """Return non-pixel dataset columns that need z-score normalization."""
    columns = []
    for key in ("keys_to_load", "keys_to_cache"):
        for col in dataset_cfg.get(key, []) or []:
            if col.startswith("pixels"):
                continue
            if col not in columns:
                columns.append(col)

    keys_to_merge = dataset_cfg.get("keys_to_merge", None)
    if keys_to_merge is not None:
        for col in keys_to_merge.keys():
            if col.startswith("pixels"):
                continue
            if col not in columns:
                columns.append(col)

    return columns


def _column_mean_std(dataset, col: str) -> tuple[np.ndarray, np.ndarray, int]:
    if hasattr(dataset, "get_col_stats"):
        mean, std = dataset.get_col_stats(col)
        count = int(dataset.get_col_data(col).shape[0])
        return (
            np.asarray(mean, dtype=np.float64),
            np.asarray(std, dtype=np.float64),
            count,
        )

    data = np.asarray(dataset.get_col_data(col))
    flat = data.reshape(data.shape[0], -1)
    valid_mask = ~np.isnan(flat).any(axis=1)
    valid = data[valid_mask].astype(np.float64, copy=False)
    count = int(valid.shape[0])
    if count == 0:
        raise ValueError(f"Column {col!r} has no finite rows.")

    mean = valid.mean(axis=0, keepdims=True)
    std = valid.std(axis=0, ddof=1 if count > 1 else 0, keepdims=True)
    return mean, std, count


def build_normalizer_artifact(cfg, train_dataset) -> dict:
    """Build serializable training normalizer stats for planning eval."""
    dataset_cfg = cfg.data.dataset
    stats = {}
    for col in normalizer_columns_from_dataset_cfg(dataset_cfg):
        mean, std, count = _column_mean_std(train_dataset, col)
        stats[col] = {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "count": count,
        }

    return {
        "format": "lejepa_training_normalizer_v1",
        "stats": stats,
        "metadata": {
            "dataset_name": str(dataset_cfg.get("name", "")),
            "columns": list(stats.keys()),
            "std_ddof": 1,
        },
    }


def save_normalizer_artifact(artifact: dict, run_dir: str | Path) -> Path:
    path = Path(run_dir) / NORMALIZER_ARTIFACT_FILENAME
    torch.save(artifact, path)
    return path


def load_normalizer_artifact(path: str | Path) -> dict:
    artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict):
        raise TypeError(f"Normalizer artifact must be a dict, got {type(artifact)}.")
    if artifact.get("format") != "lejepa_training_normalizer_v1":
        raise ValueError(
            f"Unsupported normalizer artifact format: {artifact.get('format')!r}."
        )
    if not isinstance(artifact.get("stats"), dict):
        raise ValueError("Normalizer artifact is missing a 'stats' dictionary.")
    return artifact
