from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import stable_worldmodel as swm
import torch


NORMALIZER_ARTIFACT_FILENAME = "normalizer.pt"
IMAGENET_STATS = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class ColumnTransform:
    """Apply `fn` to `sample[source]` and store the result in `sample[target]`."""

    def __init__(self, fn, source: str, target: str):
        self.fn, self.source, self.target = fn, source, target

    def __call__(self, sample: dict) -> dict:
        sample[self.target] = self.fn(sample[self.source])
        return sample


class Compose:
    """Apply dict-sample transforms in sequence."""

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_hdf5_dataset(dataset_cfg, cache_dir=None):
    if OmegaConf.is_config(dataset_cfg):
        cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    else:
        cfg = dict(dataset_cfg)
    cfg.pop("val_name", None)
    if cfg.pop("type", None) == "droid":
        from droid_data import DROIDDataset

        return DROIDDataset(**cfg)
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

    normalizer = ColumnTransform(norm_fn, source=source, target=target)

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

    return ColumnTransform(norm_fn, source=source, target=target)


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
        count = int(dataset.lengths.sum())
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
