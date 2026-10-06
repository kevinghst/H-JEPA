"""Compute and store per-frame ego-to-closest-distractor distance.

The distance respects the four-room layout: same room uses the straight-line
Euclidean distance, otherwise it is the shortest path summing Euclidean
segments through door centers (one door for adjacent rooms, two for diagonal
rooms), excluding the episode's closed door. This reuses the env's own
`shortest_path`. Frames with no active distractor are stored as NaN so the
probing pipeline masks them out automatically.
"""

import inspect

import h5py
import numpy as np
from loguru import logger as logging
from tqdm import tqdm

from stable_worldmodel.envs.four_room.env import DOOR_ORDER, FourRoomDistractorsEnv


def _build_env(world_config):
    cfg = dict(world_config)
    cfg.pop('env_name', None)
    params = inspect.signature(FourRoomDistractorsEnv.__init__).parameters
    kwargs = {key: value for key, value in cfg.items() if key in params}
    return FourRoomDistractorsEnv(**kwargs)


def compute_min_distractor_distances(h5_path, world_config):
    """Return an (N, 1) float32 array of min distances, NaN where none active."""
    with h5py.File(h5_path, 'r') as f:
        if 'distractor_xy' not in f:
            return None
        ego = f['xy'][:].astype(np.float64)
        distractors = f['distractor_xy'][:].astype(np.float64)
        closed = f['closed_door_idx'][:].astype(np.int64)

    env = _build_env(world_config)
    n = ego.shape[0]
    out = np.full((n, 1), np.nan, dtype=np.float32)

    prev_idx = None
    for i in tqdm(range(n), desc='min_distractor_dist'):
        idx = int(closed[i])
        if idx != prev_idx:
            env.closed_door = None if idx == -1 else DOOR_ORDER[idx]
            prev_idx = idx

        ego_i = ego[i]
        best = np.inf
        for distractor in distractors[i]:
            if not np.isfinite(distractor).all():
                continue
            dist = env.shortest_path(ego_i, distractor)[0]
            if not np.isfinite(dist):  # rooms unreachable (shouldn't happen)
                dist = float(np.linalg.norm(ego_i - distractor))
            best = min(best, dist)

        if np.isfinite(best):
            out[i, 0] = best

    return out


def write_min_distractor_distances(h5_path, dists):
    with h5py.File(h5_path, 'a') as f:
        if 'min_distractor_dist' in f:
            del f['min_distractor_dist']
        f.create_dataset(
            'min_distractor_dist',
            data=dists,
            dtype=np.float32,
            maxshape=(None, 1),
            chunks=(min(len(dists), 4096), 1),
        )


def backfill_h5(h5_path, world_config):
    dists = compute_min_distractor_distances(h5_path, world_config)
    if dists is None:
        logging.info(f'{h5_path}: no distractor_xy column; skipping.')
        return False
    write_min_distractor_distances(h5_path, dists)
    n_valid = int(np.isfinite(dists).sum())
    logging.success(
        f'{h5_path}: wrote min_distractor_dist for {n_valid}/{len(dists)} frames.'
    )
    return True
