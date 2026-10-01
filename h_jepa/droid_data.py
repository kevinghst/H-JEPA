import json
from math import ceil
import os
from pathlib import Path

import decord
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from stable_worldmodel.data.dataset import HDF5Dataset
from stable_worldmodel.data.utils import get_cache_dir

decord.bridge.set_bridge("native")

ASSETS_DIR = Path(__file__).resolve().parent / "droid_assets"


def load_droid_norm_stats(key: str) -> dict[str, torch.Tensor]:
    with open(ASSETS_DIR / "norm_stats_droid.json") as f:
        entry = json.load(f)[key]
    return {
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in entry.items()
        if k != "provenance"
    }


def poses_to_diffs(poses: np.ndarray) -> np.ndarray:
    matrices = [Rotation.from_euler("xyz", theta).as_matrix() for theta in poses[:, 3:6]]
    xyz_diff = poses[1:, :3] - poses[:-1, :3]
    angle_diff = np.stack(
        [
            Rotation.from_matrix(matrices[t + 1] @ matrices[t].T).as_euler("xyz")
            for t in range(len(matrices) - 1)
        ]
    )
    gripper_diff = poses[1:, -1:] - poses[:-1, -1:]
    return np.concatenate([xyz_diff, angle_diff, gripper_diff], axis=1)


def _read_metadata(path: str) -> dict:
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(path, filename)) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"No metadata for video {path=}")


class DROIDClipReader(torch.utils.data.Dataset):
    """One num_frames clip per DROID episode, decoded from mp4 at `fps`.

    Returns (obs, actions, states, reward, env_info): obs["visual"] uint8 [T, C, H, W] at the
    video's resolution, actions [T, 7] pose deltas (last row zero), states [T, 7]
    cartesian_position ++ gripper_position. `frozen_clips` pins view and frame indices per clip
    (plan-eval manifest); otherwise the view and the window are drawn at random per access.
    Relative paths (the CSV, its episode entries, the manifest's episode_path) resolve against
    $STABLEWM_HOME/droid.
    """

    def __init__(
        self,
        data_path: str | None,
        camera_views: list[str],
        num_frames: int,
        fps: int,
        frozen_clips: str | None = None,
        deterministic_getitem: bool = False,
    ):
        self.camera_views = list(camera_views)
        self.num_frames = int(num_frames)
        self.fps = fps
        self.deterministic_getitem = deterministic_getitem
        self.frozen = None
        root = get_cache_dir() / "droid"
        if frozen_clips is not None:
            with open(frozen_clips) as f:
                self.frozen = json.load(f)
            self.samples = [str(root / row["episode_path"]) for row in self.frozen]
        else:
            with open(root / data_path) as f:
                self.samples = [str(root / line.split()[0]) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def _rng(self, idx: int) -> np.random.Generator:
        if self.deterministic_getitem or self.frozen is not None:
            return np.random.default_rng(np.random.SeedSequence([0, int(idx)]))
        return np.random.default_rng()

    def _load_states(self, path: str, camera_name: str):
        # The extrinsics are unused, but read (and indexed in _load_clip) so that an episode
        # with missing or short extrinsics fails and is replaced by the retry loop, as in the
        # loader the paper runs used.
        npz_path = os.path.join(path, "trajectory.npz")
        if os.path.exists(npz_path):
            with np.load(npz_path) as tr:
                extrinsics = tr[f"camera_extrinsics/{camera_name}_left"]
                cart = tr["cartesian_position"]
                grip = tr["gripper_position"]
        else:
            with h5py.File(os.path.join(path, "trajectory.h5"), "r") as tr:
                obs = tr["observation"]
                extrinsics = np.asarray(obs["camera_extrinsics"][f"{camera_name}_left"])
                cart = np.asarray(obs["robot_state"]["cartesian_position"])
                grip = np.asarray(obs["robot_state"]["gripper_position"])
        return extrinsics, np.concatenate([cart, grip[:, None]], axis=1)

    def _load_clip(self, path: str, rng: np.random.Generator, frozen: dict | None):
        metadata = _read_metadata(path)
        # The view is drawn before the window: with 2 views this draw shifts the rng stream,
        # so the sampled windows depend on it.
        if frozen is not None:
            view_idx = int(frozen["view_idx"])
        else:
            view_idx = int(rng.integers(0, len(self.camera_views)))
        mp4_name = metadata[self.camera_views[view_idx]].split("recordings/MP4/")[-1]
        extrinsics, states = self._load_states(path, mp4_name.split(".")[0])

        vpath = os.path.join(path, "recordings/MP4", mp4_name)
        if not os.path.exists(vpath):
            raise FileNotFoundError(f"Video file missing: {vpath}")
        vr = decord.VideoReader(vpath, ctx=decord.cpu(0), num_threads=1)
        vlen = len(vr)
        # DROID mp4s are tagged at 4x their 15 Hz recording rate (60 fps).
        fstp = ceil(vr.get_avg_fps() / (4 * self.fps))
        nframes = self.num_frames * fstp

        if frozen is not None:
            indices = np.asarray(frozen["indices"], dtype=np.int64)
        else:
            if vlen < nframes:
                raise RuntimeError(f"Video is too short {path=}, {nframes=}, {vlen=}")
            end = int(rng.integers(nframes, vlen))
            indices = np.arange(end - nframes, end, fstp).astype(np.int64)

        states = states[indices]
        extrinsics[indices]
        vr.seek(0)
        frames = torch.from_numpy(vr.get_batch(indices).asnumpy())
        frames = frames.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
        return frames, poses_to_diffs(states), states

    def __getitem__(self, idx: int):
        rng = self._rng(idx)
        frozen = self.frozen[idx] if self.frozen is not None else None
        max_retries = 1 if frozen is not None else 10
        orig_idx = idx
        for attempt in range(max_retries):
            try:
                visual, actions, states = self._load_clip(self.samples[idx], rng, frozen)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to load video after {max_retries} attempts. "
                        f"Last path: {self.samples[idx]}, error: {e}"
                    ) from e
                retry_rng = np.random.default_rng(
                    np.random.SeedSequence([0, int(orig_idx), attempt])
                )
                idx = int(retry_rng.integers(0, len(self)))
                rng = self._rng(idx)

        actions = np.concatenate([actions, np.zeros((1, actions.shape[-1]))], axis=0)
        actions = torch.tensor(actions, dtype=torch.float32)  # [T, 7]
        states = torch.tensor(states, dtype=torch.float32)  # [T, 7]
        env_info = (
            {"grasp_pos": int(frozen["grasp_pos"])}
            if frozen is not None and "grasp_pos" in frozen
            else None
        )
        return {"visual": visual, "proprio": states}, actions, states, torch.tensor(0.0), env_info


class DROIDDataset(HDF5Dataset):
    def __init__(
        self,
        name: str,
        camera_views: list[str],
        fps: int,
        norm_stats: str,
        keys_to_load: list[str],
        level1: dict,
        deterministic_getitem: bool = False,
        **level_kwargs: dict,
    ):
        self._setup_levels(level1, **level_kwargs)
        span = self.last_level["span"]
        self.reader = DROIDClipReader(
            name, camera_views, span, fps, deterministic_getitem=deterministic_getitem
        )
        self._keys = list(keys_to_load)
        self.lengths = np.full(len(self.reader), span)
        self.clip_indices = [(ep, 0) for ep in range(len(self.reader))]
        self.transform = None
        stats = load_droid_norm_stats(norm_stats)
        self.stats = {
            "action": (stats["action_mean"][None], stats["action_std"][None]),
            "proprio": (stats["state_mean"][None], stats["state_std"][None]),
        }

    def _load_slice(self, ep_idx, start, end, frameskip, apply_transform=True) -> dict:
        obs, actions, states, _, _ = self.reader[int(ep_idx)]
        steps = {"pixels": obs["visual"], "action": actions, "proprio": states}
        return {col: steps[col] for col in self._keys}

    def get_col_stats(self, col: str) -> tuple[np.ndarray, np.ndarray]:
        mean, std = self.stats[col]
        return mean.numpy(), std.numpy()

    def get_dim(self, col: str) -> int:
        return 7
