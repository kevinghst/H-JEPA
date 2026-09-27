#!/usr/bin/env python
"""Generate FourRoomDistractors eval tasks that share ego + distractor behavior
across a sweep of active distractor counts.

Non-canonical protocol. A single rollout is generated with the full distractor
capacity active. Then one eval-traj file is emitted per active-count k in
[0, max_distractors]: the ego trajectory and the trajectories of distractors
0..k-1 are byte-identical across every file, and the only differences are that
distractor slots >= k are NaN'd and the pixels (including the goal frame) are
re-rendered showing only the first k distractors. This is the "erase one
distractor" construction: file k is file k+1 with its last distractor removed.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_REPO_ROOT = REPO_ROOT.parent
for path in (REPO_ROOT, PARENT_REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from planning_eval import generation_provenance
from stable_worldmodel.envs.four_room import ExpertPolicy
from stable_worldmodel.envs.four_room.env import ROOMS


SKIP_COLUMNS = {"env_name", "id", "render_time"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a sweep of FourRoomDistractors eval-traj files that share "
            "ego and distractor behavior, varying only the active distractor "
            "count."
        )
    )
    parser.add_argument(
        "--data-config-path",
        required=True,
        help="Four-room collection sidecar or data config (world block).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the per-count .pt files.",
    )
    parser.add_argument(
        "--output-stem",
        required=True,
        help=(
            "Filename stem. Each file is written as "
            "<stem>_cross<N>_goal<G>_n<E>_d<k>.pt."
        ),
    )
    parser.add_argument(
        "--cross-n-rooms",
        dest="cross_n_rooms",
        type=int,
        default=1,
        help="Shortest-path room crossings between start and goal (0-3).",
    )
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=_optional_positive_int, default=30)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    return parser.parse_args()


def _load_data_config(path: str | Path):
    config_path = Path(path).expanduser()
    if not config_path.exists() and not config_path.is_absolute():
        for root in (PARENT_REPO_ROOT, REPO_ROOT):
            candidate = root / config_path
            if candidate.exists():
                config_path = candidate
                break
    if not config_path.exists():
        raise FileNotFoundError(f"Data config not found: {config_path}")
    return OmegaConf.load(config_path)


def _base_options(cfg) -> dict[str, Any]:
    options = cfg.get("options", None)
    if options is None:
        return {}
    if OmegaConf.is_config(options):
        options = OmegaConf.to_container(options, resolve=True)
    if not isinstance(options, dict):
        raise TypeError("data config options must be a mapping when provided.")
    return deepcopy(options)


def _build_world(cfg):
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    env_name = world_cfg.pop("env_name", "swm/FourRoomDistractors-v0")
    world_cfg["num_envs"] = 1
    return swm.World(env_name, **world_cfg, render_mode="rgb_array", verbose=0)


def _set_deterministic_expert(world, seed: int) -> None:
    policy = ExpertPolicy(
        action_noise=0.0,
        action_repeat_prob=0.0,
        seed=seed,
    )
    world.set_policy(policy)


def _unwrapped_env(world):
    return world.envs.envs[0].unwrapped


def _reset_task(
    world,
    seed: int,
    options: dict[str, Any],
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
) -> None:
    task_options = deepcopy(options)
    task_options["state"] = np.asarray(start_xy, dtype=np.float32)
    task_options["target_state"] = np.asarray(goal_xy, dtype=np.float32)
    world.reset(seed=seed, options=task_options)
    world.terminateds = np.zeros(world.num_envs, dtype=bool)
    world.truncateds = np.zeros(world.num_envs, dtype=bool)


def _extract_env_value(infos: dict[str, Any], key: str, env_idx: int = 0):
    value = infos[key]
    if isinstance(value, np.ndarray):
        if value.ndim > 1 and value.shape[1] == 1:
            value = np.squeeze(value, axis=1)
        item = value[env_idx]
        if isinstance(item, np.ndarray):
            return item.copy()
        return np.asarray(item).copy()
    return value


def _dump_current_info(infos: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    step = {}
    for key in columns:
        if key not in infos:
            continue
        value = _extract_env_value(infos, key)
        if isinstance(value, np.ndarray):
            step[key] = value.copy()
        else:
            step[key] = value
    return step


def _saved_columns(keys: Iterable[str]) -> list[str]:
    return [
        key
        for key in keys
        if not key.startswith("_") and key not in SKIP_COLUMNS
    ]


def _shift_action_aligned_columns(
    steps: list[dict[str, Any]],
    columns: Iterable[str],
) -> None:
    for key in columns:
        if key not in steps[0]:
            continue

        values = [step[key] for step in steps]
        replacement = np.full_like(np.asarray(values[-1]), np.nan, dtype=np.float32)
        shifted = values[1:] + [replacement]
        for step, value in zip(steps, shifted, strict=True):
            step[key] = value


def _step_list_to_episode(
    steps: list[dict[str, Any]],
    columns: list[str],
) -> dict[str, Any]:
    episode = {}
    for key in columns:
        if key not in steps[0]:
            continue

        values = [step[key] for step in steps]
        if key == "action" and len(values) > 1:
            last_value = np.asarray(values[-1])
            if np.issubdtype(last_value.dtype, np.floating) and np.isnan(last_value).all():
                values = values[:-1]

        first = values[0]
        if isinstance(first, str):
            episode[key] = first
            continue

        array = np.asarray(values)
        if array.dtype.kind in ("U", "S", "O"):
            episode[key] = values[0]
            continue

        tensor = torch.from_numpy(array)
        if key in {"pixels", "goal"} and tensor.ndim == 4 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
        episode[key] = tensor

    return episode


def _rollout_full_task(
    world,
    seed: int,
    options: dict[str, Any],
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    rollout_budget: int,
) -> list[dict[str, Any]] | None:
    _reset_task(world, seed, options, start_xy, goal_xy)
    columns = _saved_columns(world.infos.keys())
    steps = [_dump_current_info(world.infos, columns)]

    for _ in range(int(rollout_budget)):
        if bool(world.terminateds[0]) or bool(world.truncateds[0]):
            break
        world.step()
        steps.append(_dump_current_info(world.infos, columns))
        if bool(world.terminateds[0]):
            return steps

    return None


def _make_episode_from_steps(
    steps: list[dict[str, Any]],
    columns: list[str],
    episode_idx: int,
) -> dict[str, Any]:
    segment_steps = [deepcopy(step) for step in steps]
    for step_idx, step in enumerate(segment_steps):
        step["step_idx"] = np.asarray(step_idx, dtype=np.int64)
        step["ep_idx"] = np.asarray(episode_idx, dtype=np.int64)
    if "ep_idx" not in columns:
        columns.append("ep_idx")

    _shift_action_aligned_columns(segment_steps, ("action",))
    return _step_list_to_episode(segment_steps, columns)


def _pixels_to_frames(pixels: torch.Tensor | np.ndarray) -> np.ndarray:
    frames = pixels.detach().cpu().numpy() if torch.is_tensor(pixels) else np.asarray(pixels)
    if frames.ndim != 4:
        raise ValueError(
            "Expected pixels with shape [T, C, H, W] or [T, H, W, C], "
            f"got {frames.shape}."
        )

    if frames.shape[1] in (1, 3, 4):
        frames = np.transpose(frames, (0, 2, 3, 1))
    elif frames.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unable to infer pixel channel placement from {frames.shape}.")

    if frames.dtype != np.uint8:
        if np.issubdtype(frames.dtype, np.floating):
            scale = 255.0 if np.nanmax(frames) <= 1.0 else 1.0
            frames = frames * scale
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    return frames


def _save_selected_videos(
    data: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    output_path: Path,
    fps: int,
) -> list[Path]:
    import imageio.v2 as imageio

    video_dir = output_path.with_suffix("")
    video_dir.mkdir(parents=True, exist_ok=True)

    written_paths = []
    for task_idx, (episode, task) in enumerate(zip(data, tasks, strict=True)):
        if "pixels" not in episode:
            raise KeyError(f"Episode {task_idx} is missing 'pixels'.")
        frames = _pixels_to_frames(episode["pixels"])
        video_path = (
            video_dir
            / (
                f"traj_{task_idx:04d}"
                f"_len_{int(task['segment_frame_count'])}"
                f"_actions_{int(task['segment_action_count'])}.mp4"
            )
        )
        with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
            for frame in frames:
                writer.append_data(frame)
        written_paths.append(video_path)

    return written_paths


def main() -> None:
    args = parse_args()
    cfg = _load_data_config(args.data_config_path)

    # Force every episode to use the full distractor capacity so that any lower
    # count is a strict prefix of the same distractor set.
    max_distractors = int(cfg.world.max_distractors)
    cfg.world.min_distractors = max_distractors

    if args.cross_n_rooms < 0 or args.cross_n_rooms > 3:
        raise ValueError("--cross-n-rooms must be in [0, 3].")
    close_door_prob = float(cfg.world.get("close_door_prob", 0.0))
    if args.cross_n_rooms == 3 and close_door_prob <= 0.0:
        raise ValueError("--cross-n-rooms=3 requires world.close_door_prob > 0.")

    num_episodes = int(args.num_episodes)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    max_attempts = int(args.max_attempts or 100 * num_episodes)
    rollout_budget = int(cfg.world.max_episode_steps)
    if args.max_steps is not None and args.max_steps > rollout_budget:
        raise ValueError(
            f"--max-steps cannot exceed world.max_episode_steps ({rollout_budget})."
        )

    rng = np.random.default_rng(seed)
    options = _base_options(cfg)
    world = _build_world(cfg)
    _set_deterministic_expert(world, seed)

    variant_data = {k: [] for k in range(max_distractors + 1)}
    tasks = []
    render_check_done = False

    try:
        progress = tqdm(total=num_episodes, desc="Generating")
        attempts = 0
        while len(tasks) < num_episodes and attempts < max_attempts:
            attempts += 1
            attempt_seed = int(rng.integers(0, 1_000_000_000))

            world.reset(seed=attempt_seed, options=options)
            env = _unwrapped_env(world)
            try:
                start_xy, goal_xy, start_room, goal_room = _sample_crossing_task(
                    env, rng, int(args.cross_n_rooms)
                )
            except NoRoomPairError:
                continue

            steps = _rollout_full_task(
                world, attempt_seed, options, start_xy, goal_xy, rollout_budget
            )
            if steps is None:
                continue

            columns = _saved_columns(steps[0].keys())
            original_action_count = len(steps) - 1
            if args.max_steps is None:
                segment_steps = steps
                source_window = (0, len(steps) - 1)
            elif original_action_count < args.max_steps:
                continue
            elif original_action_count == args.max_steps:
                segment_steps = steps
                source_window = (0, len(steps) - 1)
            else:
                windows = _valid_crossing_windows(
                    env, steps, int(args.max_steps), int(args.cross_n_rooms)
                )
                if not windows:
                    continue
                source_window = windows[int(rng.integers(0, len(windows)))]
                segment_steps = steps[source_window[0] : source_window[1] + 1]

            episode_idx = len(tasks)

            # Sanity check on the first accepted frame: re-rendering with the
            # full count must reproduce the live-captured pixels.
            if not render_check_done and "pixels" in segment_steps[0]:
                captured = np.asarray(segment_steps[0]["pixels"], dtype=np.uint8)
                if captured.ndim == 3 and captured.shape[0] in (1, 3, 4):
                    captured = captured.transpose(1, 2, 0)
                rerender = _render_variant_pixels(
                    env,
                    segment_steps[0]["xy"],
                    segment_steps[0]["distractor_xy"],
                    max_distractors,
                )
                max_diff = int(np.abs(captured.astype(np.int32) - rerender.astype(np.int32)).max())
                print(f"[render check] full-count re-render max pixel diff = {max_diff}")
                render_check_done = True

            for k in range(max_distractors + 1):
                variant_data[k].append(
                    _make_variant_episode(env, segment_steps, columns, episode_idx, k)
                )

            segment_start_xy = np.asarray(segment_steps[0]["xy"], dtype=np.float32)
            segment_goal_xy = np.asarray(segment_steps[-1]["xy"], dtype=np.float32)
            _, sampled_path_doors = env.shortest_path(start_xy, goal_xy)
            _, segment_path_doors = env.shortest_path(segment_start_xy, segment_goal_xy)
            tasks.append(
                {
                    "episode_idx": episode_idx,
                    "attempt": attempts,
                    "seed": attempt_seed,
                    "closed_door": env.closed_door,
                    "sampled_start_xy": np.asarray(start_xy).tolist(),
                    "sampled_goal_xy": np.asarray(goal_xy).tolist(),
                    "sampled_start_room": start_room,
                    "sampled_goal_room": goal_room,
                    "sampled_path_doors": list(sampled_path_doors),
                    "sampled_cross_n_rooms": int(len(sampled_path_doors)),
                    "segment_start_xy": segment_start_xy.tolist(),
                    "segment_goal_xy": segment_goal_xy.tolist(),
                    "segment_start_room": env._room_of(segment_start_xy),
                    "segment_goal_room": env._room_of(segment_goal_xy),
                    "segment_path_doors": list(segment_path_doors),
                    "segment_cross_n_rooms": int(len(segment_path_doors)),
                    "source_window": list(source_window),
                    "source_action_count": int(original_action_count),
                    "segment_action_count": int(len(segment_steps) - 1),
                    "segment_frame_count": int(len(segment_steps)),
                }
            )
            progress.update(1)
            progress.set_postfix(attempts=attempts, refresh=True)
        progress.close()
    finally:
        world.close()

    if len(tasks) < num_episodes:
        raise ValueError(
            f"Only generated {len(tasks)} eval trajectories after {max_attempts} "
            "attempts."
        )

    goal_offset = int(args.max_steps) if args.max_steps is not None else None
    goal_tag = goal_offset if goal_offset is not None else rollout_budget
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    world_container = OmegaConf.to_container(cfg.world, resolve=True)
    for k in range(max_distractors + 1):
        data = variant_data[k]
        columns = list(data[0].keys())
        payload = {
            "format": "fourroom_expert_eval_trajs_v1",
            "data": data,
            "columns": columns,
            "episodes_idx": list(range(len(data))),
            "start_steps": [0] * len(data),
            "goal_steps": [int(t["segment_frame_count"]) - 1 for t in tasks],
            "segment_lengths": [int(t["segment_frame_count"]) for t in tasks],
            "goal_offset_steps": goal_offset,
            "eval_budget": int(rollout_budget),
            "process": {},
            "task_info": {
                "data_config_path": str(Path(args.data_config_path).expanduser()),
                "world": world_container,
                "num_episodes": int(num_episodes),
                "seed": int(seed),
                "cross_n_rooms": int(args.cross_n_rooms),
                "max_steps": goal_offset,
                "rollout_budget": int(rollout_budget),
                "num_active_distractors": int(k),
                "max_distractors": int(max_distractors),
                "shared_behavior_sweep": True,
                "tasks": tasks,
                **generation_provenance(),
            },
        }
        out_path = output_dir / (
            f"{args.output_stem}_cross{int(args.cross_n_rooms)}"
            f"_goal{goal_tag}_n{num_episodes}_d{k}.pt"
        )
        torch.save(payload, out_path)
        print(f"Saved {len(data)} eval trajs (d={k}) to {out_path}")

        if args.save_videos:
            _save_selected_videos(data, tasks, out_path, args.video_fps)


class NoRoomPairError(ValueError):
    """Raised when the sampled door graph cannot realize a crossing count."""


def _optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive, null, or none.")
    return parsed


def _room_graph(env) -> dict[str, list[str]]:
    graph = {room: [] for room in ROOMS}
    for room, edges in env._valid_door_graph().items():
        graph.setdefault(room, [])
        for next_room, _door_name, _door_center in edges:
            graph[room].append(next_room)
    return graph


def _room_crossing_count(env, start_room: str, goal_room: str) -> int | None:
    if start_room == goal_room:
        return 0

    graph = _room_graph(env)
    queue = deque([(start_room, 0)])
    visited = {start_room}
    while queue:
        room, dist = queue.popleft()
        for next_room in graph.get(room, []):
            if next_room in visited:
                continue
            if next_room == goal_room:
                return dist + 1
            visited.add(next_room)
            queue.append((next_room, dist + 1))
    return None


def _room_pairs_for_crossings(env, cross_n_rooms: int) -> list[tuple[str, str]]:
    pairs = []
    for start_room in ROOMS:
        for goal_room in ROOMS:
            count = _room_crossing_count(env, start_room, goal_room)
            if count == cross_n_rooms:
                pairs.append((start_room, goal_room))
    return pairs


def _sample_room_pair(
    env,
    rng: np.random.Generator,
    cross_n_rooms: int,
) -> tuple[str, str]:
    pairs = _room_pairs_for_crossings(env, cross_n_rooms)
    if not pairs:
        closed = env.closed_door if env.closed_door is not None else "none"
        raise NoRoomPairError(
            f"No room pair has {cross_n_rooms} crossings with closed_door={closed}."
        )
    return pairs[int(rng.integers(0, len(pairs)))]


def _room_bounds(env, room: str) -> tuple[float, float, float, float]:
    radius = float(env.variation_space["agent"]["radius"].value.item())
    half_wall = int(env.variation_space["wall"]["thickness"].value.item()) // 2
    wall_gap = half_wall + radius + 1.0

    pos_min = float(env.BORDER_SIZE) + radius
    pos_max = float(env.IMG_SIZE - env.BORDER_SIZE) - radius
    center = float(env.WALL_CENTER)

    if room.endswith("_left"):
        x_low, x_high = pos_min, center - wall_gap
    else:
        x_low, x_high = center + wall_gap, pos_max

    if room.startswith("top_"):
        y_low, y_high = pos_min, center - wall_gap
    else:
        y_low, y_high = center + wall_gap, pos_max

    if x_low >= x_high or y_low >= y_high:
        raise ValueError(f"No valid sampling range for room {room!r}.")
    return x_low, x_high, y_low, y_high


def _sample_position_in_room(
    env,
    rng: np.random.Generator,
    room: str,
) -> np.ndarray:
    x_low, x_high, y_low, y_high = _room_bounds(env, room)
    for _ in range(100):
        xy = np.array(
            [
                rng.uniform(x_low, x_high),
                rng.uniform(y_low, y_high),
            ],
            dtype=np.float32,
        )
        if env._room_of(xy) == room:
            return xy
    raise ValueError(f"Failed to sample a valid position in room {room!r}.")


def _sample_crossing_task(
    env,
    rng: np.random.Generator,
    cross_n_rooms: int,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    start_room, goal_room = _sample_room_pair(env, rng, cross_n_rooms)
    start_xy = _sample_position_in_room(env, rng, start_room)
    goal_xy = _sample_position_in_room(env, rng, goal_room)
    return start_xy, goal_xy, start_room, goal_room


def _position_crossing_count(
    env,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
) -> int | None:
    path_length, door_names = env.shortest_path(start_xy, goal_xy)
    if not np.isfinite(path_length):
        return None
    return len(door_names)


def _valid_crossing_windows(
    env,
    steps: list[dict[str, Any]],
    max_steps: int,
    cross_n_rooms: int,
) -> list[tuple[int, int]]:
    windows = []
    if len(steps) <= max_steps:
        return windows

    for start_idx in range(0, len(steps) - max_steps):
        goal_idx = start_idx + max_steps
        start_xy = np.asarray(steps[start_idx]["xy"], dtype=np.float32)
        goal_xy = np.asarray(steps[goal_idx]["xy"], dtype=np.float32)
        if _position_crossing_count(env, start_xy, goal_xy) == cross_n_rooms:
            windows.append((start_idx, goal_idx))
    return windows


def _render_variant_pixels(env, xy, distractor_xy_full, k):
    """Re-render one frame showing only distractors 0..k-1 (agent on top)."""
    positions = np.asarray(distractor_xy_full, dtype=np.float32).reshape(-1, 2).copy()
    positions[k:] = np.nan
    env.distractor_positions = torch.as_tensor(positions, dtype=torch.float32)
    env._n_active = int(k)
    img_chw = env._render_frame(agent_pos=torch.as_tensor(xy, dtype=torch.float32))
    return img_chw.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)


def _make_variant_episode(env, segment_steps, columns, episode_idx, k):
    """Build one episode dict for active-count k from the shared segment steps."""
    variant_steps = []
    for step in segment_steps:
        s = deepcopy(step)
        xy = np.asarray(step["xy"], dtype=np.float32)
        dxy = np.asarray(step["distractor_xy"], dtype=np.float32).reshape(-1, 2).copy()
        dxy[k:] = np.nan
        s["distractor_xy"] = dxy
        for idx in range(env.max_distractors):
            s[f"distractor{idx}_xy"] = dxy[idx].copy()
        all_agent = np.concatenate([xy[None], dxy], axis=0).astype(np.float32)
        s["all_agent_xy"] = all_agent
        s["full_state"] = all_agent.reshape(-1).astype(np.float32)
        s["pixels"] = _render_variant_pixels(env, xy, step["distractor_xy"], k)
        variant_steps.append(s)
    return _make_episode_from_steps(variant_steps, list(columns), episode_idx)


if __name__ == "__main__":
    main()
