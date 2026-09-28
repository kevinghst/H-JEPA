#!/usr/bin/env python
"""Generate locomaze eval tasks by rolling out the OGBench expert policy."""

import argparse
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from planning_eval import generation_provenance, load_eval_config
from stable_worldmodel.envs.ogbench import (
    AntMazeEnv,
    AntMazeExplorePolicy,
    HumanoidMazeEnv,
    HumanoidMazeExplorePolicy,
)


CELL_SETS = ("all", "vertex")
ENV_SPECS = {
    "swm/OGBAntMaze-v0": {
        "env_cls": AntMazeEnv,
        "policy_cls": AntMazeExplorePolicy,
        "label": "AntMaze",
    },
    "swm/OGBHumanoidMaze-v0": {
        "env_cls": HumanoidMazeEnv,
        "policy_cls": HumanoidMazeExplorePolicy,
        "label": "HumanoidMaze",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate visual locomaze eval trajectories by sampling start/goal "
            "cells at controlled shortest-path grid distances and rolling out "
            "the matching OGBench expert policy to the sampled goal."
        )
    )
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config-name",
        help="Eval config name under config/eval, e.g. ant_d1_gd_online_proprio.",
    )
    config_group.add_argument("--config-path", help="Path to an eval config YAML.")
    parser.add_argument("--output-path", required=True, help="Where to write the .pt file.")
    parser.add_argument("--d-low", type=int, required=True, help="Minimum grid distance.")
    parser.add_argument("--d-high", type=int, required=True, help="Maximum grid distance.")
    parser.add_argument(
        "--num-episodes",
        type=int,
        required=True,
        help="Number of successful expert trajectories to save.",
    )
    parser.add_argument(
        "--goal-cell-set",
        choices=CELL_SETS,
        default="all",
        help="Goal cell candidate set. Starts always use all walkable cells.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Generation seed.")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help=(
            "Deprecated no-op kept for command compatibility. Evaluation "
            "normalizers are loaded from the policy training run."
        ),
    )
    parser.add_argument(
        "--policy-dataset-type",
        default="stitch",
        choices=("navigate", "stitch"),
        help=(
            "Expert policy mode. 'stitch' follows the fixed goal without "
            "resampling a new navigation goal after success."
        ),
    )
    parser.add_argument(
        "--policy-noise",
        type=float,
        default=None,
        help="Gaussian action noise for the expert policy. Defaults to policy config/null.",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=None,
        help="Optional minimum successful trajectory length in environment steps.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional maximum successful trajectory length in environment steps.",
    )
    parser.add_argument(
        "--rollout-budget",
        type=int,
        required=True,
        help="Maximum expert rollout steps per sampled expert task attempt.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum sampled tasks to try. Defaults to 20 * num_episodes.",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help=(
            "Also save one mp4 per selected trajectory in a sibling directory "
            "named after the output .pt file stem."
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Frames per second for videos saved with --save-videos.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.d_low < 0:
        raise ValueError("--d-low must be non-negative.")
    if args.d_high < args.d_low:
        raise ValueError("--d-high must be >= --d-low.")
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")
    if args.min_steps is not None and args.min_steps < 2:
        raise ValueError("--min-steps must be at least 2.")
    if args.max_steps is not None and args.max_steps < 2:
        raise ValueError("--max-steps must be at least 2.")
    if args.min_steps is not None and args.max_steps is not None:
        if args.max_steps < args.min_steps:
            raise ValueError("--max-steps must be >= --min-steps.")
    if args.rollout_budget is not None and args.rollout_budget < 1:
        raise ValueError("--rollout-budget must be positive.")
    if args.max_attempts is not None and args.max_attempts < args.num_episodes:
        raise ValueError("--max-attempts must be >= --num-episodes.")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive.")


def _open_neighbors(
    cell: tuple[int, int],
    open_cells: set[tuple[int, int]],
) -> Iterable[tuple[int, int]]:
    i, j = cell
    for di, dj in ((-1, 0), (0, -1), (1, 0), (0, 1)):
        neighbor = (i + di, j + dj)
        if neighbor in open_cells:
            yield neighbor


def _bfs_distances(
    start: tuple[int, int],
    open_cells: set[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    distances = {start: 0}
    queue = deque([start])

    while queue:
        cell = queue.popleft()
        next_distance = distances[cell] + 1
        for neighbor in _open_neighbors(cell, open_cells):
            if neighbor in distances:
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)

    return distances


def _distance_candidates(
    starts: list[tuple[int, int]],
    goals: set[tuple[int, int]],
    open_cells: set[tuple[int, int]],
    d_low: int,
    d_high: int,
) -> dict[int, dict[tuple[int, int], list[tuple[int, int]]]]:
    candidates: dict[int, dict[tuple[int, int], list[tuple[int, int]]]] = {
        distance: {} for distance in range(d_low, d_high + 1)
    }

    for start in starts:
        distances = _bfs_distances(start, open_cells)
        for goal, distance in distances.items():
            if distance < d_low or distance > d_high or goal not in goals:
                continue
            candidates[distance].setdefault(start, []).append(goal)

    return {
        distance: by_start
        for distance, by_start in candidates.items()
        if by_start
    }


def _sample_xy_from_cell(
    env: AntMazeEnv | HumanoidMazeEnv,
    cell: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    base_xy = np.asarray(env.env.ij_to_xy(cell), dtype=np.float64)
    noise = rng.uniform(
        low=-float(env.env._noise),
        high=float(env.env._noise),
        size=2,
    )
    noise *= float(env.env._maze_unit) / 4.0
    return (base_xy + noise).astype(np.float32)


def _env_spec(cfg) -> dict[str, Any]:
    env_name = str(cfg.world.get("env_name", ""))
    if env_name not in ENV_SPECS:
        supported = "', '".join(ENV_SPECS)
        raise ValueError(
            f"This script only supports world.env_name in ('{supported}'), "
            f"got {env_name!r}."
        )
    return ENV_SPECS[env_name]


def _build_locomaze_env(cfg) -> AntMazeEnv | HumanoidMazeEnv:
    spec = _env_spec(cfg)

    allowed_keys = {
        "maze_type",
        "dataset_type",
        "stitch_distance",
        "ob_type",
        "width",
        "height",
        "camera_name",
        "terminate_at_goal",
        "success_timing",
        "add_noise_to_goal",
        "reward_task_id",
        "use_oracle_rep",
    }
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    env_kwargs = {
        key: value for key, value in world_cfg.items() if key in allowed_keys
    }
    return spec["env_cls"](**env_kwargs)


def _find_wrapped_env_with_attr(env, attr_name: str):
    cur = env
    while cur is not None:
        if hasattr(cur, attr_name):
            return getattr(cur, attr_name)
        cur = getattr(cur, "env", None)
    return None


def _set_info_value(infos: dict[str, Any], key: str, value: Any) -> None:
    value = np.asarray(value)
    if key not in infos:
        infos[key] = value[None]
        return

    current = infos[key]
    if isinstance(current, np.ndarray):
        current[0] = value
    else:
        infos[key] = value[None]


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


def _write_task_info(
    infos: dict[str, Any],
    start_info: dict[str, np.ndarray],
    goal_info: dict[str, Any],
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
) -> None:
    _set_info_value(infos, "goal", goal_info["pixels"])
    _set_info_value(infos, "goal_xy", goal_info["xy"])
    _set_info_value(infos, "goal_qpos", goal_info["qpos"])
    _set_info_value(infos, "goal_qvel", goal_info["qvel"])
    _set_info_value(infos, "goal_proprio", goal_info["proprio"])
    _set_info_value(infos, "goal_ij", np.asarray(goal_cell, dtype=np.int32))
    _set_info_value(infos, "init_ij", np.asarray(start_cell, dtype=np.int32))
    _set_info_value(infos, "init_xy", start_info["xy"])


def _step_list_to_episode(steps: list[dict[str, Any]], columns: list[str]) -> dict[str, Any]:
    episode = {}
    for key in columns:
        if key not in steps[0]:
            continue

        values = [step[key] for step in steps]
        if key in {"action", "direction"} and len(values) > 1:
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
        if key in ("pixels", "goal") and tensor.ndim == 4 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
        episode[key] = tensor

    return episode


def _merge_source_value(
    info: dict[str, Any],
    source,
    dim: int = -1,
) -> np.ndarray:
    if OmegaConf.is_config(source):
        source = OmegaConf.to_container(source, resolve=True)

    if isinstance(source, str):
        if source not in info:
            raise KeyError(f"Cannot merge missing generated column {source!r}.")
        return np.asarray(info[source])

    if isinstance(source, dict):
        key = source.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("Merge source dicts must define a non-empty 'key'.")
        if key not in info:
            raise KeyError(f"Cannot merge missing generated column {key!r}.")

        value = np.asarray(info[key])
        start = source.get("start", None)
        end = source.get("end", None)
        if start is None and end is None:
            return value

        axis = dim if dim >= 0 else value.ndim + dim
        if axis < 0 or axis >= value.ndim:
            raise ValueError(
                f"Cannot slice merge source {key!r} along dim={dim}; "
                f"source has {value.ndim} dimensions."
            )
        slices = [slice(None)] * value.ndim
        slices[axis] = slice(
            None if start is None else int(start),
            None if end is None else int(end),
        )
        return value[tuple(slices)]

    raise TypeError(f"Unsupported merge source type: {type(source).__name__}.")


def _apply_configured_merges(info: dict[str, Any], cfg) -> None:
    keys_to_merge = cfg.dataset.get("keys_to_merge", None)
    if keys_to_merge is None:
        return

    for target, source in keys_to_merge.items():
        if OmegaConf.is_config(source):
            source = OmegaConf.to_container(source, resolve=True)

        if isinstance(source, (str, dict)):
            source_items = [source]
        else:
            source_items = list(source)
        parts = [_merge_source_value(info, item) for item in source_items]
        info[target] = np.concatenate(parts, axis=-1).astype(np.float32)


def _shift_action_aligned_columns(steps: list[dict[str, Any]], columns: Iterable[str]) -> None:
    for key in columns:
        if key not in steps[0]:
            continue

        values = [step[key] for step in steps]
        replacement = np.full_like(np.asarray(values[-1]), np.nan, dtype=np.float32)
        shifted = values[1:] + [replacement]
        for step, value in zip(steps, shifted, strict=True):
            step[key] = value


def _is_success(info: dict[str, Any]) -> bool:
    success = info.get("success")
    if success is None:
        return False
    return bool(float(np.asarray(success).reshape(-1)[0]) > 0.0)


def _trajectory_length_ok(length: int, args: argparse.Namespace) -> bool:
    if args.min_steps is not None and length < args.min_steps:
        return False
    if args.max_steps is not None and length > args.max_steps:
        return False
    return True


def _build_world(cfg, rollout_budget: int):
    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))
    world_cfg.num_envs = 1
    world_cfg.max_episode_steps = int(rollout_budget)
    world_cfg.terminate_at_goal = True

    img_size = cfg.eval.img_size
    image_shape = tuple(int(size) for size in img_size) if isinstance(img_size, list | tuple) else (int(img_size), int(img_size))
    return swm.World(**world_cfg, image_shape=image_shape, verbose=0)


def _build_policy(cfg, args: argparse.Namespace) -> AntMazeExplorePolicy | HumanoidMazeExplorePolicy:
    policy_cfg = {}
    if "policy" in cfg and OmegaConf.is_config(cfg.policy):
        policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)

    policy_cfg["dataset_type"] = args.policy_dataset_type
    policy_cfg["seed"] = args.seed
    if args.policy_noise is not None:
        policy_cfg["noise"] = args.policy_noise

    return _env_spec(cfg)["policy_cls"](**policy_cfg)


def _prepare_rollout(
    world,
    start_info: dict[str, np.ndarray],
    goal_info: dict[str, Any],
    *,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    seed: int,
) -> None:
    state = np.concatenate([start_info["qpos"], start_info["qvel"]]).astype(np.float64)
    options = {
        "state": state,
        "task_info": {
            "init_ij": tuple(start_cell),
            "goal_ij": tuple(goal_cell),
        },
    }
    world.reset(seed=seed, options=options)

    env = world.envs.unwrapped.envs[0]
    set_goal = _find_wrapped_env_with_attr(env, "set_navigation_goal")
    if set_goal is None:
        raise RuntimeError("Locomaze env does not expose set_navigation_goal().")
    set_goal(goal_xy=goal_info["xy"])
    _write_task_info(world.infos, start_info, goal_info, start_cell, goal_cell)


def _rollout_expert_task(
    world,
    start_info: dict[str, np.ndarray],
    goal_info: dict[str, Any],
    task: dict[str, Any],
    cfg,
    *,
    seed: int,
    rollout_budget: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    _prepare_rollout(
        world,
        start_info,
        goal_info,
        start_cell=tuple(task["start_cell"]),
        goal_cell=tuple(task["goal_cell"]),
        seed=seed,
    )

    columns = [key for key in world.infos.keys() if not key.startswith("_")]
    steps = [_dump_current_info(world.infos, columns)]
    success = _is_success(world.infos)

    for _ in range(int(rollout_budget)):
        if success:
            break
        world.step()
        _write_task_info(
            world.infos,
            start_info,
            goal_info,
            tuple(task["start_cell"]),
            tuple(task["goal_cell"]),
        )
        steps.append(_dump_current_info(world.infos, columns))
        success = bool(world.terminateds[0]) or _is_success(world.infos)

    if not success:
        return None

    _shift_action_aligned_columns(steps, ("action", "direction"))
    for step in steps:
        _apply_configured_merges(step, cfg)
    episode = _step_list_to_episode(steps, columns)
    segment_len = len(steps)
    if not _trajectory_length_ok(segment_len, args):
        return None

    final_xy = np.asarray(steps[-1].get("xy"), dtype=np.float32)
    goal_xy = np.asarray(goal_info["xy"], dtype=np.float32)
    task_info = {
        "episode_idx": int(task["episode_idx"]),
        "grid_distance": int(task["grid_distance"]),
        "start_ij": list(task["start_cell"]),
        "goal_ij": list(task["goal_cell"]),
        "requested_start_xy": np.asarray(task["requested_start_xy"]).tolist(),
        "requested_goal_xy": np.asarray(task["requested_goal_xy"]).tolist(),
        "settled_start_xy": start_info["xy"].tolist(),
        "settled_goal_xy": goal_info["xy"].tolist(),
        "final_xy": final_xy.tolist(),
        "final_goal_l2": float(np.linalg.norm(final_xy - goal_xy)),
        "segment_len": int(segment_len),
    }
    return episode, task_info


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
    data: list[dict],
    tasks: list[dict],
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
                f"_d{int(task['grid_distance'])}"
                f"_len_{int(task['segment_len'])}.mp4"
            )
        )
        with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
            for frame in frames:
                writer.append_data(frame)
        written_paths.append(video_path)

    return written_paths


def main() -> None:
    args = parse_args()
    _validate_args(args)

    cfg = load_eval_config(config_name=args.config_name, config_path=args.config_path)
    rng = np.random.default_rng(args.seed)
    rollout_budget = int(args.rollout_budget)

    spec = _env_spec(cfg)
    env_label = spec["label"]
    sampler_env = _build_locomaze_env(cfg)
    try:
        sampler_env.reset(seed=args.seed)
        all_cells = [tuple(cell) for cell in sampler_env.all_cells]
        vertex_cells = {tuple(cell) for cell in sampler_env.vertex_cells}
        open_cells = set(all_cells)
        goal_cells = open_cells if args.goal_cell_set == "all" else vertex_cells
        candidates = _distance_candidates(
            starts=all_cells,
            goals=goal_cells,
            open_cells=open_cells,
            d_low=args.d_low,
            d_high=args.d_high,
        )
        if not candidates:
            raise ValueError(
                f"No valid start/goal cell pairs found for D in [{args.d_low}, {args.d_high}]."
            )

        available_distances = np.asarray(sorted(candidates), dtype=np.int64)
        world = _build_world(cfg, rollout_budget)
        try:
            world.set_policy(_build_policy(cfg, args))
            data = []
            task_infos = []
            attempts = 0
            max_attempts = args.max_attempts or (20 * args.num_episodes)

            while len(data) < args.num_episodes and attempts < max_attempts:
                attempts += 1
                distance = int(rng.choice(available_distances))
                by_start = candidates[distance]
                start_cells = list(by_start)
                start_cell = start_cells[int(rng.integers(len(start_cells)))]
                goal_options = by_start[start_cell]
                goal_cell = goal_options[int(rng.integers(len(goal_options)))]

                requested_start_xy = _sample_xy_from_cell(sampler_env, start_cell, rng)
                requested_goal_xy = _sample_xy_from_cell(sampler_env, goal_cell, rng)
                start_info = sampler_env.settled_info_from_xy(
                    requested_start_xy,
                    render_pixels=True,
                )
                goal_info = sampler_env.settled_info_from_xy(
                    requested_goal_xy,
                    render_pixels=True,
                )
                _apply_configured_merges(start_info, cfg)
                _apply_configured_merges(goal_info, cfg)

                task = {
                    "episode_idx": len(data),
                    "grid_distance": distance,
                    "start_cell": start_cell,
                    "goal_cell": goal_cell,
                    "requested_start_xy": requested_start_xy,
                    "requested_goal_xy": requested_goal_xy,
                }
                result = _rollout_expert_task(
                    world,
                    start_info,
                    goal_info,
                    task,
                    cfg,
                    seed=args.seed * 1_000_003 + attempts,
                    rollout_budget=rollout_budget,
                    args=args,
                )
                if result is None:
                    continue

                episode, task_info = result
                data.append(episode)
                task_infos.append(task_info)

            if len(data) < args.num_episodes:
                raise ValueError(
                    f"Only generated {len(data)} successful trajectories after "
                    f"{attempts} attempts. Increase --max-attempts, relax step "
                    "filters, or increase --rollout-budget."
                )
        finally:
            world.close()
    finally:
        sampler_env.close()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(data[0].keys())
    payload = {
        "format": "ant_expert_grid_trajs_v1",
        "data": data,
        "columns": columns,
        "episodes_idx": list(range(len(data))),
        "start_steps": [0] * len(data),
        "goal_steps": [int(task["segment_len"]) - 1 for task in task_infos],
        "segment_lengths": [int(task["segment_len"]) for task in task_infos],
        "goal_offset_steps": None,
        "eval_budget": int(cfg.eval.eval_budget),
        "task_info": {
            "maze_type": str(cfg.world.maze_type),
            "d_low": args.d_low,
            "d_high": args.d_high,
            "num_episodes": args.num_episodes,
            "goal_cell_set": args.goal_cell_set,
            "seed": args.seed,
            "normalizer_source": "policy_training_run",
            "policy_dataset_type": args.policy_dataset_type,
            "policy_noise": args.policy_noise,
            "min_steps": args.min_steps,
            "max_steps": args.max_steps,
            "rollout_budget": rollout_budget,
            "attempts": attempts,
            "tasks": task_infos,
            "config_name": args.config_name,
            "config_path": args.config_path,
            **generation_provenance(),
        },
    }
    torch.save(payload, output_path)

    if args.save_videos:
        video_paths = _save_selected_videos(data, task_infos, output_path, args.video_fps)
    else:
        video_paths = []

    candidate_counts = {
        int(distance): sum(len(goals) for goals in by_start.values())
        for distance, by_start in candidates.items()
    }
    sampled_distances = [task["grid_distance"] for task in task_infos]
    sampled_counts = {
        int(distance): int(sampled_distances.count(int(distance)))
        for distance in sorted(set(sampled_distances))
    }
    lengths = np.asarray([task["segment_len"] for task in task_infos])

    print(f"Saved {len(data)} {env_label} expert-grid eval trajectories to {output_path}")
    if video_paths:
        print(f"Saved {len(video_paths)} trajectory videos to {output_path.with_suffix('')}")
    print(f"Available pair counts by distance: {candidate_counts}")
    print(f"Sampled task counts by distance: {sampled_counts}")
    print(f"Generation attempts: {attempts}")
    print(
        "Segment lengths: "
        f"min={int(lengths.min())}, mean={float(lengths.mean()):.2f}, "
        f"max={int(lengths.max())}"
    )


if __name__ == "__main__":
    main()
