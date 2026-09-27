#!/usr/bin/env python
"""Dump eval trajectory chunks sampled directly from an offline dataset."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from planning_eval import generation_provenance, get_dataset, get_episodes_length, load_eval_config


SAMPLING_MODES = ("random", "cube_pickup_centered", "cube_pickup_centered_lift")
FILTER_NAMES = ("upright", "moving")


@dataclass
class CandidateSource:
    mode: str
    candidate_count: int
    max_attempts: int
    random_episode_ids: np.ndarray | None = None
    random_valid_counts: np.ndarray | None = None
    candidate_episodes: np.ndarray | None = None
    candidate_starts: np.ndarray | None = None
    candidate_order: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate eval trajectory files by sampling fixed-offset chunks "
            "directly from an offline dataset."
        )
    )
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config-name",
        help="Eval config name under config/eval.",
    )
    config_group.add_argument("--config-path", help="Path to an eval config YAML.")
    parser.add_argument("--output-path", required=True, help="Where to write the .pt file.")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--goal-offset-steps", type=int, default=None)
    parser.add_argument("--eval-budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset to sample. Defaults to eval.dataset_name from the config.",
    )
    parser.add_argument(
        "--traj-sampling-mode",
        choices=SAMPLING_MODES,
        default=None,
        help=(
            "Candidate sampling mode. 'cube_pickup_centered' requires "
            "privileged_block_0_pos. Defaults to eval.traj_sampling_mode."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Maximum unique candidates to test. Defaults to up to 100 * "
            "num_episodes for random sampling, or all candidates for "
            "precomputed sampling modes."
        ),
    )
    parser.add_argument(
        "--filter",
        dest="filters",
        choices=FILTER_NAMES,
        action="append",
        default=[],
        help=(
            "Chunk filter to apply. Can be passed more than once. "
            "'upright' requires qpos; 'moving' requires xy or qpos."
        ),
    )
    parser.add_argument(
        "--require-upright-moving",
        action="store_true",
        help="Legacy alias for --filter upright --filter moving.",
    )
    parser.add_argument(
        "--require-upright",
        action="store_true",
        help="Legacy alias for --filter upright.",
    )
    parser.add_argument(
        "--min-torso-height",
        type=float,
        default=0.7,
        help="Minimum qpos[2] for every step when upright filtering is enabled.",
    )
    parser.add_argument(
        "--require-moving",
        action="store_true",
        help="Legacy alias for --filter moving.",
    )
    parser.add_argument(
        "--min-xy-displacement",
        type=float,
        default=0.5,
        help="Minimum final-minus-start XY displacement for moving chunks.",
    )
    parser.add_argument(
        "--min-mean-xy-speed",
        type=float,
        default=0.01,
        help="Minimum mean per-step XY speed for moving chunks.",
    )
    parser.add_argument(
        "--moving-speed-threshold",
        type=float,
        default=0.005,
        help="Per-step XY speed threshold used by --min-moving-step-frac.",
    )
    parser.add_argument(
        "--min-moving-step-frac",
        type=float,
        default=0.5,
        help="Minimum fraction of steps above --moving-speed-threshold.",
    )
    parser.add_argument(
        "--pickup-height-delta",
        type=float,
        default=None,
        help=(
            "Cube pickup-centered sampling height delta. Defaults to "
            "eval.pickup_height_delta or 0.02."
        ),
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
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf dotlist overrides applied to the eval config.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_episodes is not None and args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")
    if args.goal_offset_steps is not None and args.goal_offset_steps <= 0:
        raise ValueError("--goal-offset-steps must be positive.")
    if args.eval_budget is not None and args.eval_budget <= 0:
        raise ValueError("--eval-budget must be positive.")
    if args.max_attempts is not None and args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive.")
    if args.min_torso_height < 0.0:
        raise ValueError("--min-torso-height must be non-negative.")
    if args.min_xy_displacement < 0.0:
        raise ValueError("--min-xy-displacement must be non-negative.")
    if args.min_mean_xy_speed < 0.0:
        raise ValueError("--min-mean-xy-speed must be non-negative.")
    if args.moving_speed_threshold < 0.0:
        raise ValueError("--moving-speed-threshold must be non-negative.")
    if not 0.0 <= args.min_moving_step_frac <= 1.0:
        raise ValueError("--min-moving-step-frac must be in [0, 1].")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive.")


def _cfg_int(cfg, path: str, fallback: int | None = None) -> int:
    value = OmegaConf.select(cfg, path, default=fallback)
    if value is None:
        raise ValueError(f"Missing required config value: {path}")
    return int(value)


def _episode_column(dataset) -> str:
    if "episode_idx" in dataset.column_names:
        return "episode_idx"
    if "ep_idx" in dataset.column_names:
        return "ep_idx"
    raise KeyError("Dataset must contain 'episode_idx' or 'ep_idx'.")


def _random_candidate_info(dataset, goal_offset_steps: int) -> tuple[np.ndarray, np.ndarray]:
    col_name = _episode_column(dataset)
    ep_ids = np.unique(dataset.get_col_data(col_name))
    episode_len = get_episodes_length(dataset, ep_ids)
    valid_counts = episode_len - goal_offset_steps
    keep = valid_counts > 0
    return ep_ids[keep].astype(np.int64), valid_counts[keep].astype(np.int64)


def _sample_random_candidate(
    rng: np.random.Generator,
    episode_ids: np.ndarray,
    valid_counts: np.ndarray,
) -> tuple[int, int]:
    cumulative = np.cumsum(valid_counts)
    flat_idx = int(rng.integers(int(cumulative[-1])))
    episode_pos = int(np.searchsorted(cumulative, flat_idx, side="right"))
    previous_total = 0 if episode_pos == 0 else int(cumulative[episode_pos - 1])
    return int(episode_ids[episode_pos]), int(flat_idx - previous_total)


def _candidate_starts_cube_pickup_centered(
    cfg,
    dataset,
    goal_offset_steps: int,
    pickup_height_delta: float,
    center_on: str = "grasp",
) -> tuple[np.ndarray, np.ndarray]:
    required_cols = ["privileged_block_0_pos"]
    if center_on == "grasp":
        required_cols.append("proprio_gripper_contact")
    for required in required_cols:
        if required not in dataset.column_names:
            raise ValueError(
                f"cube_pickup_centered sampling requires '{required}'."
            )

    midpoint = goal_offset_steps // 2
    contact_threshold = float(cfg.eval.get("pickup_contact_threshold", 0.5))
    block_pos = dataset.get_col_data("privileged_block_0_pos")
    if block_pos.ndim != 2 or block_pos.shape[1] < 3:
        raise ValueError(
            "'privileged_block_0_pos' must have shape [num_steps, >=3]."
        )
    if center_on == "grasp":
        contact_col = dataset.get_col_data("proprio_gripper_contact")
        contact_col = contact_col[:, 0] if contact_col.ndim == 2 else contact_col

    candidate_episodes = []
    candidate_starts = []
    for episode_idx, (offset, length) in enumerate(zip(dataset.offsets, dataset.lengths)):
        if length < goal_offset_steps:
            continue

        z_pos = block_pos[offset : offset + length, 2]
        lifted_steps = np.flatnonzero(z_pos >= z_pos[0] + pickup_height_delta)
        if lifted_steps.size == 0:
            continue
        lift_step = int(lifted_steps[0])

        if center_on == "lift":
            center_step = lift_step
        else:
            # Center on the grasp, not the lift: walk back from the lift over the
            # contiguous gripper-contact run to the step contact began. The lift
            # only fires once the cube is airborne (~4-5 steps after the grasp).
            contact = contact_col[offset : offset + length] >= contact_threshold
            if contact[lift_step]:
                grasp_end = lift_step
            else:
                prior_contact = np.flatnonzero(contact[:lift_step + 1])
                if prior_contact.size == 0:
                    continue
                grasp_end = int(prior_contact[-1])
            grasp_step = grasp_end
            while grasp_step - 1 >= 0 and contact[grasp_step - 1]:
                grasp_step -= 1
            center_step = grasp_step

        start_step = center_step - midpoint
        end_step = start_step + goal_offset_steps
        if start_step < 0 or end_step > length:
            continue

        candidate_episodes.append(episode_idx)
        candidate_starts.append(start_step)

    return (
        np.asarray(candidate_episodes, dtype=np.int64),
        np.asarray(candidate_starts, dtype=np.int64),
    )


def _candidate_starts(
    cfg,
    dataset,
    *,
    sampling_mode: str,
    goal_offset_steps: int,
    pickup_height_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sampling_mode in ("cube_pickup_centered", "cube_pickup_centered_lift"):
        return _candidate_starts_cube_pickup_centered(
            cfg,
            dataset,
            goal_offset_steps,
            pickup_height_delta,
            center_on="lift" if sampling_mode == "cube_pickup_centered_lift" else "grasp",
        )
    raise ValueError(f"Unsupported eval trajectory sampling mode: {sampling_mode}")


def _build_candidate_source(
    cfg,
    dataset,
    *,
    sampling_mode: str,
    goal_offset_steps: int,
    pickup_height_delta: float,
    rng: np.random.Generator,
    max_attempts: int | None,
    num_episodes: int,
) -> CandidateSource:
    if sampling_mode == "random":
        episode_ids, valid_counts = _random_candidate_info(
            dataset,
            goal_offset_steps,
        )
        if episode_ids.size == 0:
            raise ValueError("No candidate eval chunks found.")

        candidate_count = int(np.sum(valid_counts))
        attempt_budget = int(max_attempts or (100 * num_episodes))
        return CandidateSource(
            mode=sampling_mode,
            candidate_count=candidate_count,
            max_attempts=min(attempt_budget, candidate_count),
            random_episode_ids=episode_ids,
            random_valid_counts=valid_counts,
        )

    candidate_episodes, candidate_starts = _candidate_starts(
        cfg,
        dataset,
        sampling_mode=sampling_mode,
        goal_offset_steps=goal_offset_steps,
        pickup_height_delta=pickup_height_delta,
    )
    if candidate_episodes.size == 0:
        raise ValueError("No candidate eval chunks found.")

    candidate_count = int(candidate_episodes.size)
    attempt_budget = int(max_attempts or candidate_count)
    order = rng.permutation(candidate_count)
    return CandidateSource(
        mode=sampling_mode,
        candidate_count=candidate_count,
        max_attempts=min(attempt_budget, candidate_count),
        candidate_episodes=candidate_episodes,
        candidate_starts=candidate_starts,
        candidate_order=order[:attempt_budget],
    )


def _sample_candidate(
    source: CandidateSource,
    rng: np.random.Generator,
    attempt: int,
    attempted_pairs: set[tuple[int, int]],
) -> tuple[int, int] | None:
    if source.mode == "random":
        if len(attempted_pairs) >= source.candidate_count:
            return None
        if (
            source.random_episode_ids is None
            or source.random_valid_counts is None
        ):
            raise ValueError(
                "Random candidate source is missing random metadata."
            )

        episode_idx, start_step = _sample_random_candidate(
            rng,
            source.random_episode_ids,
            source.random_valid_counts,
        )
        while (episode_idx, start_step) in attempted_pairs:
            episode_idx, start_step = _sample_random_candidate(
                rng,
                source.random_episode_ids,
                source.random_valid_counts,
            )
        return episode_idx, start_step

    if (
        source.candidate_order is None
        or source.candidate_episodes is None
        or source.candidate_starts is None
        or attempt > len(source.candidate_order)
    ):
        return None

    candidate_idx = int(source.candidate_order[attempt - 1])
    return (
        int(source.candidate_episodes[candidate_idx]),
        int(source.candidate_starts[candidate_idx]),
    )


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _chunk_xy(chunk: dict[str, Any]) -> np.ndarray:
    if "xy" in chunk:
        xy = _to_numpy(chunk["xy"])
    elif "qpos" in chunk:
        xy = _to_numpy(chunk["qpos"])[..., :2]
    else:
        raise KeyError("Moving filter requires dataset column 'xy' or 'qpos'.")

    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Expected XY chunk with shape [T, >=2], got {xy.shape}.")
    return xy[:, :2].astype(np.float64)


def _chunk_qpos(chunk: dict[str, Any]) -> np.ndarray:
    if "qpos" not in chunk:
        raise KeyError("Upright filter requires dataset column 'qpos'.")
    qpos = _to_numpy(chunk["qpos"])
    if qpos.ndim != 2 or qpos.shape[1] <= 2:
        raise ValueError(f"Expected qpos chunk with shape [T, >2], got {qpos.shape}.")
    return qpos.astype(np.float64)


def _filter_chunk(
    chunk: dict[str, Any],
    *,
    filters: tuple[str, ...],
    args: argparse.Namespace,
) -> tuple[bool, dict[str, float]]:
    stats = {}

    if "upright" in filters:
        qpos = _chunk_qpos(chunk)
        min_torso_height = float(np.nanmin(qpos[:, 2]))
        stats["min_torso_height"] = min_torso_height
        if min_torso_height < float(args.min_torso_height):
            return False, stats

    if "moving" in filters:
        xy = _chunk_xy(chunk)
        step_speeds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        displacement = float(np.linalg.norm(xy[-1] - xy[0]))
        mean_speed = float(np.nanmean(step_speeds)) if step_speeds.size else 0.0
        moving_frac = (
            float(np.mean(step_speeds >= float(args.moving_speed_threshold)))
            if step_speeds.size
            else 0.0
        )
        stats.update(
            {
                "xy_displacement": displacement,
                "mean_xy_speed": mean_speed,
                "moving_step_frac": moving_frac,
            }
        )
        if displacement < float(args.min_xy_displacement):
            return False, stats
        if mean_speed < float(args.min_mean_xy_speed):
            return False, stats
        if moving_frac < float(args.min_moving_step_frac):
            return False, stats

    return True, stats


def _requested_filters(args: argparse.Namespace) -> tuple[str, ...]:
    requested = set(args.filters)
    if args.require_upright or args.require_upright_moving:
        requested.add("upright")
    if args.require_moving or args.require_upright_moving:
        requested.add("moving")
    return tuple(name for name in FILTER_NAMES if name in requested)


def _summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.nanmin(arr)),
        "mean": float(np.nanmean(arr)),
        "max": float(np.nanmax(arr)),
    }


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


def _episode_pixels(episode: dict[str, Any], episode_idx: int):
    for key in ("pixels", "observation"):
        if key in episode:
            return episode[key]
    raise KeyError(f"Episode {episode_idx} is missing 'pixels' and 'observation'.")


def _save_selected_videos(
    data: list[dict[str, Any]],
    output_path: Path,
    fps: int,
) -> list[Path]:
    import imageio.v2 as imageio

    video_dir = output_path.with_suffix("")
    video_dir.mkdir(parents=True, exist_ok=True)

    written_paths = []
    for episode_idx, episode in enumerate(data):
        frames = _pixels_to_frames(_episode_pixels(episode, episode_idx))
        video_path = video_dir / f"traj_{episode_idx:04d}_len_{len(frames)}.mp4"
        with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
            for frame in frames:
                writer.append_data(frame)
        written_paths.append(video_path)

    return written_paths


def main() -> None:
    args = parse_args()
    _validate_args(args)

    cfg = load_eval_config(config_name=args.config_name, config_path=args.config_path)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    num_episodes = args.num_episodes or _cfg_int(cfg, "eval.num_eval")
    goal_offset_steps = args.goal_offset_steps or _cfg_int(
        cfg,
        "eval.goal_offset_steps",
    )
    eval_budget = args.eval_budget or _cfg_int(cfg, "eval.eval_budget")
    seed = args.seed if args.seed is not None else _cfg_int(cfg, "seed")
    dataset_name = args.dataset_name or str(cfg.eval.dataset_name)
    sampling_mode = args.traj_sampling_mode or str(
        cfg.eval.get("traj_sampling_mode", "random")
    )
    pickup_height_delta = (
        float(args.pickup_height_delta)
        if args.pickup_height_delta is not None
        else float(cfg.eval.get("pickup_height_delta", 0.02))
    )

    filters = _requested_filters(args)

    dataset = get_dataset(cfg, dataset_name)
    rng = np.random.default_rng(seed)
    candidate_source = _build_candidate_source(
        cfg,
        dataset,
        sampling_mode=sampling_mode,
        goal_offset_steps=goal_offset_steps,
        pickup_height_delta=pickup_height_delta,
        rng=rng,
        max_attempts=args.max_attempts,
        num_episodes=num_episodes,
    )

    data = []
    episodes_idx = []
    start_steps = []
    selected_stats = []
    rejected = 0
    attempted_pairs: set[tuple[int, int]] = set()

    with tqdm(
        total=num_episodes,
        desc="Sampling dataset eval trajectories",
        unit="episode",
        dynamic_ncols=True,
    ) as progress:
        for attempt in range(1, candidate_source.max_attempts + 1):
            candidate = _sample_candidate(
                candidate_source,
                rng,
                attempt,
                attempted_pairs,
            )
            if candidate is None:
                break
            episode_idx, start_step = candidate

            attempted_pairs.add((episode_idx, start_step))
            end_step = start_step + goal_offset_steps

            chunk = dataset.load_chunk(
                np.asarray([episode_idx], dtype=np.int64),
                np.asarray([start_step], dtype=np.int64),
                np.asarray([end_step], dtype=np.int64),
            )[0]
            keep, stats = _filter_chunk(
                chunk,
                filters=filters,
                args=args,
            )
            if not keep:
                rejected += 1
                progress.set_postfix(
                    attempts=attempt,
                    rejected=rejected,
                    refresh=True,
                )
                continue

            data.append(chunk)
            episodes_idx.append(episode_idx)
            start_steps.append(start_step)
            selected_stats.append(stats)
            progress.update(1)
            progress.set_postfix(
                attempts=attempt,
                rejected=rejected,
                start=start_step,
                refresh=True,
            )

            if len(data) >= num_episodes:
                break

    if len(data) < num_episodes:
        raise ValueError(
            f"Only sampled {len(data)} eval trajectories from "
            f"{candidate_source.max_attempts} "
            "tested candidates. Relax filters, increase --max-attempts, or "
            "use a larger dataset."
        )

    stat_keys = sorted({key for stats in selected_stats for key in stats})
    filter_stats = {
        key: _summarize([stats[key] for stats in selected_stats if key in stats])
        for key in stat_keys
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "dataset_eval_trajs_v1",
        "data": data,
        "columns": list(dataset.column_names),
        "episodes_idx": episodes_idx,
        "start_steps": start_steps,
        "goal_offset_steps": int(goal_offset_steps),
        "eval_budget": int(eval_budget),
        "process": {},
        "task_info": {
            "dataset_name": dataset_name,
            "world_config_path": str(cfg.get("world_config_path", "")),
            "world": OmegaConf.to_container(cfg.get("world", {}), resolve=True),
            "sampling_mode": sampling_mode,
            "filters": list(filters),
            "seed": int(seed),
            "num_episodes": int(num_episodes),
            "candidate_count": int(candidate_source.candidate_count),
            "attempts": int(len(data) + rejected),
            "rejected": int(rejected),
            "require_upright": "upright" in filters,
            "require_moving": "moving" in filters,
            "min_torso_height": float(args.min_torso_height),
            "min_xy_displacement": float(args.min_xy_displacement),
            "min_mean_xy_speed": float(args.min_mean_xy_speed),
            "moving_speed_threshold": float(args.moving_speed_threshold),
            "min_moving_step_frac": float(args.min_moving_step_frac),
            "filter_stats": filter_stats,
            **generation_provenance(),
        },
    }
    torch.save(payload, output_path)

    if args.save_videos:
        video_paths = _save_selected_videos(data, output_path, args.video_fps)
    else:
        video_paths = []

    print(f"Saved {len(data)} dataset eval trajectories to {output_path}")
    if video_paths:
        print(f"Saved {len(video_paths)} trajectory videos to {output_path.with_suffix('')}")
    print(f"Candidate chunks: {candidate_source.candidate_count}")
    print(f"Generation attempts: {len(data) + rejected}")
    print(f"Rejected chunks: {rejected}")
    for key, stats in filter_stats.items():
        if stats is None:
            continue
        print(
            f"{key}: min={stats['min']:.4f}, "
            f"mean={stats['mean']:.4f}, max={stats['max']:.4f}"
        )


if __name__ == "__main__":
    main()
