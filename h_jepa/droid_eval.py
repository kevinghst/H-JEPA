"""Offline DROID planning eval on the 16 curated waypoint clips (5 fps, goal window 36), run by eval.py.

Each clip is planned start -> goal in one open-loop call and scored against the
ground-truth actions:
  * ATE: net-displacement L1 of the planned vs GT delta-pose sums; the headline is
    ate/end_distance_xyz, plus the reach / transport split at the grasp and the
    per-step (pairwise) variant.
  * Frechet skill-over-floor: discrete-Frechet distance between the planned and GT
    cumulative xyz paths, as a fraction of the do-nothing floor.

Planner settings come from config/eval/droid_{flat,l2}.yaml, overridable from the command line.
Paper cells (planner lr per level, level 1 first):
  LeWM+IDM  --config-name droid_flat solver.optimizer_kwargs.lr=0.01
  HWM       --config-name droid_l2 solver.solvers.level1.optimizer_kwargs.lr=0.01
            solver.solvers.level2.optimizer_kwargs.lr=0.3
  H-JEPA    --config-name droid_l2 solver.solvers.level1.optimizer_kwargs.lr=0.01
            solver.solvers.level2.optimizer_kwargs.lr=0.01

Outputs under output.dir (relative: to the ckpt dir): ep_{k}/actions.pt ({"planned", "gt",
"grasp_pos"}, raw units; k = manifest index), ep_{k}/planning_compute.json, plan_config.yaml and
eval.csv, aggregated over every ep_*/actions.pt present so one-clip shards
(start_index=k num_eval=1) can share one output dir. Each clip plans with planner seed + k
(as eval.chunk_size=1 in planning_eval), so a rerun into the same dir (e.g. a requeued job)
skips the clips already saved and gives the same result as one unbroken run.

  python eval.py --config-name droid_flat policy=$RUN/<run>_object.ckpt seed=1 output.dir=$OUT
"""

import csv
import json
import time
from pathlib import Path

import gymnasium
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import v2 as transforms

from data import IMAGENET_STATS
from eval_config_utils import _build_policy_plan_config
from planning_eval import (
    _load_policy_normalizer_artifact,
    build_solver,
    load_model,
    pop_solve_records,
    resolve_results_dir,
)
from traj_metrics import cumulative_delta, frechet_skill_over_floor


def _ate_components(delta: torch.Tensor, suffix: str = "") -> dict:
    return {
        f"ate/end_distance{suffix}": delta.sum().item(),
        f"ate/end_distance_xyz{suffix}": delta[:3].sum().item(),
        f"ate/end_distance_orientation{suffix}": delta[3:6].sum().item(),
        f"ate/end_distance_closure{suffix}": delta[6:].sum().item(),
    }


def clip_metrics(planned: torch.Tensor, gt: torch.Tensor, grasp_pos) -> dict:
    plan_len = planned.shape[0]
    metrics = _ate_components(cumulative_delta(planned, gt))
    if grasp_pos is not None and 0 < grasp_pos < plan_len:
        metrics.update(_ate_components(cumulative_delta(planned, gt, 0, grasp_pos), "_reach"))
        metrics.update(
            _ate_components(cumulative_delta(planned, gt, grasp_pos, plan_len), "_transport")
        )
    metrics.update(_ate_components((planned - gt).abs().sum(0), "_pairwise"))
    skill, model_frechet, floor = frechet_skill_over_floor(planned, gt)
    metrics.update({"frechet/model": model_frechet, "frechet/floor": floor, "frechet/skill": skill})
    return metrics


def aggregate(out_dir: Path) -> dict:
    # Mean over episodes for every key; the Frechet skill also as a median with each
    # episode repeated in proportion to its floor (GT path length).
    paths = sorted(out_dir.glob("ep_*/actions.pt"), key=lambda p: int(p.parent.name[3:]))
    per_ep = [clip_metrics(**torch.load(p)) for p in paths]
    agg = {k: float(np.nanmean([m[k] for m in per_ep])) for k in per_ep[0]}
    skills = np.array([m["frechet/skill"] for m in per_ep], dtype=float)
    floors = np.array([m["frechet/floor"] for m in per_ep], dtype=float)
    valid = ~np.isnan(skills)
    agg["frechet/skill_mean"] = float(np.mean(skills[valid]))
    w = np.clip(np.round(floors[valid] / max(floors[valid].min(), 1e-6)), 1, None).astype(int)
    agg["frechet/skill_median_weighted"] = float(np.median(np.repeat(skills[valid], w)))
    agg["n_episodes"] = len(per_ep)
    return agg


def _center_square_crop(x: torch.Tensor) -> torch.Tensor:
    # Eval-time crop of the training pipeline: the centered short-side square of the
    # wide DROID frame (resizing the full frame would squash it).
    h, w = x.shape[-2], x.shape[-1]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return x[..., top : top + s, left : left + s]


def img_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Lambda(_center_square_crop),
            transforms.Resize(size=[img_size, img_size], antialias=False),
            transforms.Normalize(**IMAGENET_STATS),
        ]
    )


def plan_clip(solver, tf, obs: dict, action_dim: int, eval_budget: int) -> np.ndarray:
    info = {
        "pixels": tf(obs["visual"][0])[None, None].cuda(),  # [1, 1, C, H, W]
        "goal": tf(obs["visual"][-1])[None, None].cuda(),  # [1, 1, C, H, W]
        "action": torch.zeros(1, 1, action_dim, device="cuda"),
    }
    return solver(info, steps_taken=0, eval_budget=eval_budget)["actions"][0].cpu().numpy()  # [K, A], normalized


def run_clip_eval(cfg: DictConfig, model=None, results_dir: str | Path | None = None) -> None:
    from droid_data import DROIDClipReader

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    out_dir = resolve_results_dir(cfg, results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = Path(__file__).resolve().parent / cfg.clips
    with open(clips) as f:
        clip0 = json.load(f)[0]
    ds = DROIDClipReader(
        data_path=None,
        camera_views=["left_mp4_path"],
        num_frames=clip0["num_frames"],
        fps=clip0["fps"],
        frozen_clips=str(clips),
        deterministic_getitem=True,
    )
    stop = len(ds) if cfg.num_eval is None else min(cfg.start_index + cfg.num_eval, len(ds))
    clip_ids = range(cfg.start_index, stop)

    model = load_model(cfg, model)
    action_stats = _load_policy_normalizer_artifact(cfg)["stats"]["action"]
    act_mean = np.asarray(action_stats["mean"]).reshape(-1)  # [A]
    act_std = np.asarray(action_stats["std"]).reshape(-1)  # [A]
    action_dim = act_mean.shape[0]

    solver = build_solver(cfg, model)
    solver.configure(
        action_space=gymnasium.spaces.Box(-1.0, 1.0, shape=(1, action_dim)),
        n_envs=1,
        config=_build_policy_plan_config(cfg),
    )
    OmegaConf.save(cfg, out_dir / "plan_config.yaml", resolve=True)
    print(f"clips {clip_ids.start}..{clip_ids.stop - 1} of {len(ds)} | ckpt={cfg.policy}")

    tf = img_transform(cfg.img_size)
    # As planning_eval's eval.chunk_size=1: every clip reseeds the planner generators with seed + k, so
    # its plan does not depend on the clips run before it and a rerun skips the clips already saved.
    leaves = getattr(solver, "level_solvers", {1: solver}).values()
    gens = [(s.torch_gen, s.torch_gen.initial_seed()) for s in leaves]
    t0 = time.time()
    for k in clip_ids:
        ep_dir = out_dir / f"ep_{k}"
        if (ep_dir / "actions.pt").exists():
            continue
        for gen, seed in gens:
            gen.manual_seed(seed + k)
        obs, actions, _states, _reward, env_info = ds[k]
        planned = plan_clip(solver, tf, obs, action_dim, int(cfg.eval_budget))  # [K, A]
        ep = {
            "planned": torch.tensor(planned * act_std + act_mean, dtype=torch.float32),  # [K, A]
            "gt": actions[: planned.shape[0]].to(torch.float32),  # [K, A]
            "grasp_pos": env_info["grasp_pos"] if env_info else None,
        }
        ep_dir.mkdir(exist_ok=True)
        (ep_dir / "planning_compute.json").write_text(json.dumps(pop_solve_records(solver)))
        torch.save(ep, ep_dir / "actions.pt")  # written last: marks the clip done
        m = clip_metrics(**ep)
        print(
            f"ep {k:02d}: ate_xyz={m['ate/end_distance_xyz']:.4f}m "
            f"frechet_skill={m['frechet/skill']:.4f}"
        )

    agg = aggregate(out_dir)
    with open(out_dir / "eval.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(agg))
        writer.writerow(list(agg.values()))
    print(f"{agg['n_episodes']} episodes in {out_dir} ({time.time() - t0:.1f}s)")
    print(f"ATE end_distance_xyz: {agg['ate/end_distance_xyz']:.4f} m")
    print(f"Frechet skill mean: {agg['frechet/skill_mean']:.4f}")
    print(f"Frechet skill median (weighted): {agg['frechet/skill_median_weighted']:.4f}")

