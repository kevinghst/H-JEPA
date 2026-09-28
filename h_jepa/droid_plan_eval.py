"""Offline DROID planning eval on the 16 curated waypoint clips (fps20, goal window 36).

Each clip is planned start -> goal in one open-loop call and scored against the
ground-truth actions:
  * ATE: net-displacement L1 of the planned vs GT delta-pose sums; the headline is
    ate/end_distance_xyz, plus the reach / transport split at the grasp and the
    per-step (pairwise) variant.
  * Frechet skill-over-floor: discrete-Frechet distance between the planned and GT
    cumulative xyz paths, as a fraction of the do-nothing floor.

Planner settings come from config/eval/<--config>.yaml (droid_flat, droid_l2, droid_l3,
droid_l4). --lr / --num-samples take one value per level, level 1 first; the last value
also applies to the levels above. Paper cells:
  flat     --lr 0.01 --num-samples 16
  HWM      --config droid_l2 --lr 0.01 0.1  --num-samples 16
  H-JEPA   --config droid_l2 --lr 0.03 0.01 --num-samples 4

Outputs under <out>/<tag>/: ep_{k}/actions.pt ({"planned", "gt", "grasp_pos"}, raw
units; k = manifest index), ep_{k}/planning_compute.json, plan_config.yaml and
eval.csv, aggregated over every ep_*/actions.pt present so one-clip shards
(--start-index k --num-eval 1) can share one out dir.

  python droid_plan_eval.py --ckpt $RUN/<run>_object.ckpt
  python droid_plan_eval.py --config droid_l2 --ckpt $HRUN/<run>_object.ckpt --out $OUT
"""

import argparse
import csv
import json
import time
from pathlib import Path

import gymnasium
import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from torchvision.transforms import v2 as transforms

from data import NORMALIZER_ARTIFACT_FILENAME, load_normalizer_artifact
from eval_config_utils import _build_policy_plan_config, _is_hierarchical_solver
from stable_worldmodel.solver.hierarchical_solver import build_hierarchical_solver
from traj_metrics import cumulative_delta, frechet_skill_over_floor

CONFIG_DIR = Path(__file__).resolve().parent / "config" / "eval"


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
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        ]
    )


def load_config(args):
    cfg = OmegaConf.load(CONFIG_DIR / f"{args.config}.yaml")
    if args.seed is not None:
        cfg.seed = args.seed
    levels = cfg.solver.solvers.values() if _is_hierarchical_solver(cfg) else [cfg.solver]
    for i, solver_cfg in enumerate(levels):
        if args.lr:
            solver_cfg.optimizer_kwargs.lr = args.lr[min(i, len(args.lr) - 1)]
        if args.num_samples:
            solver_cfg.num_samples = args.num_samples[min(i, len(args.num_samples) - 1)]
    return cfg


def build_solver(cfg, model, action_dim: int):
    if _is_hierarchical_solver(cfg):
        solver = build_hierarchical_solver(cfg, model)
    else:
        solver = hydra.utils.instantiate(cfg.solver, model=model)
    solver.configure(
        action_space=gymnasium.spaces.Box(-1.0, 1.0, shape=(1, action_dim)),
        n_envs=1,
        config=_build_policy_plan_config(cfg),
    )
    return solver


def pop_solve_records(solver) -> list:
    solvers = getattr(solver, "level_solvers", {1: solver})
    records = [r for level in sorted(solvers) for r in solvers[level].solve_records]
    for s in solvers.values():
        s.solve_records = []
    return records


def plan_clip(solver, tf, obs: dict, action_dim: int, eval_budget: int) -> np.ndarray:
    info = {
        "pixels": tf(obs["visual"][0])[None, None].cuda(),  # [1, 1, C, H, W]
        "goal": tf(obs["visual"][-1])[None, None].cuda(),  # [1, 1, C, H, W]
        "action": torch.zeros(1, 1, action_dim, device="cuda"),
    }
    return solver(info, steps_taken=0, eval_budget=eval_budget)["actions"][0].cpu().numpy()  # [K, A], normalized


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", required=True, help="<run>_object.ckpt")
    p.add_argument("--config", default="droid_flat", help="Eval config in config/eval/")
    p.add_argument("--out", default=None, help="Base output dir (default: the ckpt dir)")
    p.add_argument("--tag", default="wp_20fps_gw36_cur16v2", help="Output subfolder")
    p.add_argument("--start-index", type=int, default=0, help="First manifest clip")
    p.add_argument("--num-eval", type=int, default=None, help="Clips from --start-index")
    p.add_argument("--seed", type=int, default=None, help="Planner seed (default: yaml)")
    p.add_argument("--lr", type=float, nargs="+", help="AdamW lr per level (default: yaml)")
    p.add_argument("--num-samples", type=int, nargs="+", help="Samples per level (default: yaml)")
    args = p.parse_args()
    cfg = load_config(args)

    from droid_data import DROIDClipReader

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out or ckpt_path.parent) / args.tag
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
    stop = len(ds) if args.num_eval is None else min(args.start_index + args.num_eval, len(ds))
    clip_ids = range(args.start_index, stop)

    if _is_hierarchical_solver(cfg):
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        model = swm.policy.AutoCostModel(str(ckpt_path))
    model = model.eval().to("cuda")
    model.interpolate_pos_encoding = True

    normalizer = load_normalizer_artifact(ckpt_path.parent / NORMALIZER_ARTIFACT_FILENAME)
    action_stats = normalizer["stats"]["action"]
    act_mean = np.asarray(action_stats["mean"]).reshape(-1)  # [A]
    act_std = np.asarray(action_stats["std"]).reshape(-1)  # [A]
    action_dim = act_mean.shape[0]

    solver = build_solver(cfg, model, action_dim)
    cfg.ckpt = str(ckpt_path)
    OmegaConf.save(cfg, out_dir / "plan_config.yaml", resolve=True)
    print(f"clips {clip_ids.start}..{clip_ids.stop - 1} of {len(ds)} | ckpt={ckpt_path}")

    tf = img_transform(cfg.img_size)
    t0 = time.time()
    for k in clip_ids:
        obs, actions, _states, _reward, env_info = ds[k]
        planned = plan_clip(solver, tf, obs, action_dim, int(cfg.eval_budget))  # [K, A]
        ep = {
            "planned": torch.tensor(planned * act_std + act_mean, dtype=torch.float32),  # [K, A]
            "gt": actions[: planned.shape[0]].to(torch.float32),  # [K, A]
            "grasp_pos": env_info["grasp_pos"] if env_info else None,
        }
        ep_dir = out_dir / f"ep_{k}"
        ep_dir.mkdir(exist_ok=True)
        torch.save(ep, ep_dir / "actions.pt")
        (ep_dir / "planning_compute.json").write_text(json.dumps(pop_solve_records(solver)))
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


if __name__ == "__main__":
    main()
