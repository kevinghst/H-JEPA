"""Diff every release planning config against the saved eval_config.yaml of the paper eval it came from.

usage: python verify_eval.py
Run-specific keys and keys removed during the cleanup are skipped; anything printed is a real difference.
"""
import glob
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from cdiff import flat
from runs import ROOT, rename_datasets

EVAL_DIR = Path(__file__).resolve().parents[4] / "config" / "eval"
DEPTH = {  # budget-100 selections from the compute_depth sweeps
    "ant": {"flat": "9-13-1/2", "l2": "9-13-2/3", "l3": "9-13-6/7", "l4": "9-19-1/21"},
    "fourroom": {"flat": "9-13-1/3", "l2": "9-13-5/5", "l3": "9-13-3/22", "l4": "9-19-1/58"},
    "cube": {"flat": "9-13-1/5", "l2": "9-13-2/26", "l3": "9-13-3/5", "l4": "9-19-1/19"},
    "pusht": {"flat": "9-13-1/5", "l2": "9-13-2/16", "l3": "9-13-3/4"},
}
DIRS = {"ant": "ant", "fourroom": "fourroom_distractors", "cube": "ogb", "pusht": "pusht"}
IGNORED = ("hydra", "sweep", "wandb", "policy", "seed", "output", "cache_dir", "n_gpus", "cpus_per_task",
           "world.num_envs", "load_eval_trajs_path", "world_config_path", "world_config_mode",
           "proprio_ood_stats_path", "defaults")
REMOVED = [r".*\.cost\..*", r".*\.warm_start$", r".*\.horizon_decay_ratio$", r".*\.uncertainty_cost_weight$",
           r"eval\.(save_video|save_plots)$", r"solver\..*\.seed$", r"solver\.seed$",
           r".*\.action_cost_space$", r"eval\.batch_eval$",
           r"eval\.expert_action_distance_(horizon|dims)$", r"world\.image_shape$",
           r"world\.max_episode_steps$", r"eval\.(pickup_height_delta|traj_sampling_mode|start_index)$",
           r".*\.resolve_horizon$"]


def source(env, planner):
    d = DIRS[env]
    if planner in DEPTH[env]:
        return ROOT + f"{d}/compute_depth/{DEPTH[env][planner]}/seed42/eval_config.yaml"
    if planner.endswith("_project"):
        return ROOT + f"{d}/planning_evals/9-16-3/{planner}/seed42/eval_config.yaml"
    return glob.glob(ROOT + "pusht/9-9-3/0/seed42/planning_eval/epoch_*/eval_config.yaml")[0]  # pusht_l4


for env in DIRS:
    for planner in ("flat", "l2", "l3", "l4", "l2_project", "l3_project", "l4_project"):
        A = flat(yaml.safe_load(open(EVAL_DIR / f"{env}_{planner}.yaml")))
        B = rename_datasets(flat(yaml.safe_load(open(source(env, planner)))))
        diffs = [(k, A.get(k, "<none>"), B.get(k, "<none>")) for k in sorted(set(A) | set(B))
                 if not k.startswith(IGNORED) and not any(re.fullmatch(p, k) for p in REMOVED)
                 and str(A.get(k, "<none>")) != str(B.get(k, "<none>"))]
        print(f"{env}_{planner}: {'IDENTICAL' if not diffs else ''}")
        for d in diffs:
            print("   ", d)
