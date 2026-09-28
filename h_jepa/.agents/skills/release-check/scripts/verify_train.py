"""Compose every release training config and diff it against the saved config.yaml of its paper run.

usage: python verify_train.py [--seeds 42 43 44]
Keys removed on purpose during the cleanup (REMOVED) and run-specific keys (IGNORED) are skipped;
anything printed is a real difference. The four LeWM configs have known, behavior-neutral
differences listed in ../SKILL.md.
"""
import argparse
import re
import sys
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from cdiff import flat
from runs import R, ROOT

CONFIG_DIR = str(Path(__file__).resolve().parents[4] / "config" / "train")
IGNORED = ("sweep", "subdir", "output_model_name", "wandb", "planning_eval", "final_probing_decoding_eval",
           "transition_budget", "train_batch_budget")
REMOVED = [r"level\d+\.(freeze|freeze_encoder|train|detach_lower_level_inputs|debug\..*)$",
           r"level\d+\.predictor\.ensemble_size$", r"level\d+\.loss\.embed\.temp_straight\..*",
           r"level\d+\.wm\.type$", r"level\d+\.(encoder\.|encoder\.pixel_encoder\.encoder\.)resnet9$",
           r"data\.dataset\.(random_waypoints|augment_static_window_prob|precompute_levels)$",
           r"data\.dataset\.level\d+\.(load|sample_range_low|sample_range_high)$",
           r"(encoder_resnet9|projector_loss_weight|train_value_function)$", r"optimizer\.decoder\..*",
           r"(local_cache_dir|train_split)$", r"level\d+\.encoder\.residual$", r"level\d+\.action_pooler\.uniform_input$",
           r"level\d+\.wm\.(xy|qpos|qvel|observation|state|block_pos|block_ori|block_xy|agent_xy|agent_vel|effector_pos|"
           r"effector_yaw|gripper|arm_joint_pos|arm_joint_vel|gripper_vel|distractor0_xy|min_distractor_dist)_dim$",
           r"level([2-9])\.wm\.action_dim$", r"level\d+\.wm\.proprio_dim$"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        for env in R:
            for model, path in R[env].items():
                for seed in args.seeds:
                    cfg = compose(config_name=f"{env}_{model}", overrides=[f"seed={seed}"])
                    A = flat(OmegaConf.to_container(cfg, resolve=False))
                    B = flat(yaml.safe_load(open(ROOT + path.format(s=seed) + "/config.yaml")))
                    diffs = [(k, A.get(k, "<none>"), B.get(k, "<none>")) for k in sorted(set(A) | set(B))
                             if not k.startswith(IGNORED) and not any(re.fullmatch(p, k) for p in REMOVED)
                             and str(A.get(k, "<none>")) != str(B.get(k, "<none>"))]
                    print(f"{env}_{model} seed{seed}: {'IDENTICAL' if not diffs else ''}")
                    for d in diffs:
                        print("   ", d)


if __name__ == "__main__":
    main()
