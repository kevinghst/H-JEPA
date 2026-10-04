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
from runs import R, ROOT, rename_datasets

CONFIG_DIR = str(Path(__file__).resolve().parents[4] / "config" / "train")
IGNORED = ("sweep", "env", "subdir", "output_model_name", "wandb", "planning_eval", "final_probing_decoding_eval",
           "transition_budget", "train_batch_budget", "save_every_n_epochs", "resume_every_n_steps")
REMOVED = [r"level\d+\.(freeze|freeze_encoder|train|detach_lower_level_inputs|debug\..*)$",
           r"level\d+\.predictor\.ensemble_size$", r"level\d+\.loss\.embed\.temp_straight\..*",
           r"level\d+\.wm\.type$", r"level\d+\.(encoder\.|encoder\.pixel_encoder\.encoder\.)resnet9$",
           r"data\.dataset\.(random_waypoints|augment_static_window_prob|precompute_levels)$",
           r"data\.dataset\.(sources|mix_mode|subset_unit)$",
           r"data\.dataset\.level\d+\.(load|sample_range_low|sample_range_high)$",
           r"(encoder_resnet9|projector_loss_weight|train_value_function)$", r"optimizer\.decoder\..*",
           r"(local_cache_dir|train_split)$", r"level\d+\.encoder\.residual$", r"level\d+\.action_encoder\.uniform_input$",
           r"level\d+\.wm\.(xy|qpos|qvel|observation|state|block_pos|block_ori|block_xy|agent_xy|agent_vel|effector_pos|"
           r"effector_yaw|gripper|arm_joint_pos|arm_joint_vel|gripper_vel|distractor0_xy|min_distractor_dist)_dim$",
           r"level([2-9])\.wm\.action_dim$", r"level\d+\.wm\.proprio_dim$", r"max_train_batches_total$"]


def rename_action_keys(cfg):
    # The paper runs predate the action_encoder->action_embed, action_pooler->action_encoder rename
    # (queue_size moved from the old action_encoder to the new one).
    out = {}
    for k, v in cfg.items():
        k = re.sub(r"\.action_encoder\.(?!queue_size$)", ".action_embed.", k).replace(".action_pooler.", ".action_encoder.")
        out[k] = v.replace(".action_pooler.", ".action_encoder.") if isinstance(v, str) else v
    return out


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
                    B = rename_datasets(rename_action_keys(flat(yaml.safe_load(open(ROOT + path.format(s=seed) + "/config.yaml")))))
                    if "data.dataset.sources" in B:
                        # Ant: the mix's transition budget is now the whole pre-built train file
                        del B["data.dataset.total_transitions"]
                    diffs = [(k, A.get(k, "<none>"), B.get(k, "<none>")) for k in sorted(set(A) | set(B))
                             if not k.startswith(IGNORED) and not any(re.fullmatch(p, k) for p in REMOVED)
                             and str(A.get(k, "<none>")) != str(B.get(k, "<none>"))]
                    print(f"{env}_{model} seed{seed}: {'IDENTICAL' if not diffs else ''}")
                    for d in diffs:
                        print("   ", d)


if __name__ == "__main__":
    main()
