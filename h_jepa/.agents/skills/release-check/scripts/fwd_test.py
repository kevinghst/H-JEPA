"""One seeded training step (model init + hjepa_forward on a fixed random batch), saving every loss term.

Run it on two code versions and compare with `python fwd_test.py --compare a.pt b.pt`; a refactor that
preserves training should give bit-identical losses and parameter counts.

usage (from <code_root>/h_jepa, with PYTHONPATH=<code_root>:<code_root>/h_jepa):
    python <this> <out.pt>
    python <this> --compare before.pt after.pt
"""
import os
import sys

import torch

DIMS = {"ant": {"action": 8, "xy": 2, "qpos": 15, "qvel": 14, "proprio": 27},
        "fourroom": {"action": 2, "xy": 2, "distractor0_xy": 2, "min_distractor_dist": 1},
        "pusht": {"action": 2, "proprio": 4, "state": 7, "block_ori": 2},
        "cube": {"action": 5}}
CONFIGS = ["ant_hjepa_l3", "fourroom_hjepa_l4", "pusht_hwm_l2", "cube_lewm", "cube_hjepa_l2"]


def run(out_path):
    import hydra
    from omegaconf import open_dict

    import main_hjepa  # registers the eval resolver
    from hjepa_utils import create_world_model, hjepa_forward

    out = {}
    for name in CONFIGS:
        env = name.split("_")[0]
        with hydra.initialize_config_dir(version_base=None, config_dir=os.path.abspath("config/train")):
            cfg = hydra.compose(config_name=name)
        with open_dict(cfg):
            for level in range(1, cfg.num_levels + 1):
                for col, dim in DIMS[env].items():
                    setattr(cfg[f"level{level}"].wm, f"{col}_dim", dim)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model, losses, _ = create_world_model(cfg)
        model = model.cuda().train()

        class Module:
            pass

        module = Module()
        module.model = model
        module.log_dict = lambda *a, **k: None
        for key, loss in losses.items():
            setattr(module, key, loss.cuda())

        g = torch.Generator().manual_seed(1)
        T, frameskip = 40, int(cfg.data.dataset.level1.frameskip)
        batch = {
            "pixels_level1": torch.randint(0, 255, (4, T, 3, cfg.img_size, cfg.img_size), generator=g,
                                           dtype=torch.uint8).cuda(),
            "action_level1": torch.randn(4, T, frameskip * DIMS[env]["action"], generator=g).cuda(),
        }
        for col in cfg.data.dataset.keys_to_load:
            if col not in ("pixels", "action"):
                batch[f"{col}_level1"] = torch.randn(4, T, DIMS[env].get(col, 1), generator=g).cuda()
        torch.manual_seed(2)
        torch.cuda.manual_seed_all(2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = hjepa_forward(module, batch, "train", cfg,
                                   normalize_batch=main_hjepa._normalize_uint8_images_on_device)
        out[name] = {k: v.detach().float().cpu() for k, v in output.items() if "loss" in k and torch.is_tensor(v)}
        out[name]["n_params"] = torch.tensor(sum(p.numel() for p in model.parameters()))
        print(name, {k: round(float(v), 6) for k, v in out[name].items() if k in ("loss", "n_params")}, flush=True)
    torch.save(out, out_path)


def compare(a_path, b_path):
    a, b = torch.load(a_path), torch.load(b_path)
    for name in a:
        keys = sorted(set(a[name]) | set(b.get(name, {})))
        differ = [k for k in keys if k not in a[name] or k not in b[name] or not torch.equal(a[name][k], b[name][k])]
        print(name, f"{len(keys)} terms,", "IDENTICAL" if not differ else f"differ: {differ}")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1])
