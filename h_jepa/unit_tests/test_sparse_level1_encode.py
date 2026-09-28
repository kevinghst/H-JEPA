"""Dense vs sparse level-1 encoding must give identical per-level outputs.

    cd h_jepa && PYTHONPATH=. python unit_tests/test_sparse_level1_encode.py
"""
import itertools
from pathlib import Path
import hydra
import torch
from omegaconf import OmegaConf, open_dict

import main_hjepa  # noqa: F401  (registers the eval resolver)
from hjepa_utils import create_world_model

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "config" / "train")
DIMS = {"action": 8, "xy": 2, "qpos": 15, "qvel": 14, "proprio": 27}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(num_levels, stride, window):
    with hydra.initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = hydra.compose(config_name="ant_hjepa_l4")
    with open_dict(cfg):
        cfg.num_levels = num_levels
        for level in range(1, num_levels + 1):
            for col, dim in DIMS.items():
                setattr(cfg[f"level{level}"].wm, f"{col}_dim", dim)
        for level in range(2, num_levels + 1):
            level_cfg = cfg[f"level{level}"]
            level_cfg.stride = stride
            level_cfg.window_size = window
            if window > 1:
                level_cfg.encoder = OmegaConf.create({
                    "type": "seq_encoder", "output_dim": 384, "input_dim": 384,
                    "max_chunk": window,
                    "d_model": 128, "nhead": 4, "num_layers": 1,
                    "projector": {"type": "identity"},
                })
    model, _, _ = create_world_model(cfg)
    return model.to(DEVICE).eval()


def level1_span(model):
    frames = model.get_level(model.num_levels).target_length
    for level in range(model.num_levels, 1, -1):
        stride, window = model._level_geometry(level)
        frames = (frames - 1) * stride + window
    return frames


def make_batch(span, batch_size=2):
    g = torch.Generator().manual_seed(0)
    return {
        "pixels_level1": torch.randn(batch_size, span, 3, 64, 64, generator=g).to(DEVICE),
        "proprio_level1": torch.randn(batch_size, span, 27, generator=g).to(DEVICE),
        "action_level1": torch.randn(batch_size, span, 40, generator=g).to(DEVICE),
        "xy_level1": torch.randn(batch_size, span, 2, generator=g).to(DEVICE),
    }


def run(model, batch, sparse, frames_seen):
    encode = model.jepas[0].encode

    def counting_encode(info, key="pixels", **kwargs):
        frames_seen.append(info[key].size(1))
        return encode(info, key=key, **kwargs)

    model.jepas[0].encode = counting_encode
    torch.manual_seed(123)
    try:
        with torch.no_grad():
            return model.encode_hierarchical(dict(batch), sparse_level1_encode=sparse)
    finally:
        model.jepas[0].encode = encode


def main():
    for num_levels, stride, window in itertools.product((2, 3, 4), (1, 2, 3), (1, 2, 3)):
        model = build_model(num_levels, stride, window)
        span = level1_span(model)
        batch = make_batch(span)
        dense_frames, sparse_frames = [], []
        dense = run(model, batch, sparse=False, frames_seen=dense_frames)
        sparse = run(model, batch, sparse=True, frames_seen=sparse_frames)

        keys = [k for k in dense if torch.is_tensor(dense[k]) and (k.startswith("embed_") or k.startswith("action_") or "_level" in k)]
        for k in keys:
            assert k in sparse, f"{k} missing from sparse output"
            assert dense[k].shape == sparse[k].shape, (k, dense[k].shape, sparse[k].shape)
            assert torch.allclose(dense[k], sparse[k], atol=1e-5, rtol=1e-5), f"mismatch in {k} (levels={num_levels}, stride={stride}, window={window})"
        print(f"levels={num_levels} stride={stride} window={window}: span={span:3d} "
              f"ViT frames dense={dense_frames[0]:3d} sparse={sparse_frames[0]:3d}  ok ({len(keys)} keys)")
    print("PASS")


if __name__ == "__main__":
    main()
