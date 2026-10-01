"""Split LeWM's cube_single_expert.h5 into the train (first 9,800 episodes) and val (last 200) files.

usage: python scripts/data/split_cube.py   (reads and writes $STABLEWM_HOME)
Each file renumbers `ep_idx` and the per-row `id` from its own first row.
"""
import os
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

HOME = Path(os.environ["STABLEWM_HOME"])
SRC = HOME / "cube_single_expert.h5"
SPLITS = {"cube_single_expert_train": (0, 9800), "cube_single_expert_val": (9800, 10000)}
STEP = 3300

with h5py.File(SRC, "r") as src:
    ep_len, ep_offset = src["ep_len"][:], src["ep_offset"][:]
    for name, (e0, e1) in SPLITS.items():
        r0, r1 = int(ep_offset[e0]), int(ep_offset[e1 - 1] + ep_len[e1 - 1])
        with h5py.File(HOME / f"{name}.h5", "w-") as dst:
            dst["ep_len"] = ep_len[e0:e1]
            dst["ep_offset"] = ep_offset[e0:e1] - r0
            for k, d in src.items():
                if k in ("ep_len", "ep_offset"):
                    continue
                kw = {}
                if d.ndim == 4:
                    kw["compression"] = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
                out = dst.create_dataset(k, shape=(r1 - r0, *d.shape[1:]), dtype=d.dtype, chunks=d.chunks, **kw)
                for i in range(r0, r1, STEP):
                    block = d[i:min(i + STEP, r1)]
                    if k == "ep_idx":
                        block = block - e0
                    elif k == "id":
                        block = block - r0
                    out[i - r0:i - r0 + len(block)] = block
        print(f"wrote {name}.h5: {e1 - e0} episodes, {r1 - r0} steps")
