"""Append block_ori = [cos, sin] of the T-block angle (state[:, 4]) to pusht_expert_*.h5.

state[:, 4] wraps at 0/2pi, so an MSE probe of it has to fit a target that jumps by the
full range between visually identical poses. cos/sin carries the same information as a
continuous function of the pose. Idempotent: skips files that already have the column.
"""

from pathlib import Path

import h5py
import numpy as np


def add_block_ori(h5_path):
    h5_path = Path(h5_path)
    with h5py.File(h5_path, 'r+') as f:
        if 'block_ori' in f:
            print(f'{h5_path.name}: block_ori already present, skipping')
            return
        theta = f['state'][:, 4].astype(np.float64)
        ori = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
        f.create_dataset('block_ori', data=ori, chunks=(min(1000, len(ori)), 2))

    with h5py.File(h5_path, 'r') as f:
        theta = f['state'][:, 4].astype(np.float64)
        cos, sin = f['block_ori'][:, 0], f['block_ori'][:, 1]
        assert np.allclose(cos**2 + sin**2, 1.0, atol=1e-5)
        recovered = np.arctan2(sin, cos)
        circular_err = np.abs(np.mod(recovered - theta + np.pi, 2 * np.pi) - np.pi).max()
        assert circular_err < 1e-4, circular_err
    print(f'{h5_path.name}: wrote block_ori {ori.shape}, max circular error {circular_err:.2e}')


if __name__ == '__main__':
    import sys

    for path in sys.argv[1:]:
        add_block_ori(path)
