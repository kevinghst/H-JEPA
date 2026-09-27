"""Flatten a pusht_noise split (.pth + obses/*.mp4) into a single .h5 matching pusht_expert_*.h5."""

import pickle
from pathlib import Path

import h5py
import hdf5plugin
import imageio.v3 as iio
import numpy as np
import torch


def convert(src_dir, out_path):
    src_dir = Path(src_dir)
    states = torch.load(src_dir / 'states.pth', map_location='cpu', weights_only=False).numpy()
    velocities = torch.load(src_dir / 'velocities.pth', map_location='cpu', weights_only=False).numpy()
    rel_actions = torch.load(src_dir / 'rel_actions.pth', map_location='cpu', weights_only=False).numpy()
    with open(src_dir / 'seq_lengths.pkl', 'rb') as fh:
        seq_lengths = [int(x) for x in pickle.load(fh)]

    n_ep = len(seq_lengths)
    total = sum(seq_lengths)

    with h5py.File(out_path, 'w') as f:
        pixels = f.create_dataset(
            'pixels', (total, 224, 224, 3), dtype='uint8',
            chunks=(100, 224, 224, 3), **hdf5plugin.Blosc(),
        )
        state = f.create_dataset('state', (total, 7), dtype='float32', chunks=(1000, 7))
        block_ori = f.create_dataset('block_ori', (total, 2), dtype='float32', chunks=(1000, 2))
        proprio = f.create_dataset('proprio', (total, 4), dtype='float32', chunks=(1000, 4))
        action = f.create_dataset('action', (total, 2), dtype='float32', chunks=(1000, 2))
        episode_idx = f.create_dataset('episode_idx', (total,), dtype='int64', chunks=(min(1000, total),))
        step_idx = f.create_dataset('step_idx', (total,), dtype='int64', chunks=(min(1000, total),))
        ep_len = f.create_dataset('ep_len', (n_ep,), dtype='int32', chunks=(min(1024, n_ep),))
        ep_offset = f.create_dataset('ep_offset', (n_ep,), dtype='int64', chunks=(min(1024, n_ep),))

        offset = 0
        for i, L in enumerate(seq_lengths):
            frames = iio.imread(src_dir / 'obses' / f'episode_{i:03d}.mp4')
            assert frames.shape[0] == L, f'ep {i}: {frames.shape[0]} frames != seq_len {L}'
            sl = slice(offset, offset + L)
            pixels[sl] = frames
            state[sl] = np.concatenate([states[i, :L, :5], velocities[i, :L, :2]], axis=1)
            block_ori[sl] = np.stack([np.cos(states[i, :L, 4]), np.sin(states[i, :L, 4])], axis=1)
            proprio[sl] = np.concatenate([states[i, :L, :2], velocities[i, :L, :2]], axis=1)
            action[sl] = rel_actions[i, :L, :2] * 0.01
            episode_idx[sl] = i
            step_idx[sl] = np.arange(L)
            ep_len[i] = L
            ep_offset[i] = offset
            offset += L

    print(f'wrote {out_path}: {n_ep} episodes, {total} steps')


if __name__ == '__main__':
    import sys
    convert(sys.argv[1], sys.argv[2])
