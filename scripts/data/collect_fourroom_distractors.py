from pathlib import Path

import h5py
import hydra
import numpy as np
from loguru import logger as logging
from omegaconf import OmegaConf

import stable_worldmodel as swm
from fourroom_distractor_distance import backfill_h5
from stable_worldmodel.envs.four_room import EpisodicPolicyMixture


def _is_set(value) -> bool:
    return value is not None


def _dataset_counts(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0

    with h5py.File(path, 'r') as f:
        if 'ep_len' not in f:
            return 0, 0
        ep_lens = f['ep_len'][:]

    num_episodes = int(len(ep_lens))
    num_transitions = int(np.maximum(ep_lens.astype(np.int64) - 1, 0).sum())
    return num_episodes, num_transitions


def _record_dataset(cfg, world, seed: int | None, options) -> None:
    cache_dir = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    h5_path = cache_dir / f'{cfg.dataset_name}.h5'

    if cfg.num_traj is not None:
        world.record_dataset(
            cfg.dataset_name,
            episodes=int(cfg.num_traj),
            seed=seed,
            cache_dir=cfg.cache_dir,
            options=options,
        )
        return

    target_transitions = int(cfg.num_transitions)
    max_episode_steps = int(cfg.world.max_episode_steps)
    if target_transitions <= 0:
        raise ValueError('num_transitions must be positive.')
    if max_episode_steps <= 0:
        raise ValueError('world.max_episode_steps must be positive.')

    num_episodes, num_transitions = _dataset_counts(h5_path)
    while num_transitions < target_transitions:
        remaining = target_transitions - num_transitions
        next_episodes = int(np.ceil(remaining / max_episode_steps))
        if not cfg.get('collection', {}).get('fixed_length_episodes', False):
            next_episodes = max(next_episodes, int(cfg.world.num_envs), 1)
        next_episodes = max(next_episodes, 1)
        target_episodes = num_episodes + next_episodes

        logging.info(
            f'Collecting to {target_transitions} transitions: currently '
            f'{num_transitions} transitions across {num_episodes} episodes; '
            f'next target is {target_episodes} episodes.'
        )
        world.record_dataset(
            cfg.dataset_name,
            episodes=target_episodes,
            seed=seed,
            cache_dir=cfg.cache_dir,
            options=options,
        )
        seed = None if seed is None else seed + next_episodes
        num_episodes, num_transitions = _dataset_counts(h5_path)

    logging.success(
        f'Reached {num_transitions} transitions across {num_episodes} '
        f'episodes for {cfg.dataset_name}'
    )


@hydra.main(
    version_base=None,
    config_path='./config',
    config_name='fourroom_tp35_d1',
)
def run(cfg):
    """Collect four-room trajectories with env-owned visual distractors."""
    assert _is_set(cfg.num_traj) ^ _is_set(cfg.num_transitions), (
        'Exactly one of num_traj and num_transitions must be set; the other '
        'must be null.'
    )

    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    collection_cfg = cfg.get('collection', {})
    if collection_cfg.get('fixed_length_episodes', False):
        world_cfg['terminate_on_goal'] = False
        world_cfg['retarget_on_goal'] = bool(
            collection_cfg.get('retarget_on_goal', True)
        )
    env_name = world_cfg.pop('env_name', 'swm/FourRoomDistractors-v0')
    world = swm.World(env_name, **world_cfg, render_mode='rgb_array')

    rng = np.random.default_rng(cfg.seed)
    total_episodes = None if cfg.num_traj is None else int(cfg.num_traj)
    policy = EpisodicPolicyMixture.from_config(
        cfg.ego_policy.mixture,
        total_episodes=total_episodes,
        seed=rng.integers(0, 1_000_000).item(),
    )
    world.set_policy(policy)

    options = cfg.get('options')
    _record_dataset(
        cfg,
        world,
        seed=rng.integers(0, 1_000_000).item(),
        options=options,
    )

    cache_dir = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    config_path = cache_dir / f'{cfg.dataset_name}.collection.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, config_path)

    h5_path = cache_dir / f'{cfg.dataset_name}.h5'
    backfill_h5(h5_path, world_cfg)

    logging.success(f'Completed data collection for {cfg.dataset_name}')
    logging.success(f'Collection config saved to {config_path}')


if __name__ == '__main__':
    run()
