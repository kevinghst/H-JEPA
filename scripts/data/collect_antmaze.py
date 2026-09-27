import os
import time
from pathlib import Path

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import h5py
import numpy as np
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.data.utils import get_cache_dir
from stable_worldmodel.envs.ogbench import AntMazeExplorePolicy


def _dataset_path(cfg: DictConfig) -> Path:
    return Path(cfg.cache_dir or get_cache_dir()) / f'{cfg.output_dataset_name}.h5'


def _num_recorded_episodes(dataset_path: Path) -> int:
    if not dataset_path.exists():
        return 0

    with h5py.File(dataset_path, 'r') as f:
        if 'ep_len' not in f:
            return 0
        return int(f['ep_len'].shape[0])


def _merged_dict(base_cfg, *overrides) -> dict:
    base = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    merged = OmegaConf.merge(base, *overrides)
    return OmegaConf.to_container(merged, resolve=True)


def _record_antmaze_dataset(
    cfg: DictConfig,
    *,
    dataset_type: str,
    num_traj: int,
    seed: int,
    target_episodes: int,
    world_overrides=None,
    policy_overrides=None,
    options=None,
) -> None:
    world_cfg = _merged_dict(
        cfg.world,
        {'dataset_type': dataset_type},
        world_overrides or {},
    )
    policy_cfg = _merged_dict(
        cfg.policy,
        {'dataset_type': dataset_type},
        policy_overrides or {},
    )

    world = swm.World(
        'swm/OGBAntMaze-v0',
        **world_cfg,
        terminate_at_goal=False,
    )

    rng = np.random.default_rng(seed)
    world.set_policy(AntMazeExplorePolicy(**policy_cfg))
    logging.info(
        f'Collecting {num_traj} {dataset_type} trajectories into '
        f'{cfg.output_dataset_name} up to {target_episodes} total episodes.'
    )
    world.record_dataset(
        cfg.output_dataset_name,
        episodes=target_episodes,
        seed=rng.integers(0, 1_000_000).item(),
        cache_dir=cfg.cache_dir,
        options=options,
    )


@hydra.main(version_base=None, config_path='./config', config_name='visual_antmaze_medium_explore_train')
def run(cfg: DictConfig):
    """Collect visual AntMaze data with the OGBench policy."""
    start_time = time.time()
    dataset_path = _dataset_path(cfg)

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None

    mixtures = cfg.get('mixtures')
    if mixtures:
        target_episodes = 0
        for mixture in mixtures:
            num_traj = int(mixture.num_traj)
            target_episodes += num_traj
            recorded_episodes = _num_recorded_episodes(dataset_path)
            if recorded_episodes >= target_episodes:
                logging.info(
                    f'Skipping {mixture.dataset_type}: {recorded_episodes} '
                    f'episodes already recorded, target is {target_episodes}.'
                )
                continue

            mix_options = mixture.get('options', options)
            if OmegaConf.is_config(mix_options):
                mix_options = OmegaConf.to_object(mix_options)
            _record_antmaze_dataset(
                cfg,
                dataset_type=str(mixture.dataset_type),
                num_traj=num_traj,
                seed=int(mixture.get('seed', cfg.seed)),
                target_episodes=target_episodes,
                world_overrides=mixture.get('world', {}),
                policy_overrides=mixture.get('policy', {}),
                options=mix_options,
            )
    else:
        _record_antmaze_dataset(
            cfg,
            dataset_type=str(cfg.dataset_type),
            num_traj=int(cfg.num_traj),
            seed=int(cfg.seed),
            target_episodes=int(cfg.num_traj),
            options=options,
        )

    config_path = dataset_path.with_suffix('.yaml')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, config_path, resolve=True)

    elapsed_seconds = time.time() - start_time
    timing_path = dataset_path.with_suffix('.txt')
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(
        f'dataset_path: {dataset_path}\n'
        f'elapsed_seconds: {elapsed_seconds:.3f}\n'
        f'elapsed_minutes: {elapsed_seconds / 60:.3f}\n'
        f'elapsed_hours: {elapsed_seconds / 3600:.3f}\n',
        encoding='utf-8',
    )

    logging.success(
        f'Completed {cfg.get("dataset_type", "mixed")} data collection for visual AntMaze in '
        f'{elapsed_seconds / 60:.2f} minutes'
    )


if __name__ == '__main__':
    run()
