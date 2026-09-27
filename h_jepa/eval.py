import os

os.environ['MUJOCO_GL'] = 'egl'

import hydra
from omegaconf import DictConfig
import wandb
from planning_eval import run_planning_eval


@hydra.main(version_base=None, config_path='config/eval', config_name='pusht')
def run(cfg: DictConfig):
    run_name = cfg.wandb.get('run_name', None)

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=dict(cfg),
            name=run_name,
        )

    metrics = run_planning_eval(cfg)

    if cfg.wandb.enabled:
        wandb.log(metrics)
        wandb.finish()

if __name__ == '__main__':
    run()
