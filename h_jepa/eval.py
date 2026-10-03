import os

os.environ['MUJOCO_GL'] = 'egl'

import hydra
from omegaconf import DictConfig

from droid_eval import run_clip_eval
from planning_eval import run_planning_eval


@hydra.main(version_base=None, config_path='config/eval', config_name='pusht')
def run(cfg: DictConfig):
    if "clips" in cfg:  # DROID offline clip eval
        run_clip_eval(cfg)
    else:
        run_planning_eval(cfg)

if __name__ == '__main__':
    run()
