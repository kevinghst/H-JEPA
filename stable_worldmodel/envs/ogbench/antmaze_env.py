"""Local SWM wrapper for OGBench visual AntMaze environments."""

from stable_worldmodel.envs.ogbench.locomaze_env import LocomazeEnv


class AntMazeEnv(LocomazeEnv):
    """AntMaze wrapper that preserves OGBench locomaze behavior."""

    def __init__(
        self,
        maze_type='large',
        dataset_type=None,
        stitch_distance=4,
        ob_type='pixels',
        width=224,
        height=224,
        camera_name='back',
        render_mode='rgb_array',
        **kwargs,
    ):
        super().__init__(
            loco_env_type='ant',
            env_name='AntMaze',
            maze_type=maze_type,
            dataset_type=dataset_type,
            stitch_distance=stitch_distance,
            ob_type=ob_type,
            width=width,
            height=height,
            camera_name=camera_name,
            render_mode=render_mode,
            goal_tol=1.0,
            **kwargs,
        )
