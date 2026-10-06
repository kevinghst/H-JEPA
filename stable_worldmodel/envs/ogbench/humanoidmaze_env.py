"""Local SWM wrapper for OGBench visual HumanoidMaze environments."""

from stable_worldmodel.envs.ogbench.locomaze_env import LocomazeEnv


class HumanoidMazeEnv(LocomazeEnv):
    """HumanoidMaze wrapper for visual OGBench locomaze data and planning."""

    XY_STATE_SETTLE_STEPS = 40

    def __init__(
        self,
        maze_type='large',
        dataset_type=None,
        stitch_distance=4,
        ob_type='pixels',
        width=64,
        height=64,
        camera_name='back',
        render_mode='rgb_array',
        **kwargs,
    ):
        super().__init__(
            loco_env_type='humanoid',
            env_name='HumanoidMaze',
            maze_type=maze_type,
            dataset_type=dataset_type,
            stitch_distance=stitch_distance,
            ob_type=ob_type,
            width=width,
            height=height,
            camera_name=camera_name,
            render_mode=render_mode,
            **kwargs,
        )
