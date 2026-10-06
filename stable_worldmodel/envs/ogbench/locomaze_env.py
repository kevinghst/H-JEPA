"""Shared OGBench visual locomaze wrapper."""

import numpy as np

from stable_worldmodel.envs.ogbench.pointmaze_env import PointMazeEnv


class LocomazeEnv(PointMazeEnv):
    """Common wrapper for OGBench visual locomaze tasks.

    OGBench provides the MuJoCo locomotion, maze layouts, reset logic, goal
    images, and oracle subgoals. This class adds the SWM-specific pieces used
    by data collection and planning eval: configurable task sampling, goal
    updates, proprio/qpos/qvel info, and action-aligned direction logging.
    """

    XY_STATE_SETTLE_STEPS = 5

    def __init__(
        self,
        loco_env_type,
        env_name,
        maze_type='large',
        dataset_type=None,
        stitch_distance=4,
        ob_type='pixels',
        width=64,
        height=64,
        camera_name='back',
        render_mode='rgb_array',
        goal_tol=None,
        **kwargs,
    ):
        self._current_direction = np.full(2, np.nan, dtype=np.float32)
        self._current_goal = None
        self._current_goal_rendered = None
        self._current_goal_info = {}
        self._current_init_ij = None
        self._current_goal_ij = None
        self._task_rng = np.random.default_rng()
        self._dataset_type = dataset_type
        self._stitch_distance = int(stitch_distance)
        super().__init__(
            loco_env_type=loco_env_type,
            maze_env_type='maze',
            maze_type=maze_type,
            ob_type=ob_type,
            width=width,
            height=height,
            camera_name=camera_name,
            render_mode=render_mode,
            drop_goal=False,
            **kwargs,
        )
        if goal_tol is not None:
            self.env._goal_tol = float(goal_tol)
        self.env_name = env_name
        self.all_cells, self.vertex_cells = self._get_cells()
        # The installed OGBench package does not expose these attributes, but
        # OGBench's data-generation code relies on them.
        self.env.all_cells = self.all_cells
        self.env.vertex_cells = self.vertex_cells

    def set_current_direction(self, direction):
        self._current_direction = np.asarray(direction, dtype=np.float32).copy()

    def sample_goal_ij(self):
        idx = self._task_rng.integers(len(self.vertex_cells))
        return tuple(self.vertex_cells[idx])

    def set_navigation_goal(self, goal_ij=None, goal_xy=None):
        self.env.set_goal(goal_ij=goal_ij, goal_xy=goal_xy)
        self._current_goal_ij = tuple(goal_ij) if goal_ij is not None else None
        self._current_goal = self._render_goal_observation()
        self._current_goal_info = {'goal': self._current_goal}

    def set_current_goal_info(
        self,
        goal=None,
        goal_xy=None,
        goal_qpos=None,
        goal_qvel=None,
        goal_proprio=None,
    ):
        goal_info = {}
        if goal is not None:
            self._current_goal = goal
            goal_info['goal'] = goal
        if goal_xy is not None:
            goal_info['goal_xy'] = goal_xy
        if goal_qpos is not None:
            goal_info['goal_qpos'] = goal_qpos
        if goal_qvel is not None:
            goal_info['goal_qvel'] = goal_qvel
        if goal_proprio is not None:
            goal_info['goal_proprio'] = goal_proprio
        self._current_goal_info = goal_info

    def neutral_state_from_xy(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        if xy.shape != (2,):
            raise ValueError(
                f'{self.env_name} xy must have shape (2,), got {xy.shape}.'
            )

        qpos = self.env.init_qpos.copy()
        qvel = self.env.init_qvel.copy()
        qpos[:2] = xy
        return qpos, qvel

    def neutral_info_from_xy(self, xy, render_pixels=False):
        qpos, qvel = self.neutral_state_from_xy(xy)
        info = {
            'xy': qpos[:2].copy(),
            'qpos': qpos.copy(),
            'qvel': qvel.copy(),
            'proprio': np.concatenate([qpos, qvel]).astype(np.float32),
        }

        if render_pixels:
            current_qpos = self.env.data.qpos.copy()
            current_qvel = self.env.data.qvel.copy()
            self.env.set_state(qpos, qvel)
            info['pixels'] = self.env.get_ob()
            self.env.set_state(current_qpos, current_qvel)

        return info

    def settled_info_from_xy(
        self,
        xy,
        render_pixels=False,
        settle_steps=None,
    ):
        qpos, qvel = self.neutral_state_from_xy(xy)
        current_qpos = self.env.data.qpos.copy()
        current_qvel = self.env.data.qvel.copy()
        settle_steps = (
            self.XY_STATE_SETTLE_STEPS if settle_steps is None else settle_steps
        )

        try:
            self.env.set_state(qpos, qvel)
            self.env.set_xy(np.asarray(xy, dtype=np.float64))
            for _ in range(int(settle_steps)):
                self.env.step(self.action_space.sample())

            settled_qpos = self.env.data.qpos.copy()
            settled_qvel = self.env.data.qvel.copy()
            info = {
                'xy': self.env.get_xy().copy(),
                'qpos': settled_qpos,
                'qvel': settled_qvel,
                'proprio': np.concatenate([settled_qpos, settled_qvel]).astype(
                    np.float32
                ),
            }

            if render_pixels:
                info['pixels'] = self.env.get_ob()

            return info
        finally:
            self.env.set_state(current_qpos, current_qvel)

    def _get_cells(self):
        all_cells = []
        vertex_cells = []
        maze_map = np.asarray(self.env.maze_map)

        for i in range(maze_map.shape[0]):
            for j in range(maze_map.shape[1]):
                if maze_map[i, j] != 0:
                    continue

                all_cells.append((i, j))

                if (
                    maze_map[i - 1, j] == 0
                    and maze_map[i + 1, j] == 0
                    and maze_map[i, j - 1] == 1
                    and maze_map[i, j + 1] == 1
                ):
                    continue
                if (
                    maze_map[i, j - 1] == 0
                    and maze_map[i, j + 1] == 0
                    and maze_map[i - 1, j] == 1
                    and maze_map[i + 1, j] == 1
                ):
                    continue

                vertex_cells.append((i, j))

        return all_cells, vertex_cells

    def _sample_stitch_goal_ij(self, init_ij):
        maze_map = np.asarray(self.env.maze_map)
        bfs_map = np.full_like(maze_map, -1)
        bfs_map[init_ij[0], init_ij[1]] = 0
        queue = [init_ij]
        adj_cells = []

        while queue:
            i, j = queue.pop(0)
            for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                ni, nj = i + di, j + dj
                if (
                    0 <= ni < bfs_map.shape[0]
                    and 0 <= nj < bfs_map.shape[1]
                    and maze_map[ni, nj] == 0
                    and bfs_map[ni, nj] == -1
                ):
                    bfs_map[ni, nj] = bfs_map[i, j] + 1
                    queue.append((ni, nj))
                    if bfs_map[ni, nj] == self._stitch_distance:
                        adj_cells.append((ni, nj))

        if not adj_cells:
            return init_ij
        idx = self._task_rng.integers(len(adj_cells))
        return tuple(adj_cells[idx])

    def _sample_task_info(self, dataset_type):
        init_idx = self._task_rng.integers(len(self.all_cells))
        init_ij = tuple(self.all_cells[init_idx])

        if dataset_type == 'stitch':
            goal_ij = self._sample_stitch_goal_ij(init_ij)
        else:
            goal_ij = self.sample_goal_ij()

        return {'init_ij': init_ij, 'goal_ij': goal_ij}

    def _prepare_reset_options(self, seed, options):
        options = dict(options or {})
        if seed is not None:
            self._task_rng = np.random.default_rng(seed)

        dataset_type = options.pop('dataset_type', self._dataset_type)
        if dataset_type is None or 'task_info' in options or 'task_id' in options:
            return options

        if dataset_type not in {'explore', 'navigate', 'stitch'}:
            raise ValueError(
                f'Unsupported {self.env_name} dataset_type: {dataset_type}'
            )

        task_info = self._sample_task_info(dataset_type)
        self._current_init_ij = tuple(task_info['init_ij'])
        self._current_goal_ij = tuple(task_info['goal_ij'])
        options['task_info'] = task_info
        return options

    def _render_goal_observation(self):
        qpos = self.env.data.qpos.copy()
        qvel = self.env.data.qvel.copy()
        self.env.set_xy(np.asarray(self.env.cur_goal_xy))
        goal = self.env.get_ob()
        self.env.set_state(qpos, qvel)
        return goal

    def _add_task_info(self, info):
        if self._current_init_ij is not None:
            info['init_ij'] = np.asarray(self._current_init_ij, dtype=np.int32)
            info['init_xy'] = np.asarray(
                self.env.ij_to_xy(self._current_init_ij), dtype=np.float32
            )
        if self._current_goal_ij is not None:
            info['goal_ij'] = np.asarray(self._current_goal_ij, dtype=np.int32)
        else:
            info['goal_ij'] = np.full(2, -1, dtype=np.int32)
        info['goal_xy'] = np.asarray(self.env.cur_goal_xy, dtype=np.float32)

    def _add_proprio_info(self, info):
        qpos = self.env.data.qpos.copy()
        qvel = self.env.data.qvel.copy()
        info.setdefault('xy', self.env.get_xy().copy())
        info['qpos'] = qpos
        info['qvel'] = qvel
        info['proprio'] = np.concatenate([qpos, qvel]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self._current_direction = np.full(2, np.nan, dtype=np.float32)
        options = self._prepare_reset_options(seed, options)
        obs, info = super().reset(seed=seed, options=options)
        self._current_goal = info.get('goal')
        self._current_goal_rendered = info.get('goal_rendered')
        self._current_goal_info = {}
        if self._current_goal_ij is None and hasattr(self.env, 'cur_task_info'):
            task_info = self.env.cur_task_info or {}
            goal_ij = task_info.get('goal_ij')
            init_ij = task_info.get('init_ij')
            self._current_goal_ij = tuple(goal_ij) if goal_ij is not None else None
            self._current_init_ij = tuple(init_ij) if init_ij is not None else None
        self._add_proprio_info(info)
        info.setdefault('prev_qpos', info['qpos'])
        info.setdefault('prev_qvel', info['qvel'])
        info.setdefault('success', 0.0)
        self._add_task_info(info)
        info.update(self._current_goal_info)
        info['direction'] = self._current_direction.copy()
        info['env_name'] = self.env_name
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self._current_goal is not None:
            info['goal'] = self._current_goal
        if self._current_goal_rendered is not None:
            info['goal_rendered'] = self._current_goal_rendered
        self._add_proprio_info(info)
        self._add_task_info(info)
        info.update(self._current_goal_info)
        info['direction'] = self._current_direction.copy()
        info['env_name'] = self.env_name
        return obs, reward, terminated, truncated, info
