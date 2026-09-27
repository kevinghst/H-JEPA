"""Continuous four-room navigation environments with visual distractors."""

from __future__ import annotations

from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.four_room.explore_policy import ExploreActionController

DEFAULT_VARIATIONS = ('agent.position', 'target.position')
DOOR_ORDER = ('top', 'bottom', 'left', 'right')
ROOMS = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
ROOM_INDEX = {name: idx for idx, name in enumerate(ROOMS)}
DOOR_EDGES = {
    'top': ('top_left', 'top_right'),
    'bottom': ('bottom_left', 'bottom_right'),
    'left': ('top_left', 'bottom_left'),
    'right': ('top_right', 'bottom_right'),
}


class FourRoomEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    IMG_SIZE = 224
    BORDER_SIZE = 14
    DOT_STD = 7.0
    PADDING = 14
    WALL_CENTER = 112
    WALL_WIDTH_DEFAULT = 10
    MAX_DOOR = 4

    def __init__(
        self,
        render_mode: str = 'rgb_array',
        render_target: bool = False,
        init_value: dict | None = None,
        target_min_steps: int | None = None,
        agent_speed: float = 5.0,
        agent_radius: float = 7.0,
        wall_thickness: int = WALL_WIDTH_DEFAULT,
        door_size: list[int] | tuple[int, int, int, int] | None = None,
        door_positions: dict[str, int] | None = None,
        close_door_prob: float = 0.0,
        terminate_on_goal: bool = True,
        retarget_on_goal: bool = False,
    ):
        assert render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self.render_target_flag = bool(render_target)
        self.target_min_steps = target_min_steps
        self.terminate_on_goal = bool(terminate_on_goal)
        self.retarget_on_goal = bool(retarget_on_goal)
        self.agent_speed = float(agent_speed)
        self.agent_radius_init = float(agent_radius)
        self.fixed_wall_thickness = int(wall_thickness)
        self.close_door_prob = float(close_door_prob)
        if not 0.0 <= self.close_door_prob <= 1.0:
            raise ValueError('close_door_prob must be in [0, 1].')
        self.door_size_init = list(door_size or [14, 14, 14, 14])
        self.door_positions_init = {
            'top': 56,
            'bottom': 168,
            'left': 56,
            'right': 168,
            **(door_positions or {}),
        }

        y = torch.arange(self.IMG_SIZE, dtype=torch.float32)
        x = torch.arange(self.IMG_SIZE, dtype=torch.float32)
        self.grid_y, self.grid_x = torch.meshgrid(y, x, indexing='ij')

        state_dim = 2 + 2 + self.MAX_DOOR * 2
        self.observation_space = spaces.Box(
            low=0,
            high=self.IMG_SIZE,
            shape=(state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.env_name = 'FourRoom'

        self.variation_space = self._build_variation_space()
        if init_value is not None:
            self.variation_space.set_init_value(init_value)

        self.agent_position = torch.zeros(2, dtype=torch.float32)
        self.target_position = torch.zeros(2, dtype=torch.float32)
        self._target_img = None

        self.wall_thickness = self.fixed_wall_thickness
        self.wall_pos = float(self.WALL_CENTER)
        self.door_sizes = torch.zeros(self.MAX_DOOR, dtype=torch.float32)
        self.door_positions: dict[str, float] = {
            name: float(self.door_positions_init[name])
            for name in DOOR_ORDER
        }
        self.closed_door: str | None = None
        self._cache_params()

    def _build_variation_space(self):
        pos_min = float(self.BORDER_SIZE)
        pos_max = float(self.IMG_SIZE - self.BORDER_SIZE - 1)
        wall_thickness = np.array([self.fixed_wall_thickness], dtype=np.int64)

        return swm_spaces.Dict(
            {
                'agent': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array([255, 0, 0], dtype=np.uint8)
                        ),
                        'radius': swm_spaces.Box(
                            low=np.array([3.5], dtype=np.float32),
                            high=np.array([14.0], dtype=np.float32),
                            init_value=np.array(
                                [self.agent_radius_init], dtype=np.float32
                            ),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        'position': swm_spaces.Box(
                            low=np.array([pos_min, pos_min], dtype=np.float32),
                            high=np.array(
                                [pos_max, pos_max], dtype=np.float32
                            ),
                            shape=(2,),
                            dtype=np.float32,
                            init_value=np.array(
                                [56.0, 56.0], dtype=np.float32
                            ),
                            constrain_fn=self._position_is_free,
                        ),
                        'speed': swm_spaces.Box(
                            low=np.array([1.0], dtype=np.float32),
                            high=np.array([10.5], dtype=np.float32),
                            init_value=np.array(
                                [self.agent_speed], dtype=np.float32
                            ),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                    },
                    sampling_order=['color', 'radius', 'position', 'speed'],
                ),
                'target': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array([0, 255, 0], dtype=np.uint8)
                        ),
                        'radius': swm_spaces.Box(
                            low=np.array([7.0], dtype=np.float32),
                            high=np.array([14.0], dtype=np.float32),
                            init_value=np.array([7.0], dtype=np.float32),
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        'position': swm_spaces.Box(
                            low=np.array([pos_min, pos_min], dtype=np.float32),
                            high=np.array(
                                [pos_max, pos_max], dtype=np.float32
                            ),
                            shape=(2,),
                            dtype=np.float32,
                            init_value=np.array(
                                [168.0, 168.0], dtype=np.float32
                            ),
                            constrain_fn=(
                                self._constrain_target_by_min_steps
                                if self.target_min_steps is not None
                                else self._position_is_free
                            ),
                        ),
                    },
                    sampling_order=['color', 'radius', 'position'],
                ),
                'wall': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array([0, 0, 0], dtype=np.uint8)
                        ),
                        'thickness': swm_spaces.Box(
                            low=wall_thickness,
                            high=wall_thickness,
                            init_value=wall_thickness,
                            shape=(1,),
                            dtype=np.int64,
                        ),
                        'border_color': swm_spaces.RGBBox(
                            init_value=np.array([0, 0, 0], dtype=np.uint8)
                        ),
                    },
                    sampling_order=['color', 'border_color', 'thickness'],
                ),
                'door': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array(
                                [255, 255, 255], dtype=np.uint8
                            )
                        ),
                        'size': swm_spaces.MultiDiscrete(
                            nvec=[32, 32, 32, 32],
                            start=[1, 1, 1, 1],
                            init_value=self.door_size_init,
                            constrain_fn=self._check_door_sizes,
                        ),
                        'positions': swm_spaces.Dict(
                            {
                                name: swm_spaces.Discrete(
                                    self.IMG_SIZE,
                                    init_value=int(
                                        self.door_positions_init[name]
                                    ),
                                    constrain_fn=(
                                        lambda value, key=name: self._check_door_position(
                                            key, value
                                        )
                                    ),
                                )
                                for name in DOOR_ORDER
                            },
                            sampling_order=list(DOOR_ORDER),
                        ),
                    },
                    sampling_order=['color', 'size', 'positions'],
                ),
                'background': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array(
                                [255, 255, 255], dtype=np.uint8
                            )
                        )
                    }
                ),
                'rendering': swm_spaces.Dict(
                    {'render_target': swm_spaces.Discrete(2, init_value=0)}
                ),
                'task': swm_spaces.Dict(
                    {
                        'min_steps': swm_spaces.Discrete(
                            150, start=1, init_value=31
                        ),
                    }
                ),
            },
            sampling_order=[
                'background',
                'wall',
                'agent',
                'door',
                'task',
                'target',
                'rendering',
            ],
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        self.closed_door = self._sample_closed_door()

        swm_spaces.reset_variation_space(
            self.variation_space, seed, options, DEFAULT_VARIATIONS
        )

        agent_pos = options.get(
            'state', self.variation_space['agent']['position'].value
        )
        target_pos = options.get(
            'target_state', self.variation_space['target']['position'].value
        )

        self.agent_position = torch.as_tensor(agent_pos, dtype=torch.float32)
        self.target_position = torch.as_tensor(target_pos, dtype=torch.float32)
        self._cache_params()
        self._target_img = self._render_frame(agent_pos=self.target_position)

        obs = self._get_obs()
        info = self._get_info()
        info['distance_to_target'] = float(
            torch.norm(self.agent_position - self.target_position)
        )
        return obs, info

    def step(self, action):
        action_t = torch.as_tensor(action, dtype=torch.float32)
        action_t = torch.clamp(action_t, -1.0, 1.0)

        speed = float(self.variation_space['agent']['speed'].value.item())
        pos_next = self.agent_position + action_t * speed
        self.agent_position = self._apply_collisions(
            self.agent_position,
            pos_next,
        )

        dist = self._distance_to_target()
        goal_reached = dist < 16.0
        terminated = goal_reached and self.terminate_on_goal
        if goal_reached and self.retarget_on_goal and not terminated:
            self._sample_new_target()
            dist = self._distance_to_target()
        truncated = False
        reward = 0.0

        obs = self._get_obs()
        info = self._get_info()
        info['distance_to_target'] = dist
        return obs, reward, terminated, truncated, info

    def render(self):
        img_chw = (
            self._render_frame(agent_pos=self.agent_position).cpu().numpy()
        )
        return img_chw.transpose(1, 2, 0)

    def _cache_params(self) -> None:
        self.wall_thickness = int(
            self.variation_space['wall']['thickness'].value.item()
        )
        self.wall_pos = float(self.WALL_CENTER)
        self.door_sizes = torch.as_tensor(
            self.variation_space['door']['size'].value,
            dtype=torch.float32,
        )
        position_space = self.variation_space['door']['positions']
        self.door_positions = {
            name: float(position_space[name].value) for name in DOOR_ORDER
        }

    def _get_obs(self):
        door_coords = []
        for name in DOOR_ORDER:
            door_coords.extend(self._door_center(name).tolist())

        return torch.tensor(
            [
                float(self.agent_position[0]),
                float(self.agent_position[1]),
                float(self.target_position[0]),
                float(self.target_position[1]),
                *door_coords,
            ],
            dtype=torch.float32,
        )

    def _get_info(self):
        return {
            'env_name': self.env_name,
            'xy': self.agent_position.detach().cpu().numpy(),
            'proprio': self.agent_position.detach().cpu().numpy(),
            'state': self.agent_position.detach().cpu().numpy(),
            'goal_state': self.target_position.detach().cpu().numpy(),
            'closed_door_idx': self._closed_door_idx(),
        }

    def _distance_to_target(self) -> float:
        return float(torch.norm(self.agent_position - self.target_position))

    def _sample_new_target(self) -> None:
        if self.target_min_steps is not None:
            agent_value = self.agent_position.detach().cpu().numpy().astype(
                np.float32
            )
            self.variation_space['agent']['position'].set_value(agent_value)

        for _ in range(1000):
            target = self.variation_space['target']['position'].sample()
            if (
                np.linalg.norm(
                    np.asarray(target, dtype=np.float32)
                    - self.agent_position.detach().cpu().numpy()
                )
                >= 16.0
            ):
                self._set_goal_state(target)
                return

        raise RuntimeError('Failed to sample a new target outside goal radius.')

    def _render_frame(
        self,
        agent_pos: torch.Tensor,
        draw_agent: bool = True,
    ):
        H = W = self.IMG_SIZE

        bg = self.variation_space['background']['color'].value
        img = torch.empty((3, H, W), dtype=torch.uint8)
        img[0].fill_(int(bg[0]))
        img[1].fill_(int(bg[1]))
        img[2].fill_(int(bg[2]))

        wall_mask, door_mask = self._wall_and_door_masks()

        door_color = self.variation_space['door']['color'].value
        if door_mask.any():
            img[0, door_mask] = int(door_color[0])
            img[1, door_mask] = int(door_color[1])
            img[2, door_mask] = int(door_color[2])

        wall_color = self.variation_space['wall']['color'].value
        if wall_mask.any():
            img[0, wall_mask] = int(wall_color[0])
            img[1, wall_mask] = int(wall_color[1])
            img[2, wall_mask] = int(wall_color[2])

        render_target = (
            bool(self.variation_space['rendering']['render_target'].value)
            or self.render_target_flag
        )
        if render_target:
            tgt_color = self.variation_space['target']['color'].value
            tgt_r = float(
                self.variation_space['target']['radius'].value.item()
            )
            tgt_dot = self._gaussian_dot(self.target_position, tgt_r)
            img = self._alpha_blend(img, tgt_dot, tgt_color)

        if draw_agent:
            agent_color = self.variation_space['agent']['color'].value
            agent_r = float(
                self.variation_space['agent']['radius'].value.item()
            )
            agent_dot = self._gaussian_dot(agent_pos, agent_r)
            img = self._alpha_blend(img, agent_dot, agent_color)

        return img

    @staticmethod
    def _alpha_blend(
        img_u8: torch.Tensor, alpha_01: torch.Tensor, rgb_u8: np.ndarray
    ):
        a = alpha_01.clamp(0, 1).to(torch.float32)
        out = img_u8.to(torch.float32)
        for c in range(3):
            out[c] = out[c] * (1.0 - a) + float(rgb_u8[c]) * a
        return out.to(torch.uint8)

    def _gaussian_dot(self, pos_xy: torch.Tensor, radius: float):
        dx = self.grid_x - float(pos_xy[0])
        dy = self.grid_y - float(pos_xy[1])
        dist2 = dx * dx + dy * dy
        std = max(1e-6, float(radius))
        dot = torch.exp(-dist2 / (2.0 * std * std))
        m = dot.max()
        if m > 0:
            dot = dot / m
        return dot

    def _wall_and_door_masks(self):
        H = W = self.IMG_SIZE
        half = self.wall_thickness // 2

        vertical_wall = (self.grid_x >= (self.WALL_CENTER - half)) & (
            self.grid_x <= (self.WALL_CENTER + half)
        )
        horizontal_wall = (self.grid_y >= (self.WALL_CENTER - half)) & (
            self.grid_y <= (self.WALL_CENTER + half)
        )

        door_mask = torch.zeros((H, W), dtype=torch.bool)
        top = self.door_positions['top']
        bottom = self.door_positions['bottom']
        left = self.door_positions['left']
        right = self.door_positions['right']
        sizes = self.door_sizes

        if self.closed_door != 'top':
            door_mask |= vertical_wall & (
                (self.grid_y >= (top - sizes[0]))
                & (self.grid_y <= (top + sizes[0]))
            )
        if self.closed_door != 'bottom':
            door_mask |= vertical_wall & (
                (self.grid_y >= (bottom - sizes[1]))
                & (self.grid_y <= (bottom + sizes[1]))
            )
        if self.closed_door != 'left':
            door_mask |= horizontal_wall & (
                (self.grid_x >= (left - sizes[2]))
                & (self.grid_x <= (left + sizes[2]))
            )
        if self.closed_door != 'right':
            door_mask |= horizontal_wall & (
                (self.grid_x >= (right - sizes[3]))
                & (self.grid_x <= (right + sizes[3]))
            )

        wall_mask = (vertical_wall | horizontal_wall) & (~door_mask)

        bs = self.BORDER_SIZE
        t = 4
        wall_mask[:, bs - t : bs] = True
        wall_mask[:, W - bs : W - bs + t] = True
        wall_mask[bs - t : bs, :] = True
        wall_mask[H - bs : H - bs + t, :] = True

        return wall_mask, door_mask

    def _apply_collisions(self, pos1: torch.Tensor, pos2: torch.Tensor):
        bs = float(self.BORDER_SIZE)
        door_margin = 1.75
        agent_r = float(self.variation_space['agent']['radius'].value.item())

        x2, y2 = float(pos2[0]), float(pos2[1])
        x2 = min(max(x2, bs + agent_r), self.IMG_SIZE - bs - agent_r)
        y2 = min(max(y2, bs + agent_r), self.IMG_SIZE - bs - agent_r)
        pos2c = torch.tensor([x2, y2], dtype=torch.float32)

        half = self.wall_thickness // 2
        c = float(self.WALL_CENTER)
        wall_left = c - half
        wall_right = c + half
        effective_left = wall_left - agent_r
        effective_right = wall_right + agent_r

        x1 = float(pos1[0])
        x2_val = float(pos2c[0])
        y2_val = float(pos2c[1])
        if x1 < c and x2_val > effective_left:
            if not self._in_vertical_door(y2_val, door_margin):
                pos2c[0] = effective_left - 0.5
        elif x1 >= c and x2_val < effective_right:
            if not self._in_vertical_door(y2_val, door_margin):
                pos2c[0] = effective_right + 0.5

        wall_top = c - half
        wall_bottom = c + half
        effective_top = wall_top - agent_r
        effective_bottom = wall_bottom + agent_r

        y1 = float(pos1[1])
        y2_val = float(pos2c[1])
        x2_val = float(pos2c[0])
        if y1 < c and y2_val > effective_top:
            if not self._in_horizontal_door(x2_val, door_margin):
                pos2c[1] = effective_top - 0.5
        elif y1 >= c and y2_val < effective_bottom:
            if not self._in_horizontal_door(x2_val, door_margin):
                pos2c[1] = effective_bottom + 0.5

        return pos2c

    def _in_vertical_door(self, coord_y: float, margin: float) -> bool:
        top = self.door_positions['top']
        bottom = self.door_positions['bottom']
        in_top = self.closed_door != 'top' and (
            top - float(self.door_sizes[0]) - margin
            <= coord_y
            <= top + float(self.door_sizes[0]) + margin
        )
        in_bottom = self.closed_door != 'bottom' and (
            bottom - float(self.door_sizes[1]) - margin
            <= coord_y
            <= bottom + float(self.door_sizes[1]) + margin
        )
        return in_top or in_bottom

    def _in_horizontal_door(self, coord_x: float, margin: float) -> bool:
        left = self.door_positions['left']
        right = self.door_positions['right']
        in_left = self.closed_door != 'left' and (
            left - float(self.door_sizes[2]) - margin
            <= coord_x
            <= left + float(self.door_sizes[2]) + margin
        )
        in_right = self.closed_door != 'right' and (
            right - float(self.door_sizes[3]) - margin
            <= coord_x
            <= right + float(self.door_sizes[3]) + margin
        )
        return in_left or in_right

    def _position_is_free(self, pos_xy) -> bool:
        pos = np.asarray(pos_xy, dtype=np.float32)
        half = int(self.variation_space['wall']['thickness'].value.item()) // 2
        radius = float(self.variation_space['agent']['radius'].value.item())
        wall_min = self.WALL_CENTER - half - radius
        wall_max = self.WALL_CENTER + half + radius
        return not (
            wall_min <= float(pos[0]) <= wall_max
            or wall_min <= float(pos[1]) <= wall_max
        )

    def _check_door_sizes(self, sizes) -> bool:
        radius = float(self.variation_space['agent']['radius'].value.item())
        return all(float(size) >= 1.1 * radius for size in sizes)

    def _check_door_position(self, name: str, value: int) -> bool:
        sizes = self.variation_space['door']['size'].value
        if sizes is None:
            sizes = self.door_size_init
        size = float(sizes[DOOR_ORDER.index(name)])
        radius = float(self.variation_space['agent']['radius'].value.item())
        half = int(self.variation_space['wall']['thickness'].value.item()) // 2
        clearance = max(size, radius) + half + 1.0
        low_outer = self.BORDER_SIZE + clearance
        high_outer = self.IMG_SIZE - self.BORDER_SIZE - clearance
        low_inner = self.WALL_CENTER + clearance
        high_inner = self.WALL_CENTER - clearance

        if name in ('top', 'left'):
            return low_outer <= float(value) <= high_inner
        return low_inner <= float(value) <= high_outer

    def _constrain_target_by_min_steps(self, target_pos) -> bool:
        if not self._position_is_free(target_pos):
            return False

        min_steps = int(self.target_min_steps)
        if min_steps <= 0:
            return True

        agent_pos = self.variation_space['agent']['position'].value
        path_length, _ = self.shortest_path(agent_pos, target_pos)
        if not np.isfinite(path_length):
            return False

        speed = float(self.variation_space['agent']['speed'].value.item())
        return path_length / speed >= min_steps

    def _door_center(self, name: str) -> np.ndarray:
        position_space = self.variation_space['door']['positions'][name]
        position = position_space.value
        if position is None:
            position = self.door_positions_init[name]

        if name in ('top', 'bottom'):
            return np.array(
                [self.WALL_CENTER, float(position)],
                dtype=np.float32,
            )
        return np.array(
            [float(position), self.WALL_CENTER],
            dtype=np.float32,
        )

    def _room_of(self, pos_xy) -> str | None:
        pos = np.asarray(pos_xy, dtype=np.float32)
        if not self._position_is_free(pos) and not self._position_in_doorway(
            pos
        ):
            return None
        x_side = 'left' if float(pos[0]) < self.WALL_CENTER else 'right'
        y_side = 'top' if float(pos[1]) < self.WALL_CENTER else 'bottom'
        return f'{y_side}_{x_side}'

    def _position_in_doorway(self, pos_xy) -> bool:
        pos = np.asarray(pos_xy, dtype=np.float32)
        half = self.wall_thickness // 2
        radius = float(self.variation_space['agent']['radius'].value.item())
        wall_min = self.WALL_CENTER - half - radius
        wall_max = self.WALL_CENTER + half + radius
        x_in_wall = wall_min <= float(pos[0]) <= wall_max
        y_in_wall = wall_min <= float(pos[1]) <= wall_max
        return (
            x_in_wall and self._in_vertical_door(float(pos[1]), 1.75)
        ) or (y_in_wall and self._in_horizontal_door(float(pos[0]), 1.75))

    def _valid_door_graph(self):
        graph = defaultdict(list)
        radius = float(self.variation_space['agent']['radius'].value.item())
        sizes = self.variation_space['door']['size'].value
        if sizes is None:
            sizes = self.door_size_init
        for name, (room_a, room_b) in DOOR_EDGES.items():
            if name == self.closed_door:
                continue
            size = float(sizes[DOOR_ORDER.index(name)])
            if size < 1.1 * radius:
                continue
            center = self._door_center(name)
            graph[room_a].append((room_b, name, center))
            graph[room_b].append((room_a, name, center))
        return graph

    def shortest_path(self, start_xy, goal_xy):
        start = np.asarray(start_xy, dtype=np.float32)
        goal = np.asarray(goal_xy, dtype=np.float32)
        start_room = self._room_of(start)
        goal_room = self._room_of(goal)
        if start_room is None or goal_room is None:
            return float('inf'), []
        if start_room == goal_room:
            return float(np.linalg.norm(goal - start)), []

        graph = self._valid_door_graph()
        best_dist = float('inf')
        best_doors: list[str] = []

        def visit(room, last_point, dist, door_names, seen):
            nonlocal best_dist, best_doors
            if room == goal_room:
                total = dist + float(np.linalg.norm(goal - last_point))
                if total < best_dist:
                    best_dist = total
                    best_doors = list(door_names)
                return

            for next_room, door_name, door_center in graph[room]:
                if next_room in seen:
                    continue
                step_dist = float(np.linalg.norm(door_center - last_point))
                if dist + step_dist >= best_dist:
                    continue
                visit(
                    next_room,
                    door_center,
                    dist + step_dist,
                    door_names + [door_name],
                    seen | {next_room},
                )

        visit(start_room, start, 0.0, [], {start_room})
        return best_dist, best_doors

    def expert_waypoint(
        self,
        agent_pos,
        goal_pos,
        door_reach_tol: float,
    ) -> np.ndarray:
        agent_room = self._room_of(agent_pos)
        _, door_names = self.shortest_path(agent_pos, goal_pos)
        if not door_names:
            return np.asarray(goal_pos, dtype=np.float32)

        agent = np.asarray(agent_pos, dtype=np.float32)
        first_door = door_names[0]
        if agent_room is not None:
            passage_point = self._door_passage_point(first_door, agent_room)
            if np.linalg.norm(passage_point - agent) > door_reach_tol:
                return passage_point

        center = self._door_center(first_door)
        if np.linalg.norm(center - agent) > door_reach_tol:
            return center
        return np.asarray(goal_pos, dtype=np.float32)

    def _door_passage_point(self, door_name: str, from_room: str) -> np.ndarray:
        center = self._door_center(door_name).copy()
        edge = DOOR_EDGES[door_name]
        if from_room not in edge:
            return center

        offset = (
            self.wall_thickness // 2
            + float(self.variation_space['agent']['radius'].value.item())
            + 3.0
        )

        if door_name in ('top', 'bottom'):
            if from_room.endswith('_left'):
                center[0] = self.WALL_CENTER + offset
            else:
                center[0] = self.WALL_CENTER - offset
        else:
            if from_room.startswith('top_'):
                center[1] = self.WALL_CENTER + offset
            else:
                center[1] = self.WALL_CENTER - offset
        return center.astype(np.float32)

    def _sample_closed_door(self) -> str | None:
        if self.close_door_prob <= 0.0:
            return None
        if float(self.np_random.uniform(0.0, 1.0)) >= self.close_door_prob:
            return None
        return str(self.np_random.choice(DOOR_ORDER))

    def _closed_door_idx(self) -> int:
        if self.closed_door is None:
            return -1
        return DOOR_ORDER.index(self.closed_door)

    def _set_closed_door_idx(self, closed_door_idx):
        idx = int(np.asarray(closed_door_idx).item())
        if idx == -1:
            self.closed_door = None
        elif 0 <= idx < len(DOOR_ORDER):
            self.closed_door = DOOR_ORDER[idx]
        else:
            raise ValueError(f'Invalid closed_door_idx: {closed_door_idx!r}')
        self._target_img = self._render_frame(agent_pos=self.target_position)

    def _set_state(self, state):
        self.agent_position = torch.tensor(state, dtype=torch.float32)

    def _set_goal_state(self, goal_state):
        self.target_position = torch.tensor(goal_state, dtype=torch.float32)
        goal_value = np.array(goal_state, dtype=np.float32)
        try:
            self.variation_space['target']['position'].set_value(goal_value)
        except ValueError:
            self.variation_space['target']['position']._value = goal_value
        self._target_img = self._render_frame(agent_pos=self.target_position)


class FourRoomDistractorsEnv(FourRoomEnv):
    """Four-room visual navigation with env-owned moving distractors."""

    DEFAULT_DISTRACTOR_COLORS = (
        (0, 114, 178),
        (230, 159, 0),
        (0, 158, 115),
        (204, 121, 167),
        (240, 228, 66),
    )
    DEFAULT_DISTRACTOR_EXPLORE = {
        'speed_noise_std': 0.0,
        'turn_autocorr': 0.0,
        'turn_noise_std': 0.0,
        'target_heading_gain': 0.0,
        'switch_tau': None,
        'switch_mode': 'periodic',
        'teleport_tau': None,
        'teleport_mode': 'periodic',
        'initial_turn_rate_std': 0.0,
        'phase_switch_tau': None,
        'phase_switch_mode': 'periodic',
    }

    def __init__(
        self,
        render_mode: str = 'rgb_array',
        render_target: bool = False,
        init_value: dict | None = None,
        target_min_steps: int | None = None,
        agent_speed: float = 5.0,
        agent_radius: float = 7.0,
        wall_thickness: int = FourRoomEnv.WALL_WIDTH_DEFAULT,
        door_size: list[int] | tuple[int, int, int, int] | None = None,
        door_positions: dict[str, int] | None = None,
        close_door_prob: float = 0.0,
        num_distractors: int = 1,
        min_distractors: int | None = None,
        max_distractors: int | None = None,
        distractor_speed: float | None = 5.0,
        distractor_policy: str = 'explore',
        distractor_explore: dict | None = None,
        distractor_action_noise: float = 0.0,
        distractor_colors: (
            list[list[int]] | tuple[tuple[int, int, int], ...] | None
        ) = None,
        distractor_radius: float | None = None,
        terminate_on_goal: bool = True,
        retarget_on_goal: bool = False,
    ):
        if distractor_radius is not None:
            raise ValueError(
                'distractor_radius is no longer supported. Use '
                'agent_radius or agent.radius so ego and distractors share '
                'one visual and physical radius.'
            )

        if max_distractors is None:
            max_distractors = num_distractors
        if min_distractors is None:
            min_distractors = max_distractors
        self.max_distractors = int(max_distractors)
        self.min_distractors = int(min_distractors)
        if not 0 <= self.min_distractors <= self.max_distractors:
            raise ValueError(
                'Require 0 <= min_distractors <= max_distractors, got '
                f'{self.min_distractors} and {self.max_distractors}.'
            )
        # Capacity: buffers, obs space, and the per-index info keys are all
        # sized to max_distractors so the recorded HDF5 schema is fixed. The
        # per-episode active count is sampled in reset; inactive slots are
        # parked at NaN (never rendered/stepped) so downstream stats/probes
        # can mask them out.
        self.num_distractors = self.max_distractors
        self._n_active = self.max_distractors

        self.distractor_policy = str(distractor_policy)
        self.distractor_speed = (
            None if distractor_speed is None else float(distractor_speed)
        )
        if self.distractor_speed is not None and self.distractor_speed < 0.0:
            raise ValueError('distractor_speed must be non-negative.')
        self.distractor_explore = dict(self.DEFAULT_DISTRACTOR_EXPLORE)
        self.distractor_explore.update(distractor_explore or {})
        self.distractor_action_noise = float(distractor_action_noise)
        self._distractor_goals: torch.Tensor | None = None
        self.distractor_colors = tuple(
            tuple(int(c) for c in color)
            for color in (distractor_colors or self.DEFAULT_DISTRACTOR_COLORS)
        )
        self.distractor_positions = torch.zeros(
            (self.num_distractors, 2),
            dtype=torch.float32,
        )
        self._distractor_controller: ExploreActionController | None = None
        self._distractor_steps_since_teleport = np.zeros(
            self.num_distractors,
            dtype=np.int64,
        )
        # expert_explore per-distractor phase: 0 = explore, 1 = expert.
        self._distractor_phase = np.zeros(self.num_distractors, dtype=np.int64)
        self._distractor_steps_since_phase_switch = np.zeros(
            self.num_distractors,
            dtype=np.int64,
        )

        super().__init__(
            render_mode=render_mode,
            render_target=render_target,
            init_value=init_value,
            target_min_steps=target_min_steps,
            agent_speed=agent_speed,
            agent_radius=agent_radius,
            wall_thickness=wall_thickness,
            door_size=door_size,
            door_positions=door_positions,
            close_door_prob=close_door_prob,
            terminate_on_goal=terminate_on_goal,
            retarget_on_goal=retarget_on_goal,
        )
        self.env_name = 'FourRoomDistractors'
        self._update_observation_space()

    def _update_observation_space(self) -> None:
        base_dim = 2 + 2 + self.MAX_DOOR * 2
        state_dim = base_dim + 2 * self.num_distractors
        self.observation_space = spaces.Box(
            low=0,
            high=self.IMG_SIZE,
            shape=(state_dim,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        options = options or {}

        distractor_xy = options.get(
            'distractor_xy',
            options.get('distractor_state', None),
        )
        if distractor_xy is None:
            self._n_active = int(
                self.np_random.integers(
                    self.min_distractors, self.max_distractors + 1
                )
            )
            self.distractor_positions = self._sample_distractor_positions()
        else:
            self._set_distractor_positions(distractor_xy)

        self._reset_distractor_controller()
        self._reset_distractor_teleport_schedule()
        self._target_img = self._render_frame(agent_pos=self.target_position)

        obs = self._get_obs()
        info = self._get_info()
        info['distance_to_target'] = float(
            torch.norm(self.agent_position - self.target_position)
        )
        return obs, info

    def step(self, action):
        action_t = torch.as_tensor(action, dtype=torch.float32)
        action_t = torch.clamp(action_t, -1.0, 1.0)

        agent_speed = float(self.variation_space['agent']['speed'].value.item())
        pos_next = self.agent_position + action_t * agent_speed
        self.agent_position = self._apply_collisions(
            self.agent_position,
            pos_next,
        )

        self._step_distractors(agent_speed)

        dist = self._distance_to_target()
        goal_reached = dist < 16.0
        terminated = goal_reached and self.terminate_on_goal
        if goal_reached and self.retarget_on_goal and not terminated:
            self._sample_new_target()
            dist = self._distance_to_target()
        truncated = False
        reward = 0.0

        obs = self._get_obs()
        info = self._get_info()
        info['distance_to_target'] = dist
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        base = super()._get_obs()
        if self.num_distractors == 0:
            return base
        distractors = torch.nan_to_num(
            self.distractor_positions.reshape(-1).to(torch.float32)
        )
        return torch.cat([base, distractors], dim=0)

    def _get_info(self):
        ego_xy = self.agent_position.detach().cpu().numpy().astype(np.float32)
        target_xy = (
            self.target_position.detach().cpu().numpy().astype(np.float32)
        )
        info = {
            'env_name': self.env_name,
            'xy': ego_xy,
            'proprio': ego_xy,
            'state': ego_xy,
            'goal_state': target_xy,
            'closed_door_idx': self._closed_door_idx(),
        }

        if self.num_distractors == 0:
            return info

        distractor_xy = (
            self.distractor_positions.detach().cpu().numpy().astype(np.float32)
        )
        all_agent_xy = np.concatenate(
            [ego_xy[None], distractor_xy],
            axis=0,
        ).astype(np.float32)

        info.update(
            {
                'distractor_xy': distractor_xy,
                'all_agent_xy': all_agent_xy,
                'full_state': all_agent_xy.reshape(-1).astype(np.float32),
            }
        )
        for idx in range(self.num_distractors):
            info[f'distractor{idx}_xy'] = distractor_xy[idx]
        return info

    def _render_frame(self, agent_pos: torch.Tensor):
        img = super()._render_frame(agent_pos, draw_agent=False)

        distractor_r = self._distractor_radius()
        for idx in range(self._n_active):
            pos = self.distractor_positions[idx]
            color = np.asarray(
                self.distractor_colors[idx % len(self.distractor_colors)],
                dtype=np.uint8,
            )
            dot = self._gaussian_dot(pos, distractor_r)
            img = self._alpha_blend(img, dot, color)

        agent_color = self.variation_space['agent']['color'].value
        agent_r = float(self.variation_space['agent']['radius'].value.item())
        agent_dot = self._gaussian_dot(agent_pos, agent_r)
        return self._alpha_blend(img, agent_dot, agent_color)

    def _distractor_radius(self) -> float:
        return float(self.variation_space['agent']['radius'].value.item())

    def _sample_distractor_positions(self) -> torch.Tensor:
        positions = torch.full(
            (self.num_distractors, 2), float('nan'), dtype=torch.float32
        )
        for idx in range(self._n_active):
            positions[idx] = self._sample_free_position()
        return positions

    def _sample_free_position(self) -> torch.Tensor:
        pos_min = float(self.BORDER_SIZE)
        pos_max = float(self.IMG_SIZE - self.BORDER_SIZE - 1)

        for _ in range(1000):
            pos = np.asarray(
                self.np_random.uniform(pos_min, pos_max, size=2),
                dtype=np.float32,
            )
            if self._position_is_free(pos):
                return torch.as_tensor(pos, dtype=torch.float32)

        raise RuntimeError('Failed to sample a valid distractor position.')

    def _reset_distractor_controller(self) -> None:
        if self.distractor_policy == 'static' or self.num_distractors == 0:
            self._distractor_controller = None
            return
        if self.distractor_policy == 'expert':
            self._distractor_controller = None
            self._sample_distractor_goals()
            return
        if self.distractor_policy not in ('explore', 'expert_explore'):
            raise ValueError(
                "distractor_policy must be one of {'explore', 'static', "
                f"'expert', 'expert_explore'}}, got {self.distractor_policy!r}."
            )

        seed = int(self.np_random.integers(0, 1_000_000_000))
        self._distractor_controller = ExploreActionController(
            self.num_distractors,
            seed=seed,
            **self._distractor_explore_kwargs(),
        )

        if self.distractor_policy == 'expert_explore':
            assert self._phase_switch_tau() is not None, (
                'expert_explore requires a finite '
                'distractor_explore.phase_switch_tau.'
            )
            self._distractor_phase[:] = 0
            self._distractor_steps_since_phase_switch[:] = 0
            self._sample_distractor_goals()

    def _distractor_explore_kwargs(self) -> dict:
        explore_kwargs = dict(self.distractor_explore)
        explore_kwargs.pop('teleport_tau', None)
        explore_kwargs.pop('teleport_mode', None)
        explore_kwargs.pop('phase_switch_tau', None)
        explore_kwargs.pop('phase_switch_mode', None)
        if self.distractor_speed is not None:
            explore_kwargs.pop('speed_mean', None)
            explore_kwargs['speed_mean'] = 1.0
        return explore_kwargs

    def _reset_distractor_teleport_schedule(self) -> None:
        self._distractor_steps_since_teleport = np.zeros(
            self.num_distractors,
            dtype=np.int64,
        )

    def _teleport_tau(self) -> float | None:
        tau = self.distractor_explore.get('teleport_tau', None)
        if tau is None:
            return None
        tau = float(tau)
        if not np.isfinite(tau) or tau <= 0:
            return None
        return tau

    def _distractor_teleport_mask(self) -> np.ndarray:
        tau = self._teleport_tau()
        if tau is None or self.num_distractors == 0:
            return np.zeros(self.num_distractors, dtype=bool)

        mode = str(self.distractor_explore.get('teleport_mode', 'periodic'))
        if mode == 'poisson':
            probability = 1.0 - np.exp(-1.0 / tau)
            return (
                self.np_random.uniform(0.0, 1.0, size=self.num_distractors)
                < probability
            )
        if mode == 'periodic':
            interval = max(1, int(round(tau)))
            self._distractor_steps_since_teleport += 1
            mask = self._distractor_steps_since_teleport >= interval
            if np.any(mask):
                self._distractor_steps_since_teleport[mask] = 0
            return mask
        raise ValueError(
            "teleport_mode must be one of {'poisson', 'periodic'}, "
            f"got {mode!r}."
        )

    def _teleport_distractors_if_needed(self) -> None:
        teleport_mask = self._distractor_teleport_mask()
        if not np.any(teleport_mask):
            return

        for idx in np.flatnonzero(teleport_mask):
            actor_idx = int(idx)
            if actor_idx >= self._n_active:
                continue
            self.distractor_positions[actor_idx] = self._sample_free_position()
            if self._distractor_controller is not None:
                self._distractor_controller.reset_actor_direction(actor_idx)

    def _sample_free_in_room(self, room: str, max_tries: int = 200):
        pos = self._sample_free_position().cpu().numpy()
        for _ in range(max_tries):
            if self._room_of(pos) == room:
                break
            pos = self._sample_free_position().cpu().numpy()
        return pos.astype(np.float32)

    def _sample_adjacent_goal(self, pos, graph):
        room = self._room_of(pos)
        neighbors = [nr for (nr, _door, _c) in graph[room]] if room else []
        if not neighbors:
            neighbors = [r for r in ROOMS if r != room]
        target_room = str(self.np_random.choice(neighbors))
        return self._sample_free_in_room(target_room)

    def _sample_distractor_goals(self) -> None:
        graph = self._valid_door_graph()
        goals = [
            self._sample_adjacent_goal(
                self.distractor_positions[idx].cpu().numpy(), graph
            )
            for idx in range(self._n_active)
        ]
        self._distractor_goals = (
            torch.as_tensor(np.stack(goals), dtype=torch.float32)
            if goals
            else torch.zeros((0, 2), dtype=torch.float32)
        )

    def _expert_step_actor(
        self, idx: int, speed: float, door_tol: float, goal_tol: float
    ) -> bool:
        """Move distractor idx one step toward its goal; return goal reached."""
        pos = self.distractor_positions[idx].cpu().numpy()
        goal = self._distractor_goals[idx].cpu().numpy()
        waypoint = self.expert_waypoint(pos, goal, door_tol)
        direction = (waypoint - pos).astype(np.float32)
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-8 else np.zeros(2, np.float32)
        if self.distractor_action_noise > 0.0:
            direction = direction + self.np_random.normal(
                0.0, self.distractor_action_noise, size=2
            )
        action = np.clip(direction, -1.0, 1.0).astype(np.float32)
        pos_next = self.distractor_positions[idx] + torch.as_tensor(
            action, dtype=torch.float32
        ) * speed
        self.distractor_positions[idx] = self._apply_collisions(
            self.distractor_positions[idx], pos_next
        )
        new_pos = self.distractor_positions[idx].cpu().numpy()
        return float(np.linalg.norm(new_pos - goal)) < goal_tol

    def _step_expert_distractors(self, speed: float) -> None:
        if self._distractor_goals is None:
            self._sample_distractor_goals()
        graph = self._valid_door_graph()
        door_tol = 10.5
        goal_tol = max(10.0, speed)
        for idx in range(self._n_active):
            reached = self._expert_step_actor(idx, speed, door_tol, goal_tol)
            if reached:
                self._distractor_goals[idx] = torch.as_tensor(
                    self._sample_adjacent_goal(
                        self.distractor_positions[idx].cpu().numpy(), graph
                    ),
                    dtype=torch.float32,
                )

    def _phase_switch_tau(self) -> float | None:
        tau = self.distractor_explore.get('phase_switch_tau', None)
        if tau is None:
            return None
        tau = float(tau)
        if not np.isfinite(tau) or tau <= 0:
            return None
        return tau

    def _phase_switch_mask(self) -> np.ndarray:
        tau = self._phase_switch_tau()
        if tau is None or self.num_distractors == 0:
            return np.zeros(self.num_distractors, dtype=bool)

        mode = str(self.distractor_explore.get('phase_switch_mode', 'periodic'))
        if mode == 'poisson':
            probability = 1.0 - np.exp(-1.0 / tau)
            return (
                self.np_random.uniform(0.0, 1.0, size=self.num_distractors)
                < probability
            )
        if mode == 'periodic':
            interval = max(1, int(round(tau)))
            self._distractor_steps_since_phase_switch += 1
            mask = self._distractor_steps_since_phase_switch >= interval
            if np.any(mask):
                self._distractor_steps_since_phase_switch[mask] = 0
            return mask
        raise ValueError(
            "phase_switch_mode must be one of {'poisson', 'periodic'}, "
            f"got {mode!r}."
        )

    def _step_expert_explore_distractors(self, speed: float) -> None:
        if self._distractor_goals is None:
            self._sample_distractor_goals()
        graph = self._valid_door_graph()
        door_tol = 10.5
        goal_tol = max(10.0, speed)

        explore_actions = self._distractor_controller.get_actions()
        switch_mask = self._phase_switch_mask()

        for idx in range(self._n_active):
            # explore -> expert: pick a fresh adjacent-room goal and start
            # navigating on this same step.
            if self._distractor_phase[idx] == 0 and switch_mask[idx]:
                self._distractor_phase[idx] = 1
                self._distractor_goals[idx] = torch.as_tensor(
                    self._sample_adjacent_goal(
                        self.distractor_positions[idx].cpu().numpy(), graph
                    ),
                    dtype=torch.float32,
                )

            if self._distractor_phase[idx] == 0:
                action_t = torch.as_tensor(
                    explore_actions[idx], dtype=torch.float32
                )
                pos_next = self.distractor_positions[idx] + action_t * speed
                self.distractor_positions[idx] = self._apply_collisions(
                    self.distractor_positions[idx], pos_next
                )
                continue

            reached = self._expert_step_actor(idx, speed, door_tol, goal_tol)
            if reached:
                # expert -> explore: resume the wander with a fresh heading and
                # restart the phase clock.
                self._distractor_phase[idx] = 0
                self._distractor_steps_since_phase_switch[idx] = 0
                self._distractor_controller.reset_actor_direction(idx)

    def _step_distractors(self, agent_speed: float) -> None:
        if self.num_distractors == 0 or self.distractor_policy == 'static':
            return
        speed = (
            self.distractor_speed
            if self.distractor_speed is not None
            else agent_speed
        )
        if self.distractor_policy == 'expert':
            self._step_expert_distractors(speed)
            return
        if self._distractor_controller is None:
            self._reset_distractor_controller()
        assert self._distractor_controller is not None

        if self.distractor_policy == 'expert_explore':
            self._step_expert_explore_distractors(speed)
            return

        actions = self._distractor_controller.get_actions()
        for idx in range(self._n_active):
            action_t = torch.as_tensor(actions[idx], dtype=torch.float32)
            pos_next = self.distractor_positions[idx] + action_t * speed
            self.distractor_positions[idx] = self._apply_collisions(
                self.distractor_positions[idx],
                pos_next,
            )
        self._teleport_distractors_if_needed()

    def _set_distractor_positions(self, distractor_xy):
        positions = torch.as_tensor(distractor_xy, dtype=torch.float32)
        positions = positions.reshape(self.num_distractors, 2)
        self.distractor_positions = positions
        # Restored frames store inactive distractors as NaN (contiguous at the
        # tail); recover the active count so eval only renders/steps real ones.
        self._n_active = int((~torch.isnan(positions).any(dim=1)).sum())
        # Active count just changed; drop stale goals so the expert policy
        # resamples one goal per active distractor on the next step.
        self._distractor_goals = None

    def set_distractor_policy(
        self,
        policy: str,
        explore_params: dict | None = None,
    ) -> None:
        self.distractor_policy = str(policy)
        if explore_params is not None:
            self.distractor_explore = dict(self.DEFAULT_DISTRACTOR_EXPLORE)
            self.distractor_explore.update(explore_params)
        self._reset_distractor_controller()
        self._reset_distractor_teleport_schedule()
