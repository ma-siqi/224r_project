import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from wrappers import DQNExplorationWrapper, PPOExplorationWrapper
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from eval import MetricWrapper

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

DIRECTIONS = np.array([
    (0, -1),   # 0: N
    (1, -1),   # 1: NE
    (1, 0),    # 2: E
    (1, 1),    # 3: SE
    (0, 1),    # 4: S
    (-1, 1),   # 5: SW
    (-1, 0),   # 6: W
    (-1, -1)   # 7: NW
])


class VacuumEnv(gym.Env):

    def __init__(
        self,
        grid_size=(6, 6),
        reward_clean_tile=1.0,
        penalty_forward=-0.05,
        penalty_rotation=-0.001,
        penalty_invalid_move=-0.5,
        reward_done_clean=5_000.0,
        reward_return_home=10_000.0,
        dirt_num=5
    ):
        super().__init__()
        self.grid_size = grid_size
        self.reward_clean_tile = reward_clean_tile
        self.penalty_forward = penalty_forward
        self.penalty_rotation = penalty_rotation
        self.penalty_invalid_move = penalty_invalid_move
        self.reward_done_clean = reward_done_clean
        self.reward_return_home = reward_return_home

        self.action_space = spaces.Discrete(3) # 0=forward, 1=rotate left, 2=rotate right

        # dirt map: 0 for clean, 1 for dirty; known obstacle map: 0 for clear, 1 for obstacle
        self.observation_space = spaces.Dict({
            "start_pos": spaces.Box(low=0, high=1, shape=self.grid_size, dtype=np.uint8),
            "agent_pos": spaces.Box(low=0, high=1, shape=self.grid_size, dtype=np.uint8),
            "agent_orient": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "dirt_map": spaces.Box(low=0, high=1, shape=self.grid_size, dtype=np.uint8),
            "obstacle_map": spaces.Box(low=0, high=1, shape=self.grid_size, dtype=np.uint8)
        })

        self.orientations = {
            0: (-1, 0), 1: (-1, 1), 2: (0, 1), 3: (1, 1),
            4: (1, 0), 5: (1, -1), 6: (0, -1), 7: (-1, -1)
        } # define the 8 directions clockwise

        self.dirt_num = dirt_num # ratio of dirty tiles
        self.rng = np.random.default_rng()  # used to generate random layouts

        self.start_pos = [0, 0] # agent starting position is the upper-left corner
        self.agent_pos = self.start_pos.copy()
        self.agent_orient = 0
        self.cleaned_map = np.zeros(self.grid_size, dtype=np.uint8)
        self.dirt_map = np.zeros(self.grid_size, dtype=np.uint8)
        self.obstacle_map = np.zeros(self.grid_size, dtype=np.uint8)
        self.known_obstacle_map = np.zeros(self.grid_size, dtype=np.float32)
        self.path_map = np.zeros(self.grid_size, dtype=np.uint8)

        self.reset() # reset the environment when initialized

    def reset(self, *, seed=None, options=None):
        self.reward_done_clean = 5_000.0

        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # layout reset
        self.cleaned_map = np.zeros(self.grid_size, dtype=np.uint8)

        # generate obstacle layout
        self.obstacle_map = np.zeros(self.grid_size, dtype=np.uint8)
        walls = options.get("walls") if options else None
        if walls is None:
            pass
        elif isinstance(walls, list) and len(walls) > 0:
            self.add_wall(walls)
        else:
            self.generate_random_rooms()

        # generate dirt layout: place dirt_num clusters on non-obstacle tiles
        self.dirt_map = np.zeros(self.grid_size, dtype=np.uint8)
        if self.dirt_num == 0:
            self.dirt_map = np.ones(self.grid_size, dtype=np.uint8) - self.obstacle_map
        else:
            self.generate_random_dirt_clusters(self.dirt_num)

        # initialize agent observations
        self.start_pos = [0, 0]
        self.agent_pos = list(self.start_pos)
        self.agent_orient = 2 # agent starting orientation is facing right

        self.cleaned_map[tuple(self.agent_pos)] = 1
        self.dirt_map[tuple(self.agent_pos)] = 0
        self.known_obstacle_map = np.zeros(self.grid_size, dtype=np.uint8)

        # store rendered path
        self.path_map = np.zeros(self.grid_size, dtype=np.uint8)
        self.path_map[tuple(self.agent_pos)] = 1  # mark the starting position

        self._update_local_view()
        self._rotate_and_update_all_directions()

        return self._get_obs(), {}

    def generate_random_rooms(self):
        """Place walls to split the grid and create doorways randomly"""
        self.obstacle_map[:] = 0
        h, w = self.grid_size
        rows, cols = 2, 2  # can be adjusted to create more rooms
        room_h, room_w = h // rows, w // cols

        for i in range(rows):
            for j in range(cols):
                top = i * room_h
                left = j * room_w
                bottom = (i + 1) * room_h
                right = (j + 1) * room_w

                # Draw vertical wall to the right (if not last col)
                if j < cols - 1:
                    wall_col = right - 1
                    self.obstacle_map[top:bottom, wall_col] = 1

                    # Punch a wider door randomly (2–3 tiles)
                    door_y = self.rng.integers(top + 1, bottom - 2)
                    door_height = self.rng.integers(2, 4)  # width of door (2–3 tiles)
                    for dy in range(door_y, min(bottom - 1, door_y + door_height)):
                        self.obstacle_map[dy, wall_col] = 0

                # Draw horizontal wall below (if not last row)
                if i < rows - 1:
                    wall_row = bottom - 1
                    self.obstacle_map[wall_row, left:right] = 1

                    # Punch a wider door randomly (2–3 tiles)
                    door_x = self.rng.integers(left + 1, right - 2)
                    door_width = self.rng.integers(2, 4)
                    for dx in range(door_x, min(right - 1, door_x + door_width)):
                        self.obstacle_map[wall_row, dx] = 0

        # Ensure start_pos is not blocked
        sx, sy = self.start_pos
        if self.obstacle_map[sx, sy] == 1:
            self.obstacle_map[sx, sy] = 0

    def add_wall(self, walls):
        """Manually add wall tiles (obstacles) at specified positions"""
        for x, y in walls:
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                self.obstacle_map[x, y] = 1
    
    def generate_random_dirt_clusters(self, num_dirt):
        """Place dirt clusters of various fixed sizes randomly on the grid"""
        self.dirt_map[:] = 0  # Clear existing dirt
        h, w = self.grid_size

        possible_sizes = [(2, 2), (4, 4), (2, 4)]
        attempts = 0
        max_attempts = num_dirt * 10  # Limit attempts to prevent infinite loop

        placed = 0
        while placed < num_dirt and attempts < max_attempts:
            dh, dw = self.rng.choice(possible_sizes)
            top = self.rng.integers(0, h - dh + 1)
            left = self.rng.integers(0, w - dw + 1)

            # Check if the area is free of obstacles
            if np.all(self.obstacle_map[top:top+dh, left:left+dw] == 0) and \
            np.all(self.dirt_map[top:top+dh, left:left+dw] == 0):
                self.dirt_map[top:top+dh, left:left+dw] += 1
                placed += 1
            attempts += 1

    def _rotate_and_update_all_directions(self):
        original_orient = self.agent_orient
        for i in range(8):
            self.agent_orient = i
            self._update_local_view()
        self.agent_orient = original_orient

    def _encode_position(self, pos):
        one_hot = np.zeros(self.grid_size, dtype=np.uint8)
        x, y = pos
        one_hot[x, y] = 1
        return one_hot

    def _obstacle_ahead(self):
        dx, dy = self.orientations[self.agent_orient]
        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        return int(
            nx < 0 or nx >= self.grid_size[0] or
            ny < 0 or ny >= self.grid_size[1] or
            self.known_obstacle_map[nx, ny] == 1
        )

    def _obstacle_in_direction(self, direction):
        shift = {"left": -1, "right": 1}
        new_orient = (self.agent_orient + shift[direction]) % 8
        dx, dy = self.orientations[new_orient]
        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        return int(
            nx < 0 or nx >= self.grid_size[0] or
            ny < 0 or ny >= self.grid_size[1] or
            self.known_obstacle_map[nx, ny] == 1
        )

    def _compute_local_view(self):
        view = np.zeros(8, dtype=np.uint8)
        for i, (dx, dy) in enumerate(DIRECTIONS):
            nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            if not (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]):
                view[i] = 1
            elif self.known_obstacle_map[nx, ny] == 1:
                view[i] = 1
            else:
                view[i] = 0
        return view

    def _get_obs(self):
        theta = 2 * np.pi * self.agent_orient / 8
        orient_vec = [np.cos(theta), np.sin(theta)]

        return {
            "start_pos": np.array(self._encode_position(self.start_pos), dtype=np.uint8),
            "agent_pos": np.array(self._encode_position(self.agent_pos), dtype=np.uint8),
            "agent_orient": np.array(orient_vec, dtype=np.float32),
            "dirt_map": np.array(self.dirt_map, dtype=np.uint8),
            "obstacle_map": np.array(self.obstacle_map, dtype=np.uint8)
        }

    def _update_local_view(self):
        rotated_dirs = np.roll(DIRECTIONS, -self.agent_orient, axis=0)
        for i in range(8):
            x = self.agent_pos[0] + rotated_dirs[i][0]
            y = self.agent_pos[1] + rotated_dirs[i][1]
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                if self.obstacle_map[x, y] == 1:
                    self.known_obstacle_map[x, y] = 1

    def step(self, action):
        """Take a step in the environment"""
        reward = 0
        terminated = False
        truncated = False

        # take action
        if action == 0:  # move forward
            dx, dy = self.orientations[self.agent_orient]
            nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            if (
                0 <= nx < self.grid_size[0]
                and 0 <= ny < self.grid_size[1]
                and self.obstacle_map[nx, ny] == 0
            ):
                self.agent_pos = [nx, ny]
                reward += self.penalty_forward # forward movement cost
            else:
                reward += self.penalty_invalid_move # invalid move (out-of-bounds or run into obstacles)
        elif action == 1: # rotate left
            self.agent_orient = (self.agent_orient - 1) % 8
            reward += self.penalty_rotation # rotation cost
            #self._update_local_view() # after rotation, can also update local knowledge

        elif action == 2: # rotate right
            self.agent_orient = (self.agent_orient + 1) % 8
            reward += self.penalty_rotation # rotation cost
            #self._update_local_view() # after rotation, can also update local knowledge

        self._update_local_view() # update local knowledge after action

        x, y = self.agent_pos

        # update the grids status
        if self.dirt_map[x, y] == 1:
            reward += self.reward_clean_tile
        self.cleaned_map[x, y] = 1
        self.dirt_map[x, y] = 0
        self.path_map[x, y] += 1
        if self.path_map[x, y] >= 127:
            truncated = True

        # update reward if all cleaned, for now just clean up everything
        if np.all(self.dirt_map == 0):
            if self.agent_pos == self.start_pos:
                reward += self.reward_return_home
                terminated = True
            else:
                reward += self.reward_done_clean
                self.reward_done_clean = 0

        return self._get_obs(), reward, terminated, truncated, {}

    def render_frame(self):
        """Return a visual frame/image"""
        h, w = self.grid_size
        grid_image = np.zeros((h, w))

        # Encode layers
        grid_image[self.cleaned_map == 1] = 3
        grid_image[self.dirt_map == 1] = 2
        grid_image[self.obstacle_map == 1] = 1

        for (i, j), count in np.ndenumerate(getattr(self, "path_map", np.zeros_like(grid_image))):
            if count > 0 and grid_image[i, j] == 0:
                grid_image[i, j] = 0.5  # trail

        x, y = self.agent_pos
        grid_image[x, y] = 4

        cmap = mcolors.ListedColormap(["white", "black", "saddlebrown", "lightblue", "gray", "red"])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(grid_image, cmap=cmap, norm=norm)
        ax.set_title(f"Agent @ {self.agent_pos}, facing {self.agent_orient}")
        ax.axis("off")

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)

        return img

    def _compute_coverage_ratio(self):
        total_dirt = np.sum(self.dirt_map == 1) + np.sum(self.cleaned_map == 1)
        cleaned = np.sum(self.cleaned_map == 1)
        return cleaned / total_dirt if total_dirt > 0 else 0

    def _compute_redundancy_rate(self):
        path_visits = self.path_map[self.obstacle_map == 0]
        return np.mean(path_visits) if path_visits.size > 0 else 0

    def _compute_revisit_ratio(self):
        """Calculate the ratio of revisits to total steps.
        A revisit is counted when a cell is visited more than once."""
        total_steps = np.sum(self.path_map)
        if total_steps == 0:
            return 0.0
        revisits = np.sum(np.maximum(self.path_map - 1, 0))
        return revisits / total_steps

    def compute_metrics(self):
        return {
            "coverage_rate": self._compute_coverage_ratio(),
            "redundancy_rate": self._compute_redundancy_rate(),
            "revisit_ratio": self._compute_revisit_ratio()
        }

class WrappedVacuumEnv:
    def __init__(self, grid_size, dirt_num, max_steps, algo, walls=None):
        self.walls = walls
        self.grid_size = grid_size
        self.dirt_num = dirt_num
        self.max_steps = max_steps
        self.base_env = None
        self.algo = algo

    def __call__(self):
        env = gym.make("VacuumEnv-v0", grid_size=self.grid_size, dirt_num=self.dirt_num)
        env = TimeLimit(env, max_episode_steps=self.max_steps)
        if self.algo == "ppo":
            env = PPOExplorationWrapper(env)
        elif self.algo == "dqn":
            env = DQNExplorationWrapper(env)
        env = MetricWrapper(env)
        env = Monitor(env)

        env = FlattenObservation(env)
        env.reset(options={"walls": self.walls})
        self.base_env = env
        return env

class GridFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        grid_shape = observation_space['dirt_map'].shape
        self.grid_h, self.grid_w = grid_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            sample_input = torch.zeros(1, 4, self.grid_h, self.grid_w)
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 2, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        spatial_maps = torch.stack([
            obs["start_pos"],
            obs["agent_pos"],
            obs["dirt_map"],
            obs["obstacle_map"]
        ], dim=1).float()

        cnn_out = self.cnn(spatial_maps)
        orient = obs["agent_orient"].float()

        return self.linear(torch.cat([cnn_out, orient], dim=1))

