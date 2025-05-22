import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class VacuumEnv(gym.Env):
    metadata = {"render_modes": ["human", "plot"]}

    def __init__(
        self,
        grid_size=(6, 6),
        reward_clean_tile=1.0,
        penalty_revisit=-0.5,
        penalty_forward=-0.1,
        penalty_rotation=-0.05,
        penalty_invalid_move=-1.0,
        reward_return_home=100.0,
        penalty_delay_return=-0.05,
        render_mode=None
    ):
        super().__init__()
        self.grid_size = grid_size

        self.reward_clean_tile = reward_clean_tile
        self.penalty_revisit = penalty_revisit
        self.penalty_forward = penalty_forward
        self.penalty_rotation = penalty_rotation
        self.penalty_invalid_move = penalty_invalid_move
        self.reward_return_home = reward_return_home
        self.penalty_delay_return = penalty_delay_return

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.MultiDiscrete([grid_size[0], grid_size[1]]),
            "agent_orient": spaces.Discrete(8),
            "local_view": spaces.MultiBinary(3)
        })

        self.orientations = {
            0: (-1, 0), 1: (-1, 1), 2: (0, 1), 3: (1, 1),
            4: (1, 0), 5: (1, -1), 6: (0, -1), 7: (-1, -1)
        }

        self.rng = np.random.default_rng()
        self.start_pos = [0, 0]
        self.render_mode = render_mode
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.cleaned_map = np.zeros(self.grid_size, dtype=np.uint8)
        self.dirt_map = np.ones(self.grid_size, dtype=np.uint8)
        self.obstacle_map = np.zeros(self.grid_size, dtype=np.uint8)
        walls = options.get("walls") if options else None
        if walls is not None:
            self.add_wall(walls)
        #else:
        #    self.generate_random_rooms()

        self.agent_pos = list(self.start_pos)
        self.agent_orient = 2

        self.cleaned_map[tuple(self.agent_pos)] = 1
        self.dirt_map[tuple(self.agent_pos)] = 0

        # add this to store rendered path
        self.path_map = np.zeros(self.grid_size, dtype=np.uint8)
        self.path_map[tuple(self.agent_pos)] = 1  # mark starting position

        # add the counter to avoid being stuck
        self.prev_pos = list(self.agent_pos)
        self.stay_counter = 0


        return self._get_obs(), {}

    def generate_random_rooms(self):
        self.obstacle_map[:] = 0
        h, w = self.grid_size

        v_split = self.rng.integers(2, w - 2)
        h_split = self.rng.integers(2, h - 2)

        self.obstacle_map[:, v_split] = 1
        door_y = self.rng.integers(1, h - 1)
        self.obstacle_map[door_y, v_split] = 0

        self.obstacle_map[h_split, :] = 1
        door_x = self.rng.integers(1, w - 1)
        self.obstacle_map[h_split, door_x] = 0

        
    def add_wall(self, walls):
        """
        Manually add wall tiles (obstacles) at specified positions.

        Args:
            positions (list of tuple): A list of (x, y) coordinates where walls should be placed.
        """
        for x, y in walls:
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                self.obstacle_map[x, y] = 1

    def _get_obs(self):
        front = self._check_cell_in_direction(self.agent_orient)
        left = self._check_cell_in_direction((self.agent_orient - 2) % 8)
        right = self._check_cell_in_direction((self.agent_orient + 2) % 8)
        return {
            "agent_pos": np.array(self.agent_pos, dtype=np.int32),
            "agent_orient": self.agent_orient,
            "local_view": np.array([front, left, right], dtype=np.int8)
        }

    def _check_cell_in_direction(self, direction):
        dx, dy = self.orientations[direction]
        x, y = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        if (
            0 <= x < self.grid_size[0]
            and 0 <= y < self.grid_size[1]
            and self.obstacle_map[x, y] == 0
        ):
            return 1 if self.dirt_map[x, y] > 0 else 0
        return 0

    def step(self, action):
        reward = 0
        moved = False

        if action == 0:
            dx, dy = self.orientations[self.agent_orient]
            nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            if (
                0 <= nx < self.grid_size[0]
                and 0 <= ny < self.grid_size[1]
                and self.obstacle_map[nx, ny] == 0
            ):
                self.agent_pos = [nx, ny]
                moved = True
                reward += self.penalty_forward
            else:
                reward += self.penalty_invalid_move
        elif action == 1:
            self.agent_orient = (self.agent_orient - 1) % 8
            reward += self.penalty_rotation
        elif action == 2:
            self.agent_orient = (self.agent_orient + 1) % 8
            reward += self.penalty_rotation

        x, y = self.agent_pos
        if moved:
            self.path_map[tuple(self.agent_pos)] += 1  # count visits to each cell
            if self.dirt_map[x, y] == 1:
                reward += self.reward_clean_tile
                self.dirt_map[x, y] = 0
                self.cleaned_map[x, y] = 1
            else:
                reward += self.penalty_revisit

        # check if stay
        if self.agent_pos == self.prev_pos:
            self.stay_counter += 1
        else:
            self.stay_counter = 0
        self.prev_pos = list(self.agent_pos)

        all_cleaned = np.all(self.dirt_map[self.obstacle_map == 0] == 0)
        at_start = self.agent_pos == self.start_pos
        terminated = bool(all_cleaned and at_start)
        truncated = False

        # terminate if get stuck
        if self.stay_counter >= 5:
            reward += -5.0
        if self.stay_counter >= 10:
            reward += -10.0
        if self.stay_counter >= 20:
            reward += -40.0
            #terminated = True
            #print("Terminated: Agent stayed in place too long.")

        if terminated:
            reward += self.reward_return_home
        elif all_cleaned and not at_start:
            reward += self.penalty_delay_return

        return self._get_obs(), reward, terminated, truncated, {
            "all_cleaned": all_cleaned,
            "returned_home": at_start
        }

    def render(self):
        if self.render_mode == "human":
            # existing text-based rendering
            grid = np.full(self.grid_size, '.', dtype='<U1')
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    if self.obstacle_map[i, j]:
                        grid[i, j] = '#'
                    elif self.dirt_map[i, j]:
                        grid[i, j] = 'D'
                    elif self.cleaned_map[i, j]:
                        grid[i, j] = '*'
            x, y = self.agent_pos
            grid[x, y] = 'A'
            print("\n".join(" ".join(row) for row in grid))
            print(f"Orientation: {self.agent_orient}\n")

        elif self.render_mode == "plot":
            h, w = self.grid_size
            grid_image = np.zeros((h, w))

            # Encode basic layers
            grid_image[self.cleaned_map == 1] = 3  # light blue
            grid_image[self.dirt_map == 1] = 2     # brown
            grid_image[self.obstacle_map == 1] = 1 # black

            # Overlay path as a different value (we'll blend color later)
            for (i, j), count in np.ndenumerate(self.path_map):
                if count > 0 and grid_image[i, j] == 0:
                    grid_image[i, j] = 0.5  # light gray trail

            # Agent position overrides others
            x, y = self.agent_pos
            grid_image[x, y] = 4  # red

            # Create a custom color map
            from matplotlib import colors
            cmap = colors.ListedColormap(["white", "black", "saddlebrown", "lightblue", "gray", "red"])
            bounds = [0, 1, 2, 3, 4, 5]
            norm = colors.BoundaryNorm(bounds, cmap.N)

            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 8))
            plt.imshow(grid_image, cmap=cmap, norm=norm)
            plt.title(f"Vacuum Path - Orientation: {self.agent_orient}")
            plt.xticks([]); plt.yticks([])
            plt.show()

    def render_frame(self):
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

