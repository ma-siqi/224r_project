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
        reward_return_home=10000.0,
        penalty_delay_return=-0.05,
        dirty_ratio=0.9,
        render_mode=None,
        use_counter=True
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
        self.use_internal_stuck_penalty = use_counter  # default to using counter to penalize being stuck; disable when using wrappers

        self.action_space = spaces.Discrete(3) # 0=forward, 1=rotate left, 2=rotate right
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.MultiDiscrete([grid_size[0], grid_size[1]]), # agent position on the grid
            "agent_orient": spaces.Discrete(8), # the agent can observe 8 tiles around itself
            "local_view": spaces.MultiBinary(3) # [front, left, right]
        })

        self.orientations = {
            0: (-1, 0), 1: (-1, 1), 2: (0, 1), 3: (1, 1),
            4: (1, 0), 5: (1, -1), 6: (0, -1), 7: (-1, -1)
        } # define the 8 directions

        self.start_pos = [0, 0] # agent starting position is the upper-left corner
        self.dirty_ratio = dirty_ratio # ratio of dirty tiles

        self.rng = np.random.default_rng()  # used to generate random layouts
        self.render_mode = render_mode
        self.reset() # reset the environment when initialized

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # layout reset
        self.cleaned_map = np.zeros(self.grid_size, dtype=np.uint8)

        # generate obstacle layout
        self.obstacle_map = np.zeros(self.grid_size, dtype=np.uint8)
        walls = options.get("walls") if options else None
        if walls is not None:
            self.add_wall(walls)
        else:
            self.generate_random_rooms()

        # generate dirt layout: only dirty_ratio non-obstacle tiles are dirty
        self.dirt_map = np.zeros(self.grid_size, dtype=np.uint8)
        cleanable_mask = (self.obstacle_map == 0)
        random_dirty = self.rng.random(self.grid_size) < self.dirty_ratio
        self.dirt_map[np.logical_and(cleanable_mask, random_dirty)] = 1

        self.agent_pos = list(self.start_pos)
        self.agent_orient = 2 # agent starting orientation is facing right
        self.cleaned_map[tuple(self.agent_pos)] = 1
        self.dirt_map[tuple(self.agent_pos)] = 0

        # store rendered path
        self.path_map = np.zeros(self.grid_size, dtype=np.uint8)
        self.path_map[tuple(self.agent_pos)] = 1  # mark the starting position

        # counter used in internal penalty to avoid being stuck or wrappers
        self.prev_pos = list(self.agent_pos)
        self.stay_counter = 0

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

    def _get_obs(self):
        """Check surroundings and return observation"""
        front = self._check_cell_in_direction(self.agent_orient)
        left = self._check_cell_in_direction((self.agent_orient - 2) % 8)
        right = self._check_cell_in_direction((self.agent_orient + 2) % 8)
        return {
            "agent_pos": np.array(self.agent_pos, dtype=np.int32),
            "agent_orient": self.agent_orient,
            "local_view": np.array([front, left, right], dtype=np.int8)
        }

    def _check_cell_in_direction(self, direction):
        """Check the dirt map in a specific direction"""
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
        """Take a step in the environment"""
        reward = 0
        moved = False

        if action == 0:  # move forward
            dx, dy = self.orientations[self.agent_orient]
            nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            if (
                0 <= nx < self.grid_size[0]
                and 0 <= ny < self.grid_size[1]
                and self.obstacle_map[nx, ny] == 0
            ):
                self.agent_pos = [nx, ny]
                moved = True
                reward += self.penalty_forward # forward movement cost
            else:
                reward += self.penalty_invalid_move # invalid move (out-of-bounds or run into obstacles)
        elif action == 1: # rotate left
            self.agent_orient = (self.agent_orient - 1) % 8
            reward += self.penalty_rotation # rotation cost
        elif action == 2: # rotate right
            self.agent_orient = (self.agent_orient + 1) % 8
            reward += self.penalty_rotation # rotation cost

        if not hasattr(self, 'prev_dirty_count'):
            self.prev_dirty_count = np.sum(self.dirt_map == 1)

        x, y = self.agent_pos
        if moved:
            # count visits to each cell
            max_visits = 10000
            self.path_map[tuple(self.agent_pos)] = min(self.path_map[tuple(self.agent_pos)] + 1, max_visits)
            # update dirt and cleaned maps
            if self.dirt_map[x, y] == 1:
                reward += self.reward_clean_tile
                self.dirt_map[x, y] = 0
                self.cleaned_map[x, y] = 1
            else:
                visits = self.path_map[x, y]
                revisit_penalty = self.penalty_revisit * visits
                reward += revisit_penalty # revisiting a cleaned tile penalty scaled by number of visits

        current_dirty_count = np.sum(self.dirt_map == 1)
        progress = self.prev_dirty_count - current_dirty_count
        if progress > 0:
            reward += progress * 0.5  # Can tune this
        self.prev_dirty_count = current_dirty_count

        # track previous position
        self.prev_pos = list(self.agent_pos)

        # only apply stay penalties if enabled, if using wrappers, not enabled
        if self.use_internal_stuck_penalty:
            if self.agent_pos == self.prev_pos:
                self.stay_counter += 1
            else:
                self.stay_counter = 0

            if self.stay_counter >= 5:
                reward += -5.0
            if self.stay_counter >= 10:
                reward += -10.0
            if self.stay_counter >= 20:
                reward += -40.0

        all_cleaned = np.all(self.dirt_map == 0) # check if all tiles are cleaned
        at_start = self.agent_pos == self.start_pos # check if agent is at starting position
        terminated = bool(all_cleaned and at_start) # task is done only if all cleaned and at start
        truncated = False

        if terminated:
            reward += self.reward_return_home # terminal bonus
        elif all_cleaned and not at_start:
            reward += self.penalty_delay_return

        # End episode early if the agent is stuck or doing poorly
        if self.stay_counter >= 30:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {
            "all_cleaned": all_cleaned,
            "returned_home": at_start,
            "path_map": self.path_map
        }

    def render(self):
        """Display the environment visually"""
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

