import gymnasium as gym
import numpy as np
from collections import deque

class ExplorationBonusWrapper(gym.Wrapper):
    """
    Exploration-enhancing wrapper with:
    - Decaying new-tile bonus
    - Frontier bonus with decay
    - Spin penalty when rotating too much
    - Escape bonus with decay
    - Logs reward components
    """

    def __init__(self, env, new_tile_bonus=0.3, new_obs_bonus=0.1, frontier_bonus=0.5,
                 facing_bonus_dirty=0.15, facing_bonus_unknown=0.05,
                 meaningless_turn_penalty=-0.1, stuck_penalty=-0.3,
                 spin_without_move_penalty=-0.3, escape_bonus=0.2, wall_face_penalty=-0.05,
                 decay=0.995):
        super().__init__(env)
        self.grid_size = self.env.unwrapped.grid_size
        self.orientations = self.env.unwrapped.orientations

        self.new_tile_bonus = new_tile_bonus
        self.new_obs_bonus = new_obs_bonus
        self.frontier_bonus = frontier_bonus
        self.facing_bonus_dirty = facing_bonus_dirty
        self.facing_bonus_unknown = facing_bonus_unknown
        self.meaningless_turn_penalty = meaningless_turn_penalty
        self.stuck_penalty = stuck_penalty
        self.spin_without_move_penalty = spin_without_move_penalty
        self.escape_bonus = escape_bonus
        self.wall_face_penalty = wall_face_penalty
        self.decay = decay

        self.visit_map = np.zeros(self.grid_size, dtype=np.float32)
        self.known_mask = np.zeros(self.grid_size, dtype=np.uint8)
        self.action_history = deque(maxlen=6)
        self.pos_history = deque(maxlen=10)
        self.was_stuck_or_blocked = False
        self.timestep = 0

    def reset(self, **kwargs):
        self.visit_map.fill(0)
        self.known_mask.fill(0)
        self.action_history.clear()
        self.pos_history.clear()
        self.was_stuck_or_blocked = False
        self.timestep = 0

        obs, info = self.env.reset(**kwargs)
        self._update_known_mask()
        pos = tuple(self.env.unwrapped.agent_pos)
        self.visit_map[pos] += 1
        self.pos_history.append(pos)
        return obs, info

    def step(self, action):
        self.timestep += 1
        known_before = np.copy(self.known_mask)
        prev_pos = tuple(self.env.unwrapped.agent_pos)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_known_mask()

        pos = tuple(self.env.unwrapped.agent_pos)
        orient = self.env.unwrapped.agent_orient
        dirt_map = self.env.unwrapped.dirt_map
        known_map = self.env.unwrapped.known_obstacle_map  # Still useful for obstacle checks

        info['reward_components'] = {
            'base_env_reward': reward,
            'new_tile_bonus': 0.0,
            'new_obs_bonus': 0.0,
            'frontier_bonus': 0.0,
            'facing_bonus': 0.0,
            'spin_penalty': 0.0,
            'stuck_penalty': 0.0,
            'escape_bonus': 0.0,
            'wall_face_penalty': 0.0,
            'rotate_when_blocked': 0.0,
            'spin_without_move_penalty': 0.0,
        }

        # 1. New tile bonus: different from clean a new tile reward; encourages exploration
        if self.visit_map[pos] < 0.1:
            reward += self.new_tile_bonus
            info['reward_components']['new_tile_bonus'] = self.new_tile_bonus
        self.visit_map[pos] += 1

        # 2. Obstacle discovery (newly seen tiles): encourages discovering previously unknwon obstacles
        newly_seen = (known_before == 0) & (self.known_mask == 1)
        obs_bonus = self.new_obs_bonus * np.sum(newly_seen)
        reward += obs_bonus
        info['reward_components']['new_obs_bonus'] = float(obs_bonus)

        # 3. Frontier bonus with decay: encourages exploring dirty or unexplored borders
        decay_factor = self.decay ** self.visit_map[pos]
        frontier = self._frontier_bonus(pos, dirt_map) * decay_factor
        reward += frontier
        info['reward_components']['frontier_bonus'] = frontier

        # 4. Rotation shaping
        rot_total, face_bonus, spin_penalty = self._rotation_shaping(action, pos, orient, dirt_map, known_map)
        reward += rot_total
        info['reward_components']['facing_bonus'] = face_bonus
        info['reward_components']['spin_penalty'] = spin_penalty

        # 5. Stuck penalty
        self.pos_history.append(pos)
        if self.pos_history.count(pos) > 3:
            reward += self.stuck_penalty
            info['reward_components']['stuck_penalty'] = self.stuck_penalty
            self.was_stuck_or_blocked = True

        # 6. Escape or wall face penalty
        if action == 0:
            dx, dy = self.orientations[orient]
            tx, ty = pos[0] + dx, pos[1] + dy
            if 0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]:
                if known_map[tx, ty] == 1:
                    reward += self.wall_face_penalty
                    info['reward_components']['wall_face_penalty'] = self.wall_face_penalty
                    self.was_stuck_or_blocked = True
                elif self.was_stuck_or_blocked and pos != prev_pos:
                    capped = self.escape_bonus / (1 + 0.01 * self.timestep)
                    reward += capped
                    info['reward_components']['escape_bonus'] = capped
                    self.was_stuck_or_blocked = False

        # 7. Rotate when blocked
        if self._obstacle_ahead() and action in [1, 2]:
            reward += 0.1
            info['reward_components']['rotate_when_blocked'] = 0.1

        # 8. Penalize rotation without movement
        self.action_history.append(action)
        if len(self.action_history) == self.action_history.maxlen:
            if all(a in [1, 2] for a in self.action_history) and prev_pos == pos:
                reward += self.spin_without_move_penalty
                info['reward_components']['spin_without_move_penalty'] = self.spin_without_move_penalty

        return obs, reward, terminated, truncated, info

    def _update_known_mask(self):
        """Update known_mask with all visible tiles around the agent."""
        x, y = self.env.unwrapped.agent_pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    self.known_mask[nx, ny] = 1

    def _frontier_bonus(self, pos, dirt_map):
        x, y = pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    if dirt_map[nx, ny] == 1 or self.known_mask[nx, ny] == 0:
                        return self.frontier_bonus
        return 0.0

    def _rotation_shaping(self, action, pos, orient, dirt_map, known_map):
        facing_bonus = 0.0
        spin_penalty = 0.0
        if action in [1, 2]:
            facing_bonus += self._facing_bonus(pos, orient, dirt_map)
            dx, dy = self.orientations[orient]
            tx, ty = pos[0] + dx, pos[1] + dy
            if 0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]:
                if known_map[tx, ty] == 1:
                    facing_bonus += 0.3

        if len(self.action_history) == self.action_history.maxlen:
            if sum(1 for a in self.action_history if a in [1, 2]) >= 4:
                spin_penalty = self.meaningless_turn_penalty

        return facing_bonus + spin_penalty, facing_bonus, spin_penalty

    def _facing_bonus(self, pos, orient, dirt_map):
        dx, dy = self.orientations[orient]
        tx, ty = pos[0] + dx, pos[1] + dy
        if not (0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]):
            return 0.0
        if dirt_map[tx, ty] == 1:
            return self.facing_bonus_dirty
        if self.known_mask[tx, ty] == 0:
            return self.facing_bonus_unknown
        return 0.0

    def _obstacle_ahead(self):
        dx, dy = self.env.unwrapped.orientations[self.env.unwrapped.agent_orient]
        tx = self.env.unwrapped.agent_pos[0] + dx
        ty = self.env.unwrapped.agent_pos[1] + dy
        if not (0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]):
            return True
        return self.env.unwrapped.known_obstacle_map[tx, ty] == 1

