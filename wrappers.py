import gymnasium as gym
import numpy as np
from collections import deque

class ExplorationBonusWrapper(gym.Wrapper):
    """
    Exploration-enhancing wrapper with:
    - New-tile bonus
    - Obstacle discovery bonus
    - Frontier bonus (non-decaying)
    - Facing direction bonus (toward dirt or unknown)
    - Escape bonus after being blocked
    - Rotation shaping and spin penalties
    - Logs all reward components
    """

    def __init__(self, env, new_tile_bonus=0.3, new_obs_bonus=0.1, frontier_bonus=0.5,
                 facing_bonus_dirty=0.25, facing_bonus_unknown=0.05, step_into_dirty_bonus=0.25,
                 meaningless_turn_penalty=-0.1, stuck_penalty=-0.3,
                 spin_without_move_penalty=-0.3, escape_bonus=0.1, wall_face_penalty=-0.02,
                 decay=0.995):
        super().__init__(env)
        self.grid_size = self.env.unwrapped.grid_size
        self.orientations = self.env.unwrapped.orientations

        # Reward tuning parameters
        self.new_tile_bonus = new_tile_bonus
        self.new_obs_bonus = new_obs_bonus
        self.frontier_bonus = frontier_bonus
        self.facing_bonus_dirty = facing_bonus_dirty
        self.facing_bonus_unknown = facing_bonus_unknown
        self.step_into_dirty_bonus = step_into_dirty_bonus
        self.meaningless_turn_penalty = meaningless_turn_penalty
        self.stuck_penalty = stuck_penalty
        self.spin_without_move_penalty = spin_without_move_penalty
        self.escape_bonus = escape_bonus
        self.wall_face_penalty = wall_face_penalty
        self.decay = decay  # unused now for frontier bonus

        # Internal state
        self.visit_map = np.zeros(self.grid_size, dtype=np.float32)
        self.known_mask = np.zeros(self.grid_size, dtype=np.uint8)
        self.action_history = deque(maxlen=8)
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
        known_mask_before = np.copy(self.known_mask)
        prev_pos = tuple(self.env.unwrapped.agent_pos)

        # Capture pre-step dirt status for dirty tile bonus
        pre_dirt_map = np.copy(self.env.unwrapped.dirt_map)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_known_mask()

        pos = tuple(self.env.unwrapped.agent_pos)
        orient = self.env.unwrapped.agent_orient
        dirt_map = self.env.unwrapped.dirt_map
        known_map = self.env.unwrapped.known_obstacle_map

        info['reward_components'] = {
            'base_env_reward': reward,
            'new_tile_bonus': 0.0,
            'new_obs_bonus': 0.0,
            'frontier_bonus': 0.0,
            'facing_bonus': 0.0,
            'step_into_dirty_bonus': 0.0,
            'stuck_penalty': 0.0,
            'escape_bonus': 0.0,
            'wall_face_penalty': 0.0,
            'rotate_when_blocked': 0.0,
            'spin_without_move_penalty': 0.0,
        }

        # 1. New tile bonus: encourages visiting new locations
        if self.visit_map[pos] < 0.1:
            reward += self.new_tile_bonus
            info['reward_components']['new_tile_bonus'] = self.new_tile_bonus
        self.visit_map[pos] += 1

        # 2. Obstacle discovery bonus: for revealing new obstacles
        newly_seen = (known_mask_before == 0) & (self.known_mask == 1)
        obs_bonus = self.new_obs_bonus * np.sum(newly_seen)
        reward += obs_bonus
        info['reward_components']['new_obs_bonus'] = float(obs_bonus)

        # 3. Frontier bonus (non-decaying): encourages staying near dirty or unknown areas
        frontier = self._frontier_bonus(pos, dirt_map)
        reward += frontier
        info['reward_components']['frontier_bonus'] = frontier

        # 4. Rotation shaping: rewards facing useful directions
        face_bonus = self._rotation_shaping(action, pos, orient, dirt_map, known_map, known_mask_before)
        reward += face_bonus
        info['reward_components']['facing_bonus'] = face_bonus

        # 4.5 Step into dirty tile bonus (captured BEFORE step zeroes it)
        if pre_dirt_map[pos] == 1:
            reward += self.step_into_dirty_bonus
            info['reward_components']['step_into_dirty_bonus'] = self.step_into_dirty_bonus

        # 5. Stuck penalty: penalizes staying in same position for too long
        self.pos_history.append(pos)
        if self.pos_history.count(pos) > 3:
            reward += self.stuck_penalty
            info['reward_components']['stuck_penalty'] = self.stuck_penalty
            self.was_stuck_or_blocked = True

        # 6. Escape and wall face penalty: escaping a stuck state or facing walls
        if action == 0:
            dx, dy = self.orientations[orient]
            tx, ty = pos[0] + dx, pos[1] + dy
            facing_wall_or_oob = not (0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]) or known_map[tx, ty] == 1
            if facing_wall_or_oob:
                reward += self.wall_face_penalty
                info['reward_components']['wall_face_penalty'] = self.wall_face_penalty
                self.was_stuck_or_blocked = True
            elif self.was_stuck_or_blocked and pos != prev_pos:
                capped = self.escape_bonus / (1 + 0.01 * self.timestep)
                reward += capped
                info['reward_components']['escape_bonus'] = capped
                self.was_stuck_or_blocked = False

        # 7. Rotate when blocked: small reward if turning away from obstacle
        if self._obstacle_ahead() and action in [1, 2]:
            new_orient = (self.env.unwrapped.agent_orient + (1 if action == 2 else -1)) % 8
            dx, dy = self.orientations[new_orient]
            tx, ty = pos[0] + dx, pos[1] + dy
            if 0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]:
                if known_map[tx, ty] == 0:
                    reward += 0.1
                    info['reward_components']['rotate_when_blocked'] = 0.1

        # 8. Penalize meaningless spinning without movement
        self.action_history.append(action)
        if len(self.action_history) == self.action_history.maxlen:
            recent_rotations = sum(1 for a in self.action_history if a in [1, 2])
            if recent_rotations >= 5 and pos == prev_pos:
                reward += self.spin_without_move_penalty
                info['reward_components']['spin_without_move_penalty'] = self.spin_without_move_penalty

        return obs, reward, terminated, truncated, info

    def _update_known_mask(self):
        """Update known_mask with the visible tiles around the agent."""
        x, y = self.env.unwrapped.agent_pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    self.known_mask[nx, ny] = 1

    def _frontier_bonus(self, pos, dirt_map):
        """Return frontier bonus if any adjacent tile is dirty or unknown."""
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

    def _rotation_shaping(self, action, pos, orient, dirt_map, known_map, known_mask_before):
        """Bonus for facing useful tiles after rotating."""
        facing_bonus = 0.0
        if action in [1, 2]:
            # Facing dirt or unknown
            facing_bonus += self._facing_bonus(pos, orient, dirt_map)
            # Also encourage facing *newly discovered* obstacle
            dx, dy = self.orientations[orient]
            tx, ty = pos[0] + dx, pos[1] + dy
            if 0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]:
                if known_map[tx, ty] == 1 and known_mask_before[tx, ty] == 0:
                    facing_bonus += 0.3
        return facing_bonus

    def _facing_bonus(self, pos, orient, dirt_map):
        """Bonus for facing dirt or unknown tile."""
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
        """Check if the tile ahead is a known obstacle or out of bounds."""
        dx, dy = self.env.unwrapped.orientations[self.env.unwrapped.agent_orient]
        tx = self.env.unwrapped.agent_pos[0] + dx
        ty = self.env.unwrapped.agent_pos[1] + dy
        if not (0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]):
            return True
        return self.env.unwrapped.known_obstacle_map[tx, ty] == 1