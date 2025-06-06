import gymnasium as gym
import numpy as np
from collections import deque

class ExplorationBonusWrapper(gym.Wrapper):
    """
    Exploration-enhancing wrapper with:
    - Bonus for visiting new tiles
    - Bonus for discovering new obstacles
    - Bonus for being adjacent to dirty/unknown tiles (frontier)
    - Bonus for meaningful rotations
    - Penalty for excessive or non-meaningful rotations
    - Penalty for getting stuck in corners
    - Bonus for escaping obstacle-adjacent positions
    - Penalty for facing obstacle and not moving
    - Logs individual reward components for debugging
    """
    def __init__(self, env, new_tile_bonus=0.2, new_obs_bonus=0.1,
                 frontier_bonus=0.5, facing_bonus_dirty=0.15, facing_bonus_unknown=0.05,
                 meaningless_turn_penalty=-0.1, stuck_penalty=-0.3,
                 escape_bonus=0.2, wall_face_penalty=-0.05, decay=0.995):
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
        self.escape_bonus = escape_bonus
        self.wall_face_penalty = wall_face_penalty
        self.decay = decay

        self.visit_map = np.zeros(self.grid_size, dtype=np.float32)
        self.known_mask = np.zeros(self.grid_size, dtype=np.uint8)
        self.action_history = deque(maxlen=5)
        self.pos_history = deque(maxlen=10)
        self.was_stuck_or_blocked = False

    def reset(self, **kwargs):
        self.visit_map *= self.decay
        self.known_mask.fill(0)
        self.action_history.clear()
        self.pos_history.clear()
        self.was_stuck_or_blocked = False

        obs, info = self.env.reset(**kwargs)
        self.known_mask |= self.env.unwrapped.known_obstacle_map
        pos = tuple(self.env.unwrapped.agent_pos)
        self.visit_map[pos] += 1
        self.pos_history.append(pos)
        return obs, info

    def step(self, action):
        known_before = np.copy(self.env.unwrapped.known_obstacle_map)
        prev_pos = tuple(self.env.unwrapped.agent_pos)

        obs, reward, terminated, truncated, info = self.env.step(action)

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
            'spin_penalty': 0.0,
            'stuck_penalty': 0.0,
            'escape_bonus': 0.0,
            'wall_face_penalty': 0.0,
            'rotate_when_blocked': 0.0
        }

        # New tile bonus
        if self.visit_map[pos] < 0.1:
            reward += self.new_tile_bonus
            info['reward_components']['new_tile_bonus'] = self.new_tile_bonus
        self.visit_map[pos] += 1

        # Obstacle discovery bonus
        newly_known = (known_before == 0) & (known_map == 1)
        obs_bonus = self.new_obs_bonus * np.sum(newly_known)
        reward += obs_bonus
        info['reward_components']['new_obs_bonus'] = float(obs_bonus)
        self.known_mask |= known_map

        # Frontier bonus
        frontier = self._frontier_bonus(pos, dirt_map, known_map)
        reward += frontier
        info['reward_components']['frontier_bonus'] = frontier

        # Rotation shaping
        rot_total, face_bonus, spin_penalty = self._rotation_shaping(action, pos, orient, dirt_map, known_map)
        reward += rot_total
        info['reward_components']['facing_bonus'] = face_bonus
        info['reward_components']['spin_penalty'] = spin_penalty

        # Stuck penalty
        self.pos_history.append(pos)
        if self.pos_history.count(pos) > 3:
            reward += self.stuck_penalty
            info['reward_components']['stuck_penalty'] = self.stuck_penalty
            self.was_stuck_or_blocked = True

        # Facing obstacle and trying to move forward
        if action == 0:
            dx, dy = self.orientations[orient]
            tx, ty = pos[0] + dx, pos[1] + dy
            if 0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]:
                if known_map[tx, ty] == 1:
                    reward += self.wall_face_penalty
                    info['reward_components']['wall_face_penalty'] = self.wall_face_penalty
                    self.was_stuck_or_blocked = True
                elif self.was_stuck_or_blocked and pos != prev_pos:
                    reward += self.escape_bonus
                    info['reward_components']['escape_bonus'] = self.escape_bonus
                    self.was_stuck_or_blocked = False

        # Bonus for rotating when stuck
        if self._obstacle_ahead() and action in [1, 2]:
            reward += 0.1
            info['reward_components']['rotate_when_blocked'] = 0.1

        self.action_history.append(action)
        return obs, reward, terminated, truncated, info

    def _frontier_bonus(self, pos, dirt_map, known_map):
        x, y = pos
        if dirt_map[x, y] == 1:
            return 0.0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    if dirt_map[nx, ny] == 1 or known_map[nx, ny] == 0:
                        return self.frontier_bonus
        return 0.0

    def _rotation_shaping(self, action, pos, orient, dirt_map, known_map):
        facing_bonus = 0.0
        spin_penalty = 0.0

        if action in [1, 2]:
            facing_bonus += self._facing_bonus(pos, orient, dirt_map, known_map)
            dx, dy = self.orientations[orient]
            tx, ty = pos[0] + dx, pos[1] + dy
            if 0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]:
                if known_map[tx, ty] == 1:
                    facing_bonus += 0.3

        if len(self.action_history) == self.action_history.maxlen:
            recent_turns = sum(1 for a in self.action_history if a in [1, 2])
            if recent_turns >= 4:
                spin_penalty = self.meaningless_turn_penalty

        return facing_bonus + spin_penalty, facing_bonus, spin_penalty

    def _facing_bonus(self, pos, orient, dirt_map, known_map):
        dx, dy = self.orientations[orient]
        tx, ty = pos[0] + dx, pos[1] + dy
        if not (0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]):
            return 0.0
        if dirt_map[tx, ty] == 1:
            return self.facing_bonus_dirty
        if known_map[tx, ty] == 0:
            return self.facing_bonus_unknown
        return 0.0

    def _obstacle_ahead(self):
        dx, dy = self.env.unwrapped.orientations[self.env.unwrapped.agent_orient]
        tx = self.env.unwrapped.agent_pos[0] + dx
        ty = self.env.unwrapped.agent_pos[1] + dy
        if not (0 <= tx < self.grid_size[0] and 0 <= ty < self.grid_size[1]):
            return True
        return self.env.unwrapped.known_obstacle_map[tx, ty] == 1
