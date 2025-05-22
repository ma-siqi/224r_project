import gymnasium as gym
import numpy as np

class ExplorationBonusWrapper(gym.Wrapper):
    """
    Gives a bonus reward for visiting new tiles.
    Bonus decays over time to prioritize early exploration.
    """
    def __init__(self, env, bonus=0.2, decay=0.995):
        super().__init__(env)
        # Use unwrapped env to access custom attributes
        self.grid_size = self.env.unwrapped.grid_size
        self.bonus = bonus
        self.decay = decay
        self.visit_map = np.zeros(self.grid_size, dtype=np.float32)

    def reset(self, **kwargs):
        self.visit_map *= self.decay
        obs, info = self.env.reset(**kwargs)
        pos = tuple(self.env.unwrapped.agent_pos)
        self.visit_map[pos] += 1
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = tuple(self.env.unwrapped.agent_pos)

        if self.visit_map[pos] < 1e-3:
            reward += self.bonus

        self.visit_map[pos] += 1
        return obs, reward, terminated, truncated, info


class ExploitationPenaltyWrapper(gym.Wrapper):
    """
    Penalizes inefficient behavior like:
    - Staying still
    - Taking too long after cleaning
    """
    def __init__(
        self,
        env,
        time_penalty=-0.001,
        stay_penalty=-0.1,
        stay_thresholds=(5, 10, 20),
        penalties=(-5.0, -10.0, -40.0),
        disable_internal_counter=True,
    ):
        super().__init__(env)
        self.time_penalty = time_penalty
        self.stay_penalty = stay_penalty
        self.stay_thresholds = stay_thresholds
        self.penalties = penalties

        self.prev_pos = None
        self.stay_counter = 0

        if disable_internal_counter and hasattr(env.unwrapped, "use_internal_stuck_penalty"):
            env.unwrapped.use_internal_stuck_penalty = False

    def reset(self, **kwargs):
        self.prev_pos = None
        self.stay_counter = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Penalize time usage
        reward += self.time_penalty

        # Track and penalize staying still
        curr_pos = list(self.env.unwrapped.agent_pos)
        if self.prev_pos == curr_pos:
            self.stay_counter += 1
            reward += self.stay_penalty
        else:
            self.stay_counter = 0
        self.prev_pos = curr_pos

        for threshold, penalty in zip(self.stay_thresholds, self.penalties):
            if self.stay_counter == threshold:
                reward += penalty

        return obs, reward, terminated, truncated, info