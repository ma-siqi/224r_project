from stable_baselines3 import DQN
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from wrappers import ExplorationBonusWrapper, ExploitationPenaltyWrapper
from gymnasium.wrappers import TimeLimit

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def rollout_and_record(env, model, filename="vacuum_run.mp4", max_steps=100):
    obs, _ = env.reset()
    frames = []

    for _ in range(max_steps):
        fig = env.render_frame()
        frames.append(fig)

        obs_input = obs

        action, _ = model.predict(obs_input, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    # Save animation
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0])
    ax.axis("off")

    def update(i):
        im.set_array(frames[i])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=100, blit=True
    )

    ani.save(filename, writer="ffmpeg")
    plt.close(fig)
    print(f"Video saved to {filename}")

class VacuumGoalWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.base_env = env.unwrapped 
        self.grid_size = self.base_env.grid_size
        self.obs_dim = 6
        self.goal_dim = self.grid_size[0] * self.grid_size[1]

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=0, high=1, shape=(self.goal_dim,), dtype=np.uint8),
            "desired_goal": spaces.Box(low=0, high=1, shape=(self.goal_dim,), dtype=np.uint8),
        })
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, _ = self.env.reset()
        return self._wrap_obs(obs), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._wrap_obs(obs), reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        return np.array(0.0 if np.array_equal(achieved_goal, desired_goal) else -1.0, dtype=np.float32)

    def _wrap_obs(self, obs):
        pos = obs["agent_pos"]
        orient = np.array([obs["agent_orient"]])
        local = obs["local_view"]
        flat_obs = np.concatenate([pos, orient, local]).astype(np.float32)
        achieved = self.base_env.cleaned_map.flatten()
        goal = (self.base_env.obstacle_map == 0).astype(np.uint8).flatten()
        return {
            "observation": flat_obs,
            "achieved_goal": achieved,
            "desired_goal": goal
        }
    
    def render_frame(self):
        return self.base_env.render_frame()

class HerReplayBufferForDQN(DictReplayBuffer):
    def __init__(self, *args, n_sampled_goal=4, goal_selection_strategy='future', env=None, her_ratio=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy
        self.env = env
        self.her_ratio = her_ratio
        if self.env is None:
            raise ValueError("HER replay buffer needs the environment for reward computation.")

    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)

    def sample(self, batch_size, env=None):
        # Sample transitions
        batch = super().sample(batch_size)

        # Apply HER - future
        for i in range(batch_size):
            if self.size() <= self.her_ratio:
                continue
            start_idx = i
            end_idx = self.size()
            if start_idx >= end_idx -1:
                continue

            future_idx = np.random.randint(start_idx + 1, end_idx)
            new_goal = self.observations["achieved_goal"][future_idx].copy()
            new_goal_tensor = torch.tensor(
                new_goal,
                dtype=batch.observations["desired_goal"].dtype,
                device=batch.observations["desired_goal"].device,
            )

            batch.observations["desired_goal"][i] = new_goal_tensor
            batch.next_observations["desired_goal"][i] = new_goal_tensor

            achieved = batch.next_observations["achieved_goal"][i].cpu().numpy()
            reward = self.env.compute_reward(achieved, new_goal, {})
            batch.rewards[i] = float(reward)

        return batch
    