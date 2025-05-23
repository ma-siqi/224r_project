import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class MetricCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "coverage_ratio" in info:
                self.logger.record("custom/coverage_ratio", info["coverage_ratio"])
                self.logger.record("custom/path_efficiency", info["path_efficiency"])
                self.logger.record("custom/revisit_ratio", info["revisit_ratio"])
        return True

def compute_coverage_ratio(env):
    total_dirt = np.sum(env.dirt_map == 1) + np.sum(env.cleaned_map == 1)
    cleaned = np.sum(env.cleaned_map == 1)
    return cleaned / total_dirt if total_dirt > 0 else 0

def compute_redundancy_rate(env):
    path_visits = env.path_map[env.obstacle_map == 0]
    return np.mean(path_visits) if path_visits.size > 0 else 0

def compute_revisit_ratio(env):
    revisits = np.sum(env.path_map[env.path_map > 1])
    total_steps = np.sum(env.path_map)
    return (revisits - np.count_nonzero(env.path_map > 0)) / total_steps if total_steps > 0 else 0

class MetricWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.metrics = {}
        self.base_env = env.unwrapped

    def reset(self, **kwargs):
        self.metrics = {}
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            self.metrics['coverage_ratio'] = compute_coverage_ratio(self.base_env)
            self.metrics['path_efficiency'] = compute_redundancy_rate(self.base_env)
            self.metrics['revisit_ratio'] = compute_revisit_ratio(self.base_env)
            info.update(self.metrics)

        return obs, reward, terminated, truncated, info

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation):
        return self.action_space.sample(), None

def evaluate_random_agent(env, agent, episodes=1, max_steps=1000):
    all_metrics = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        base_env = env.unwrapped
        metrics = {
        "coverage_ratio": compute_coverage_ratio(base_env),
        "redundancy_rate": compute_redundancy_rate(base_env),
        "revisit_ratio": compute_revisit_ratio(base_env)
        }

        print(f"Episode {ep+1}: {metrics}")
        all_metrics.append(metrics)

    return all_metrics

