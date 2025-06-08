import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from env import VacuumEnv, WrappedVacuumEnv
from wrappers import DQNExplorationWrapper, PPOExplorationWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from run import generate_1b1b_layout_grid

import optuna
import json
import os
import numpy as np
import torch

# --------------------------------------
# Register the vacuum environment
# --------------------------------------
register(
    id="VacuumEnv-v0",
    entry_point="env:VacuumEnv",
    kwargs={"grid_size": (20, 20)},
)

# --------------------------------------
# Make monitored, wrapped environment
# --------------------------------------
def make_env(grid_size=(20, 20), max_steps=3000, dirt_num=5, algo='ppo', wall_mode=""):
    if wall_mode == "hardcoded":
        walls = generate_1b1b_layout_grid()
    elif wall_mode == "none":
        walls = None
    else:
        walls = []
    train_factory = WrappedVacuumEnv(grid_size, dirt_num, max_steps, algo=algo, walls=walls)
    train_env = DummyVecEnv([train_factory])
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)

    return train_env

# --------------------------------------
# Search spaces
# --------------------------------------
ppo_search_space = {
    "learning_rate": (1e-5, 1e-3, "log"),
    "ent_coef": (1e-4, 1e-1, "log"), # coefficient of entropy bonus: higher = more exploration, lower = more exploitation
    "n_steps": [128, 256, 512, 1024], # number of timesteps before updating policy
    "gamma": (0.90, 0.9999, "float"),
    "clip_range": (0.1, 0.3, "float"), # how far new policies are allowed to deviate from old policies
}

dqn_search_space = {
    "learning_rate": (5e-5, 5e-4, "log"), # narrowed range for more stable learning
    "buffer_size": [100_000, 200_000, 500_000, 1_000_000], # size of replay buffer
    "batch_size": [32, 64, 128], # number of samples per training step
    "exploration_fraction": (0.1, 0.4, "float"), # fraction of total training steps over which exploration decreases
    "exploration_initial_eps": (0.8, 1.0, "float"), # initial exploration probability
    "exploration_final_eps": (0.05, 0.15, "float"), # final exploration probability
    "gamma": (0.98, 0.999, "float"), # discount factor
}

# --------------------------------------
# Evaluation metric
# --------------------------------------

def evaluate_custom(model, env, n_eval_episodes=5):
    coverage_list = []
    reward_list = []

    for _ in range(n_eval_episodes):
        obs = env.reset()  # only obs returned from DummyVecEnv
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            reward = reward[0]
            done = done[0]
            episode_reward += reward

            info = info[0]

        coverage = info.get("coverage_ratio", 0.0)
        coverage_list.append(coverage)
        reward_list.append(episode_reward)

    mean_coverage = np.mean(coverage_list)
    mean_reward = np.mean(reward_list)
    return mean_coverage + 0.0005 * mean_reward

# --------------------------------------
# Objective functions
# --------------------------------------
def ppo_objective(trial):
    scores = []
    for seed in [0, 1, 2]:
        train_env = make_env(algo='ppo')
        eval_env = make_env(algo='ppo')

        train_env.seed(seed)
        eval_env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        lr_low, lr_high, lr_scale = ppo_search_space["learning_rate"]
        ent_low, ent_high, ent_scale = ppo_search_space["ent_coef"]
        gamma_low, gamma_high, _ = ppo_search_space["gamma"]
        clip_low, clip_high, _ = ppo_search_space["clip_range"]

        params = {
            "learning_rate": trial.suggest_float("learning_rate", lr_low, lr_high, log=(lr_scale == "log")),
            "ent_coef": trial.suggest_float("ent_coef", ent_low, ent_high, log=(ent_scale == "log")),
            "n_steps": trial.suggest_categorical("n_steps", ppo_search_space["n_steps"]),
            "gamma": trial.suggest_float("gamma", gamma_low, gamma_high),
            "clip_range": trial.suggest_float("clip_range", clip_low, clip_high),
        }

        model = PPO("MlpPolicy", train_env, verbose=0, **params)
        model.learn(total_timesteps=300_000)

        eval_env.ret_rms = train_env.ret_rms
        eval_env.training = False
        eval_env.norm_reward = False

        score = evaluate_custom(model, eval_env, n_eval_episodes=5)
        scores.append(score)
    return np.mean(scores)

def dqn_objective(trial):
    scores = []
    for seed in [0, 1, 2]:
        train_env = make_env(algo='dqn')
        eval_env = make_env(algo='dqn')

        train_env.seed(seed)
        eval_env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True),
            "buffer_size": trial.suggest_categorical("buffer_size", [100_000, 200_000, 500_000]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
            "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.4),
            "exploration_initial_eps": trial.suggest_float("exploration_initial_eps", 0.8, 1.0),
            "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.05, 0.15),
            "gamma": trial.suggest_float("gamma", 0.98, 0.999),
        }

        model = DQN("MlpPolicy", train_env, verbose=0, **params)
        model.learn(total_timesteps=300_000)

        # align normalization stats
        eval_env.ret_rms = train_env.ret_rms
        eval_env.training = False
        eval_env.norm_reward = False

        score = evaluate_custom(model, eval_env, n_eval_episodes=5)
        scores.append(score)

    return np.mean(scores)


# --------------------------------------
# Run tuning and retrain best
# --------------------------------------
if __name__ == "__main__":
    algo = "dqn"  # or "ppo"
    n_trials = 75

    print(f"Tuning {algo.upper()} with Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))

    if algo == "ppo":
        study.optimize(ppo_objective, n_trials=n_trials)
    else:
        study.optimize(dqn_objective, n_trials=n_trials)

    print("\nBest Trial:")
    print(study.best_trial)

    save_dir = "optuna_results"
    os.makedirs(save_dir, exist_ok=True)

    best_param_path = os.path.join(save_dir, f"{algo}_best_params.json")
    with open(best_param_path, "w") as f:
        json.dump(study.best_params, f, indent=4)

    print(f"Saved best {algo.upper()} parameters to {best_param_path}")