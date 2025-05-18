import gymnasium as gym
from gymnasium.envs.registration import register
from env import VacuumEnv  # your custom class
from gymnasium import spaces

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Register the environment
register(
    id="VacuumEnv-v0",
    entry_point="env:VacuumEnv",
    kwargs={"grid_size": (40, 30), 
    "render_mode": "plot"},
)

def generate_1b1b_layout_grid():
    wall_positions = set()

    for x in [5, 25, 32, 35]:
        for y in range(3):
            wall_positions.add((x, y))

    for x in range(5, 27):
        wall_positions.add((x, 12))

    wall_positions.add((25, 7))
    wall_positions.add((25, 8))
    wall_positions.add((25, 11))

    for y in range(7, 13):
        wall_positions.add((28, y))
    wall_positions.add((27, 12))
    wall_positions.add((25, 7))
    wall_positions.add((26, 7))
    wall_positions.add((27, 7))

    for y in [10, 15, 20]:
        for x in range(36, 41):
            wall_positions.add((x, y))

    wall_positions.add((32, 7))
    wall_positions.add((32, 8))
    wall_positions.add((32, 9))
    wall_positions.add((32, 10))
    wall_positions.add((33, 10))

    for x in [22, 25, 28]:
        for y in range(11, 18):
            wall_positions.add((x, y))

    for x in range(20, 28):
        wall_positions.add((x, 17))
    
    # remove the living room closest doors
    wall_positions.discard((23, 17))
    wall_positions.discard((24, 17))
    wall_positions.discard((28, 14))
    wall_positions.discard((28, 15))

    # living room and dinning room blocker
    for x in [25, 30, 31, 35]:
        for y in range(27, 31):
            wall_positions.add((x, y))

    return list(wall_positions)

def rollout_and_record(env, model, filename="vacuum_run.mp4", max_steps=100):
    obs, _ = env.reset()
    frames = []

    for _ in range(max_steps):
        fig = env.render_frame()
        frames.append(fig)

        action, _ = model.predict(obs)
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

###########################Train PPO#####################################

env = gym.make("VacuumEnv-v0", grid_size=(40, 30), render_mode="plot")

obs, info = env.reset()

# Training the PPO
check_env(env, warn=True)  # optional: validate compatibility
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=500000)
rollout_and_record(env.unwrapped, model, filename="ppo_vacuum.mp4", max_steps=10000)

###########################Train DQN#####################################
"""
walls = generate_1b1b_layout_grid()
eval_env = gym.make("VacuumEnv-v0", grid_size=(40, 30), render_mode="plot")
eval_env = Monitor(eval_env)
eval_env.reset(options={"walls": walls})

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=1000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
    verbose=1,
)

env = gym.make("VacuumEnv-v0", grid_size=(40, 30), render_mode="plot")
obs, info = env.reset(options={"walls": walls})

model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1000, callback=eval_callback)

rollout_and_record(env.unwrapped, model, filename="dqn_vacuum.mp4")
"""