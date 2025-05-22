import gymnasium as gym
from gymnasium.envs.registration import register
from env import VacuumEnv
from wrappers import ExplorationBonusWrapper, ExploitationPenaltyWrapper
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

# --------------------------------------
# Register the robot vacuum environment
# --------------------------------------
register(
    id="VacuumEnv-v0",
    entry_point="env:VacuumEnv",
    kwargs={"grid_size": (20, 20), 
    "render_mode": "plot"},
)

# --------------------------------------
# Fixed wall layout generator
# --------------------------------------
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

# --------------------------------------
# Rollout and save animation
# --------------------------------------
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


if __name__ == "__main__":
    # --------------------------------------
    # PPO Training with wrappers and Monitor
    # --------------------------------------
    base_env = gym.make("VacuumEnv-v0", grid_size=(20, 20), render_mode="plot")
    base_env = ExplorationBonusWrapper(base_env, bonus=0.3)
    base_env = ExploitationPenaltyWrapper(base_env, time_penalty=-0.002, stay_penalty=-0.1)

    # monitor for stats logging
    monitored_env = Monitor(base_env)

    # evaluation environment
    eval_env = gym.make("VacuumEnv-v0", grid_size=(20, 20), render_mode="plot")
    eval_env = ExplorationBonusWrapper(eval_env, bonus=0.3)
    eval_env = ExploitationPenaltyWrapper(eval_env, time_penalty=-0.002, stay_penalty=-0.1)
    eval_env = Monitor(eval_env)

    # reset before training
    # call walls = generate_1b1b_layout_grid() to generate fixed wall layout
    walls = None
    obs, info = monitored_env.reset(options={"walls": walls})
    check_env(monitored_env, warn=True)

    # PPO agent
    model = PPO(
        "MultiInputPolicy",
        monitored_env,
        verbose=1,
        ent_coef=0.01,
        tensorboard_log="./tensorboard/",
    )

    model.learn(total_timesteps=500000)

    # Save the final trajectory
    print("Saving final training trajectory...")
    rollout_and_record(monitored_env.unwrapped, model, filename="ppo_vacuum_run.mp4", max_steps=1000)

    # Save best eval trajectory
    print("Saving best eval trajectory...")
    rollout_and_record(eval_env.unwrapped, model, filename="ppo_eval.mp4", max_steps=1000)

    # --------------------------------------
    # DQN Training with wrappers and Monitor
    # --------------------------------------
    base_env = gym.make("VacuumEnv-v0", grid_size=(20, 20), render_mode="plot")
    base_env = ExplorationBonusWrapper(base_env, bonus=0.3)
    base_env = ExploitationPenaltyWrapper(base_env, time_penalty=-0.002, stay_penalty=-0.1)
    monitored_env = Monitor(base_env)

    eval_env = gym.make("VacuumEnv-v0", grid_size=(20, 20), render_mode="plot")
    eval_env = ExplorationBonusWrapper(eval_env, bonus=0.3)
    eval_env = ExploitationPenaltyWrapper(eval_env, time_penalty=-0.002, stay_penalty=-0.1)
    eval_env = Monitor(eval_env)

    walls = None
    obs, info = monitored_env.reset(options={"walls": walls})
    check_env(monitored_env, warn=True)

    # DQN agent
    dqn_model = DQN(
        "MultiInputPolicy",
        monitored_env,
        verbose=1,
        tensorboard_log="./tensorboard_dqn/",
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
    )

    dqn_model.learn(total_timesteps=500000)

    # Save final training trajectory
    print("Saving final DQN training trajectory...")
    rollout_and_record(monitored_env.unwrapped, dqn_model, filename="dqn_vacuum_run.mp4", max_steps=1000)

    # Save evaluation trajectory
    print("Saving best DQN eval trajectory...")
    rollout_and_record(eval_env.unwrapped, dqn_model, filename="dqn_eval.mp4", max_steps=1000)