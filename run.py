import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import TimeLimit
from gymnasium import spaces

from env import VacuumEnv
from wrappers import ExplorationBonusWrapper, ExploitationPenaltyWrapper
from her import VacuumGoalWrapper, HerReplayBufferForDQN

import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

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
# DQN Callback for logging
# --------------------------------------
class DQNLoggingCallback(BaseCallback):
    def __init__(self, verbose=1, log_freq=5000):
        super().__init__(verbose)
        self.episode_rewards = []
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        # Collect episode rewards from Monitor-wrapped env
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                reward = info["episode"]["r"]
                length = info["episode"]["l"]
                self.episode_rewards.append(reward)
                if self.verbose > 0:
                    print(f"[Episode] Reward: {reward:.2f} | Length: {length}")

        # Log every `log_freq` timesteps
        if self.num_timesteps % self.log_freq == 0:
            avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else float("nan")
            train_loss = self.model.logger.name_to_value.get("train/loss")

            # Print to terminal
            if self.verbose > 0:
                print(f"[Step {self.num_timesteps}] Avg Reward (last 10): {avg_reward:.2f} | Loss: {train_loss}")

            # Log to TensorBoard
            self.logger.record("rollout/ep_rew_mean", avg_reward)
            if train_loss is not None:
                self.logger.record("train/loss", train_loss)
            self.logger.dump(self.num_timesteps)

        return True

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
    obs, _ = env.reset(options={"walls": walls})
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
    # ------------------------------
    # Parse command-line arguments
    # ------------------------------
    parser = argparse.ArgumentParser(description="Train PPO or DQN on VacuumEnv")
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo", help="RL algorithm to run")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Number of training timesteps")
    parser.add_argument("--grid_size", type=int, nargs=2, default=[20, 20],
                        help="Grid size as two integers (e.g., 40 30)")
    parser.add_argument("--wall_mode", choices=["random", "hardcoded"], default="random",
                        help="Wall layout: 'random' or 'hardcoded' (only applies to 40x30)")
    args = parser.parse_args()

    algo = args.algo
    total_timesteps = args.timesteps
    grid_size = tuple(args.grid_size)
    wall_mode = args.wall_mode

    # Determine wall layout
    walls = generate_1b1b_layout_grid() if wall_mode == "hardcoded" and grid_size == (40, 30) else None

    # Load best hyperparameters if available
    param_path = Path(f"optuna_results/{algo}_best_params.json")
    if param_path.exists():
        with open(param_path, "r") as f:
            best_params = json.load(f)
    else:
        best_params = {}

    if algo == "ppo":
        # --------------------------------------
        # PPO Training with wrappers and Monitor
        # --------------------------------------
        max_steps = 3000

        base_env = gym.make("VacuumEnv-v0", grid_size=grid_size, render_mode="plot")
        base_env = TimeLimit(base_env, max_episode_steps=max_steps)
        base_env = ExplorationBonusWrapper(base_env, bonus=0.3)
        base_env = ExploitationPenaltyWrapper(base_env, time_penalty=-0.002, stay_penalty=-0.1)

        # monitor for stats logging
        monitored_env = Monitor(base_env)

        # evaluation environment
        eval_env = gym.make("VacuumEnv-v0", grid_size=grid_size, render_mode="plot")
        eval_env = TimeLimit(eval_env, max_episode_steps=max_steps)
        eval_env = ExplorationBonusWrapper(eval_env, bonus=0.3)
        eval_env = ExploitationPenaltyWrapper(eval_env, time_penalty=-0.002, stay_penalty=-0.1)
        eval_env = Monitor(eval_env)

        # reset before training
        obs, _ = monitored_env.reset(options={"walls": walls})
        obs, _ = eval_env.reset(options={"walls": walls})
        check_env(monitored_env, warn=True)
        check_env(eval_env, warn=True)

        # PPO agent with best hyperparameters if available
        model = PPO(
            "MultiInputPolicy",
            monitored_env,
            verbose=1,
            tensorboard_log="./tensorboard/",
            **best_params
        )

        model.learn(total_timesteps=total_timesteps)

        # Save the final trajectory
        print("Saving PPO training trajectory...")
        rollout_and_record(monitored_env.unwrapped, model, filename="ppo_train.mp4", max_steps=max_steps)

        # Save best eval trajectory
        print("Saving PPO eval trajectory...")
        rollout_and_record(eval_env.unwrapped, model, filename="ppo_eval.mp4", max_steps=max_steps)

    elif algo == "dqn":
        # --------------------------------------
        # DQN Training with wrappers and Monitor
        # --------------------------------------
        max_steps = 500

        base_env = gym.make("VacuumEnv-v0", grid_size=grid_size, render_mode="plot")
        base_env = TimeLimit(base_env, max_episode_steps=max_steps)
        base_env = ExplorationBonusWrapper(base_env, bonus=0.3)
        base_env = ExploitationPenaltyWrapper(base_env, time_penalty=-0.002, stay_penalty=-0.1)
        monitored_env = Monitor(base_env)

        eval_env = gym.make("VacuumEnv-v0", grid_size=grid_size, render_mode="plot")
        eval_env = TimeLimit(eval_env, max_episode_steps=max_steps)
        eval_env = ExplorationBonusWrapper(eval_env, bonus=0.3)
        eval_env = ExploitationPenaltyWrapper(eval_env, time_penalty=-0.002, stay_penalty=-0.1)
        eval_env = Monitor(eval_env)

        obs, _ = monitored_env.reset(options={"walls": walls})
        obs, _ = eval_env.reset(options={"walls": walls})
        check_env(monitored_env, warn=True)
        check_env(eval_env, warn=True)

        model = DQN(
            "MultiInputPolicy",
            monitored_env,
            verbose=1,
            tensorboard_log="./tensorboard/",
            exploration_final_eps=0.05,
            **best_params
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=DQNLoggingCallback(verbose=1, log_freq=10000),
            log_interval=1,
        )

        print("Saving DQN training trajectory...")
        rollout_and_record(monitored_env.unwrapped, model, filename="dqn_train.mp4", max_steps=1000)

        print("Saving DQN eval trajectory...")
        rollout_and_record(eval_env.unwrapped, model, filename="dqn_eval.mp4", max_steps=1000)
    
    elif algo == "her":
        # -----------------------------------------------
        # DQN with HER Training with wrappers and Monitor
        # -----------------------------------------------
        max_steps = 1000

        base_env = gym.make("VacuumEnv-v0", grid_size=grid_size, render_mode="plot")
        base_env = TimeLimit(base_env, max_episode_steps=max_steps)
        base_env = ExplorationBonusWrapper(base_env, bonus=0.3)
        base_env = ExploitationPenaltyWrapper(base_env, time_penalty=-0.002, stay_penalty=-0.1)

        # Wrap with Goal Wrapper for HER
        goal_env = VacuumGoalWrapper(base_env)
        monitored_env = Monitor(goal_env)

        # Eval env (same setup)
        eval_base = gym.make("VacuumEnv-v0", grid_size=grid_size, render_mode="plot")
        eval_base = TimeLimit(eval_base, max_episode_steps=max_steps)
        eval_base = ExplorationBonusWrapper(eval_base, bonus=0.3)
        eval_base = ExploitationPenaltyWrapper(eval_base, time_penalty=-0.002, stay_penalty=-0.1)
        eval_env = Monitor(VacuumGoalWrapper(eval_base))

        # Set fixed wall layout
        #walls = generate_1b1b_layout_grid()
        monitored_env.reset()
        eval_env.reset()

        # default DQN uses Double DQN already
        model = DQN(
            "MultiInputPolicy",
            monitored_env,
            buffer_size=100000,
            learning_starts=1000,
            train_freq=4,
            batch_size=64,
            target_update_interval=1000,
            gamma=0.98,
            replay_buffer_class=HerReplayBufferForDQN,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
                env=goal_env,
            ),
            verbose=1,
            tensorboard_log="./tensorboard_dqn_her/"
        )

        # Training
        model.learn(total_timesteps=total_timesteps)

        # Save training run video
        print("Saving final training trajectory...")
        rollout_and_record(monitored_env.unwrapped, model, filename="her_train.mp4", max_steps=3000)

        # Save evaluation run video
        print("Saving evaluation trajectory...")
        rollout_and_record(eval_env.unwrapped, model, filename="dqn_her_eval.mp4", max_steps=3000)
