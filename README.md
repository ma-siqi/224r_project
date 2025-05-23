# CS224R Robot Vacuum Project

This project implements deep reinforcement learning algorithms (e.g., PPO, Rainbow DQN) to train a robot vacuum agent in a custom Gymnasium environment. We use CleanRL for clean, single-file reference implementations and Stable-Baselines3 for baseline comparisons.

## 🔧 Setup Instructions

1. **Create and activate the conda environment:**

```bash
conda create -n cs224r_project python=3.10 -y
conda activate cs224r_project
```
2. **Install project dependencies**

```bash
pip install -r requirements.txt
````

## Running the Agent

You can train a PPO or DQN agent by running `run.py`:

```bash
python run.py --algo ppo --grid_size 40 30 --wall_mode hardcoded
```

Command-line arguments:
- `--algo`: Choose `"ppo"` or `"dqn"` (default: `ppo`)
- `--grid_size`: Specify the environment size (e.g., `--grid_size 40 30`)
- `--wall_mode`: Either `"random"` or `"hardcoded"` (only applies to 40x30 grid)

### Examples

Train PPO with hardcoded walls (40x30):
```bash
python run.py --algo ppo --grid_size 40 30 --wall_mode hardcoded
```

Train DQN on random 20x20 layout:
```bash
python run.py --algo dqn --grid_size 20 20 --wall_mode random
```

## Hyperparameter Tuning

You can tune PPO or DQN hyperparameters using Optuna by running `tune_hyperparams.py`. After tuning, the best parameters will be saved to:

```
optuna_results/ppo_best_params.json
optuna_results/dqn_best_params.json
```

These will automatically be loaded by `run.py` when training.

## Project Structure

- `env.py`: Custom Gymnasium environment (`VacuumEnv`)
- `wrappers.py`: Reward shaping wrappers (exploration bonus, efficiency penalties)
- `run.py`: Main training and evaluation script with logging and video recording
- `tune_hyperparams.py`: Optuna-based hyperparameter search

## Dependencies
See `requirements.txt`
- Gymnasium
- Stable-Baselines3
- Optuna
- CleanRL-style architecture
- PyTorch
- Matplotlib (for visualization and animation)
