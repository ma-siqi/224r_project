# CS224R Robot Vacuum Project

This project implements deep reinforcement learning algorithms (e.g., PPO, Rainbow DQN) to train a robot vacuum agent in a custom Gymnasium environment. We use CleanRL for clean, single-file reference implementations and Stable-Baselines3 for baseline comparisons.

## 🔧 Setup Instructions

1. **Create and activate the conda environment:**

```bash
conda create -n cs224r_project python=3.10 -y
conda activate cs224r_project
```

2. **Install PyTorch with conda** (choose one based on your system):

```bash
# For CUDA support (recommended if you have a GPU):
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

3. **Install other dependencies:**

```bash
pip install -r requirements.txt
```

4. **Install ffmpeg** (required for video saving):

```bash
# On macOS:
brew install ffmpeg

# On Ubuntu:
sudo apt-get install ffmpeg

# On Windows (using conda):
conda install ffmpeg
```

## Running the Agent

### Stable-Baselines3 Implementation
You can train a PPO or DQN agent by running `run.py`:

```bash
python run.py --algo ppo --grid_size 40 30 --wall_mode hardcoded
```

Command-line arguments:
- `--algo`: Choose `"ppo"` or `"dqn"` (default: `ppo`)
- `--grid_size`: Specify the environment size (e.g., `--grid_size 40 30`)
- `--wall_mode`: Either `"random"` or `"hardcoded"` (only applies to 40x30 grid)

### Rainbow DQN Implementation
We also provide a custom Rainbow DQN implementation in `rainbow_dqn.py` with the following features:
- Distributional RL (C51)
- Dueling Networks
- Noisy Networks
- Prioritized Experience Replay
- N-step Learning
- Support for different grid sizes
- Organized logging by grid size
- Support for hardcoded wall layouts (40x30 grid)

To use Rainbow DQN:

```bash
# Train from scratch, ignoring any existing checkpoints
python rainbow_dqn.py --mode train --grid_size 40 30 --wall_mode hardcoded --timesteps 50000 --from_scratch

# Train continuing from latest checkpoint (default behavior)
python rainbow_dqn.py --mode train --grid_size 40 30 --wall_mode hardcoded --timesteps 50000

# Train on a different grid size (e.g., 8x8)
python rainbow_dqn.py --mode train --grid_size 8 8 --timesteps 50000

# Train on 40x30 grid with hardcoded wall layout
python rainbow_dqn.py --mode train --grid_size 40 30 --wall_mode hardcoded --timesteps 50000

# Continue training from latest checkpoint
python rainbow_dqn.py --mode train  # Will automatically load latest_model.pth if it exists

# Evaluate a saved model on a specific grid size
python rainbow_dqn.py --mode eval --grid_size 8 8 --model_path best_model.pth --eval_episodes 5

# Evaluate on 40x30 grid with hardcoded walls
python rainbow_dqn.py --mode eval --grid_size 40 30 --wall_mode hardcoded --model_path best_model.pth --eval_episodes 5
```

The implementation saves several model checkpoints during training:
- `best_model.pth`: Model with highest reward
- `latest_model.pth`: Most recent model state
- `model_step_X.pth`: Periodic checkpoints (every 10000 steps)
- `final_model.pth`: Model at training completion

Training metrics and videos are saved in `logs/rainbow_dqn_WxH/` where W and H are the grid dimensions:
- TensorBoard logs
- Training metrics (JSON)
- Evaluation videos
- Training plots

For example:
- `logs/rainbow_dqn_6x6/` - Logs for 6x6 grid
- `logs/rainbow_dqn_8x8/` - Logs for 8x8 grid
- `logs/rainbow_dqn_10x10/` - Logs for 10x10 grid
- `logs/rainbow_dqn_40x30/` - Logs for 40x30 grid with hardcoded walls

Each grid size gets its own directory to keep training runs organized and separate.

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
- `rainbow_dqn.py`: Custom Rainbow DQN implementation with advanced features
- `tune_hyperparams.py`: Optuna-based hyperparameter search

## Dependencies
See `requirements.txt`
- Gymnasium
- Stable-Baselines3
- Optuna
- CleanRL-style architecture
- PyTorch
- Matplotlib (for visualization and animation)
- TensorBoard (for logging)
