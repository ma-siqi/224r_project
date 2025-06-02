import optuna
import os
import torch
import numpy as np
from rainbow_dqn import train, evaluate, RainbowDQN, get_log_dir
import json
from datetime import datetime
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers import RecordEpisodeStatistics
from env import VacuumEnv
from wrappers import ExplorationBonusWrapper, ExploitationPenaltyWrapper
from run import generate_1b1b_layout_grid

# Base directory for all Optuna studies
OPTUNA_STUDIES_DIR = os.path.join('logs', 'optuna_studies')
os.makedirs(OPTUNA_STUDIES_DIR, exist_ok=True)

# Register the environment
gym.register(id="Vacuum-v0", entry_point="env:VacuumEnv")

def make_env(grid_size=(40, 30), use_layout=True, max_steps=3000, dirty_ratio=0.9):
    """Create a wrapped vacuum environment with specified parameters.
    
    Args:
        grid_size (tuple): Size of the grid (width, height)
        use_layout (bool): Whether to use predefined layout (only for 40x30)
        max_steps (int): Maximum steps per episode
        dirty_ratio (float): Initial ratio of dirty cells
    
    Returns:
        callable: A function that creates and returns the wrapped environment
    """
    walls = generate_1b1b_layout_grid() if use_layout and grid_size == (40, 30) else None
    
    def _env():
        env = gym.make("Vacuum-v0", 
                      grid_size=grid_size, 
                      render_mode="plot",
                      dirty_ratio=dirty_ratio)
        env = TimeLimit(env, max_episode_steps=max_steps)
        env = ExplorationBonusWrapper(env, bonus=0.3)  # Will be overridden by trial params
        env = ExploitationPenaltyWrapper(env, time_penalty=-0.002, stay_penalty=-0.1)  # Will be overridden
        env = RecordEpisodeStatistics(env)
        if walls is not None:
            env.reset(options={"walls": walls})
        return env
    
    return _env

def objective(trial):
    # Create study directory for this trial
    study_name = trial.study.study_name
    study_dir = os.path.join(OPTUNA_STUDIES_DIR, study_name)
    trial_dir = os.path.join(study_dir, f'trial_{trial.number}')
    os.makedirs(trial_dir, exist_ok=True)
    
    # Hyperparameters to tune
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_int('batch_size', 16, 128, step=16),
        'gamma': trial.suggest_float('gamma', 0.9, 0.99),
        'n_atoms': trial.suggest_int('n_atoms', 21, 71, step=10),
        'v_min': trial.suggest_float('v_min', -20, -5),
        'v_max': trial.suggest_float('v_max', 5, 20),
        'n_step': trial.suggest_int('n_step', 1, 5),
        'alpha': trial.suggest_float('alpha', 0.4, 0.8),
        'beta': trial.suggest_float('beta', 0.3, 0.7),
        'beta_increment': trial.suggest_float('beta_increment', 0.0005, 0.002),
        'exploration_bonus': trial.suggest_float('exploration_bonus', 0.1, 0.5),
        'time_penalty': trial.suggest_float('time_penalty', -0.005, -0.001),
        'stay_penalty': trial.suggest_float('stay_penalty', -0.2, -0.05)
    }
    
    # Save hyperparameters
    with open(os.path.join(trial_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    try:
        # Override global variables in rainbow_dqn.py
        import rainbow_dqn as rdqn
        rdqn.LR = params['lr']
        rdqn.BATCH_SIZE = params['batch_size']
        rdqn.GAMMA = params['gamma']
        rdqn.N_ATOMS = params['n_atoms']
        rdqn.V_MIN = params['v_min']
        rdqn.V_MAX = params['v_max']
        rdqn.N_STEP = params['n_step']
        rdqn.ALPHA = params['alpha']
        rdqn.BETA = params['beta']
        rdqn.BETA_INCREMENT = params['beta_increment']
        
        # Create training and evaluation environments
        grid_size = (40, 30)  # Match tune_hyperparams.py
        total_timesteps = 100000  # Match tune_hyperparams.py
        eval_episodes = 5  # Match tune_hyperparams.py
        
        env_fn = make_env(
            grid_size=grid_size,
            use_layout=True,
            max_steps=3000,  # Match tune_hyperparams.py
            dirty_ratio=0.9
        )
        
        # Create training and eval environments
        train_env = env_fn()
        eval_env = env_fn()  # Separate env for evaluation
        
        # Update environment wrappers with trial-specific parameters
        train_env.env.exploration_bonus = params['exploration_bonus']  # type: ignore
        train_env.env.time_penalty = params['time_penalty']  # type: ignore
        train_env.env.stay_penalty = params['stay_penalty']  # type: ignore
        
        eval_env.env.exploration_bonus = params['exploration_bonus']  # type: ignore
        eval_env.env.time_penalty = params['time_penalty']  # type: ignore
        eval_env.env.stay_penalty = params['stay_penalty']  # type: ignore
        
        # Create training directory
        train_dir = os.path.join(trial_dir, 'training')
        os.makedirs(train_dir, exist_ok=True)
        
        # Train the model with custom log directory
        model = train(
            env=train_env,
            grid_size=grid_size,
            total_timesteps=total_timesteps,
            save_freq=total_timesteps,  # Only save at the end
            from_scratch=True,  # Always start fresh
            custom_log_dir=train_dir  # Use trial-specific directory
        )
        
        # Create evaluation directory
        eval_dir = os.path.join(trial_dir, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        
        # Evaluate the model with custom log directory
        metrics = evaluate(
            model,
            env=eval_env,
            grid_size=grid_size,
            episodes=eval_episodes,
            render=False,  # No need to render during tuning
            custom_log_dir=eval_dir  # Use trial-specific directory
        )
        
        # Calculate objective metric (you can modify this)
        avg_reward = np.mean(metrics['episode_rewards'])
        avg_coverage = np.mean(metrics['coverage_ratio'])
        avg_efficiency = np.mean(metrics['path_efficiency'])
        
        # Combine metrics into a single objective value
        objective_value = avg_reward * avg_coverage * avg_efficiency
        
        # Save trial results
        results = {
            'avg_reward': float(avg_reward),
            'avg_coverage': float(avg_coverage),
            'avg_efficiency': float(avg_efficiency),
            'objective_value': float(objective_value)
        }
        with open(os.path.join(trial_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Clean up environments
        train_env.close()
        eval_env.close()
        
        return objective_value
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        import traceback
        traceback.print_exc()  # Print the full traceback for debugging
        return float('-inf')  # Return worst possible value on failure

def main():
    # Create study name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    study_name = f"rainbow_dqn_study_{timestamp}"
    study_dir = os.path.join(OPTUNA_STUDIES_DIR, study_name)
    os.makedirs(study_dir, exist_ok=True)
    
    # Create database in the study directory
    storage_name = f"sqlite:///{os.path.join(study_dir, 'study.db')}"
    
    # Create and run study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    n_trials = 50  # Adjust based on your computational resources
    study.optimize(objective, n_trials=n_trials)
    
    # Print best trial information
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("\nBest hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    
    # Save study results in the study directory
    study_results = {
        'best_value': trial.value,
        'best_params': trial.params,
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params
            }
            for t in study.trials
        ]
    }
    
    with open(os.path.join(study_dir, 'study_results.json'), 'w') as f:
        json.dump(study_results, f, indent=4)
    
    # Create visualization plots in the study directory
    try:
        # Plot optimization history
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_html(os.path.join(study_dir, 'optimization_history.html'))
        
        # Plot parameter importances
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_html(os.path.join(study_dir, 'param_importances.html'))
        
        # Plot parallel coordinate
        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.write_html(os.path.join(study_dir, 'parallel_coordinate.html'))
    except Exception as e:
        print(f"Warning: Could not create some visualizations: {str(e)}")

if __name__ == "__main__":
    main() 