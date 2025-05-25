import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from eval import MetricWrapper

# Register custom environment
gym.register(id="Vacuum-v0", entry_point="env:VacuumEnv")

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.zeros((out_features, in_features)))
        self.weight_sigma = nn.Parameter(torch.zeros((out_features, in_features)))
        self.register_buffer('weight_epsilon', torch.zeros((out_features, in_features)))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            return F.linear(x, 
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

class QNetwork(nn.Module):
    def __init__(self, env, n_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))
        self.n_actions = env.action_space.n
        
        # Feature extraction for dictionary observation
        self.pos_net = nn.Sequential(
            layer_init(nn.Linear(2, 64)),
            nn.ReLU()
        )
        self.orient_net = nn.Sequential(
            layer_init(nn.Linear(1, 32)),
            nn.ReLU()
        )
        self.view_net = nn.Sequential(
            layer_init(nn.Linear(3, 32)),
            nn.ReLU()
        )
        
        combined_features = 128
        
        # Advantage stream
        self.advantage_hidden = NoisyLinear(combined_features, 128)
        self.advantage = NoisyLinear(128, self.n_actions * self.n_atoms)
        
        # Value stream
        self.value_hidden = NoisyLinear(combined_features, 128)
        self.value = NoisyLinear(128, self.n_atoms)

    def get_features(self, x):
        pos_features = self.pos_net(x['agent_pos'].float())
        orient_features = self.orient_net(x['agent_orient'].float().unsqueeze(-1))
        view_features = self.view_net(x['local_view'].float())
        return torch.cat([pos_features, orient_features, view_features], dim=-1)

    def forward(self, x):
        features = self.get_features(x)
        
        advantage_hidden = F.relu(self.advantage_hidden(features))
        advantage = self.advantage(advantage_hidden).view(-1, self.n_actions, self.n_atoms)
        
        value_hidden = F.relu(self.value_hidden(features))
        value = self.value(value_hidden).view(-1, 1, self.n_atoms)
        
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(q_dist, dim=-1)

    def reset_noise(self):
        self.advantage_hidden.reset_noise()
        self.advantage.reset_noise()
        self.value_hidden.reset_noise()
        self.value.reset_noise()

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    # Hyperparameters
    exp_name = "cleanrl_rainbow_vacuum"
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    total_timesteps = 50000
    learning_rate = 3e-4
    buffer_size = int(1e5)
    gamma = 0.99
    tau = 1.0
    target_network_frequency = 100
    batch_size = 32
    start_e = 1.0
    end_e = 0.05
    exploration_fraction = 0.5
    learning_starts = 1000
    train_frequency = 1
    n_atoms = 51
    v_min = -10
    v_max = 10
    
    # Logging setup
    run_name = f"{exp_name}_{seed}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in locals().items()])),
    )

    # Environment setup
    env = gym.make("Vacuum-v0")
    env = MetricWrapper(env)
    
    q_network = QNetwork(env, n_atoms, v_min, v_max)
    target_network = QNetwork(env, n_atoms, v_min, v_max)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    
    rb = ReplayBuffer(
        buffer_size,
        env.observation_space,
        env.action_space,
        device="cpu",
        handle_timeout_termination=False,
    )
    
    start_time = time.time()
    
    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=seed)
    for global_step in range(total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_step)
        if random.random() < epsilon:
            actions = env.action_space.sample()
        else:
            q_network.reset_noise()
            dist = q_network({"agent_pos": torch.Tensor([obs["agent_pos"]]),
                            "agent_orient": torch.Tensor([obs["agent_orient"]]),
                            "local_view": torch.Tensor([obs["local_view"]])})
            actions = torch.argmax((dist * q_network.atoms).sum(-1)).item()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        done = terminated or truncated

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if done:
            print(f"global_step={global_step}, episodic_return={rewards}")
            writer.add_scalar("charts/episodic_return", rewards, global_step)
            writer.add_scalar("charts/episodic_length", 1, global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)
            obs, _ = env.reset()
            continue

        # ALGO LOGIC: training.
        if global_step > learning_starts and global_step % train_frequency == 0:
            data = rb.sample(batch_size)
            with torch.no_grad():
                q_network.reset_noise()
                target_network.reset_noise()
                
                next_dist = target_network(data.next_observations)
                next_atoms = target_network.atoms.unsqueeze(0).unsqueeze(0)
                next_val = (next_dist * next_atoms).sum(-1)
                next_actions = next_val.argmax(1)
                next_dist = next_dist[range(batch_size), next_actions]
                
                t_z = data.rewards + (1 - data.dones) * gamma * target_network.atoms
                t_z = t_z.clamp(v_min, v_max)
                b = (t_z - v_min) / ((v_max - v_min) / (n_atoms - 1))
                l = b.floor().long()
                u = b.ceil().long()
                
                target_dist = torch.zeros_like(next_dist)
                for i in range(batch_size):
                    target_dist[i].index_add_(0, l[i], next_dist[i] * (u[i].float() - b[i]))
                    target_dist[i].index_add_(0, u[i], next_dist[i] * (b[i] - l[i].float()))

            q_network.reset_noise()
            current_dist = q_network(data.observations)
            current_dist = current_dist[range(batch_size), data.actions.long()]
            
            loss = -(target_dist * current_dist.clamp(min=1e-5).log()).sum(-1).mean()

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update target network
            if global_step % target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

    writer.close()
    env.close() 