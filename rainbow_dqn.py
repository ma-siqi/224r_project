import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import cv2
from env import VacuumEnv

# Register custom environment
gym.register(id="Vacuum-v0", entry_point="env:VacuumEnv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
BUFFER_SIZE = int(1e5)
START_TRAINING = 1000
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
UPDATE_TARGET_EVERY = 100

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class RainbowDQN(nn.Module):
    def __init__(self, obs_dim, n_actions, atoms=51, Vmin=-10, Vmax=10):
        super().__init__()
        self.atoms = atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.support = torch.linspace(Vmin, Vmax, atoms).to(device)
        self.delta_z = (Vmax - Vmin) / (atoms - 1)

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value = nn.Linear(128, n_actions * atoms)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.feature(x)
        x = self.value(x)
        x = x.view(batch_size, -1, self.atoms)
        return torch.softmax(x, dim=2)

    def act(self, state, epsilon, action_dim):
        if random.random() < epsilon:
            return random.randint(0, action_dim - 1)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist = self(state)
            q_values = torch.sum(dist * self.support, dim=2)
            return q_values.argmax(1).item()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.buffer)

def project_distribution(next_dist, rewards, dones, gamma):
    batch_size = rewards.size(0)
    atoms = next_dist.size(2)
    support = torch.linspace(-10, 10, atoms).to(device)
    delta_z = (support[-1] - support[0]) / (atoms - 1)
    projection = torch.zeros_like(next_dist)

    for j in range(atoms):
        tz_j = rewards + gamma * support[j] * (1 - dones)
        tz_j = tz_j.clamp(support[0], support[-1])
        b_j = (tz_j - support[0]) / delta_z
        l, u = b_j.floor().long(), b_j.ceil().long()

        for i in range(batch_size):
            if l[i] == u[i]:
                projection[i, :, l[i]] += next_dist[i, :, j]
            else:
                projection[i, :, l[i]] += next_dist[i, :, j] * (u[i] - b_j[i])
                projection[i, :, u[i]] += next_dist[i, :, j] * (b_j[i] - l[i])
    return projection

def save_video(frames, filename, fps=30):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

def evaluate(policy_net, env_id="Vacuum-v0", episodes=5, render=True):
    env = gym.make(env_id, render_mode="rgb_array")
    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done, total_reward = False, 0
        frames = []

        while not done:
            action = policy_net.act(obs, epsilon=0.0, action_dim=env.action_space.n)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if render:
                frame = env.render()
                frames.append(frame)

        rewards.append(total_reward)
        if render:
            save_video(frames, f"eval_ep_{ep + 1}.mp4")
        print(f"Episode {ep + 1}: Reward = {total_reward:.2f}")

    env.close()
    return rewards

def train(env_id="Vacuum-v0", total_timesteps=50000):
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = RainbowDQN(obs_dim, n_actions).to(device)
    target_net = RainbowDQN(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    obs, _ = env.reset()
    epsilon = 1.0
    episode_reward = 0

    for step in range(total_timesteps):
        action = policy_net.act(obs, epsilon, n_actions)
        next_obs, reward, done, _, _ = env.step(action)
        buffer.push(obs, action, reward, next_obs, float(done))
        obs = next_obs
        episode_reward += reward

        if done:
            obs, _ = env.reset()
            print(f"Step {step}: Episode Reward = {episode_reward:.2f}")
            episode_reward = 0
            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        if len(buffer) >= START_TRAINING:
            batch = buffer.sample(BATCH_SIZE)
            state = torch.FloatTensor(batch.state).to(device)
            action = torch.LongTensor(batch.action).unsqueeze(1).unsqueeze(1).expand(-1, 1, policy_net.atoms).to(device)
            reward = torch.FloatTensor(batch.reward).to(device)
            next_state = torch.FloatTensor(batch.next_state).to(device)
            done = torch.FloatTensor(batch.done).to(device)

            dist = policy_net(state)
            dist = dist.gather(1, action).squeeze(1)

            with torch.no_grad():
                next_dist = target_net(next_state)
                next_action = torch.sum(next_dist * target_net.support, dim=2).argmax(1)
                next_dist = next_dist[range(BATCH_SIZE), next_action]
                target = project_distribution(next_dist.unsqueeze(1), reward, done, GAMMA)

            loss = -torch.sum(target * dist.log(), dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % UPDATE_TARGET_EVERY == 0:
                target_net.load_state_dict(policy_net.state_dict())

    env.close()
    return policy_net

if __name__ == "__main__":
    model = train()
    evaluate(model)
