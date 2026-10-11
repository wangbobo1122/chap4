import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math

EPISODES = 1000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 0.001
LR_CRITIC = 0.002
STATE_DIM = 3
ACTION_DIM = 3
ACTION_BOUND = {
    'dx': (0.01, 0.1),
    'dy': (0.09, 0.9),
    'dz': (0., 1.)
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_DIM),
            nn.Tanh()
        )
        scales = []
        shifts = []
        for bound in [ACTION_BOUND['dx'], ACTION_BOUND['dy'], ACTION_BOUND['dz']]:
            scale = (bound[1] - bound[0]) / 2.0
            shift = (bound[0] + bound[1]) / 2.0
            scales.append(scale)
            shifts.append(shift)
        self.scale_factors = torch.tensor(scales).float().to(device)
        self.shift_values = torch.tensor(shifts).float().to(device)

    def forward(self, state):
        raw_output = self.net(state)
        return raw_output * self.scale_factors + self.shift_values

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))

class DDPGAgent:
    def __init__(self):
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)
        self.actor_target = Actor().to(device)
        self.critic_target = Critic().to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), LR_ACTOR)
        self.critic_optim = optim.Adam(self.critic.parameters(), LR_CRITIC)
        self.memory = deque(maxlen=10000)
        self.loss_func = nn.MSELoss()

        self.action_low = np.array([ACTION_BOUND['dx'][0], ACTION_BOUND['dy'][0], ACTION_BOUND['dz'][0]])
        self.action_high = np.array([ACTION_BOUND['dx'][1], ACTION_BOUND['dy'][1], ACTION_BOUND['dz'][1]])
    def get_action(self, state, noise_scale=0.01):
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()
        return np.clip(action, self.action_low, self.action_high)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states = torch.FloatTensor(np.array([x[0] for x in batch])).to(device)
        actions = torch.FloatTensor(np.array([x[1] for x in batch])).to(device)
        rewards = torch.FloatTensor(np.array([x[2] for x in batch])).to(device)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(device)
        dones = torch.FloatTensor(np.array([x[4] for x in batch])).to(device)
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, target_actions)
            target_q = rewards + (1 - dones) * GAMMA * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

agent = DDPGAgent()
best_value = -float('inf')
best_state = np.zeros(STATE_DIM)

for episode in range(EPISODES):
    state = [np.random.uniform(0.01, 1), np.random.uniform(0.09, 0.9), np.random.uniform(0., 1.)]
    episode_reward = 0

    for step in range(200):
        action = agent.get_action(state)
        dx, dy, dz = action[0], action[1], action[2]

        next_state = [dx, dy, dz]
        reward = dz - dz*(dx/math.log(1+dx,2)+dy**2) - dz**2
        done = False if abs(sum(next_state)) < 10 else True

        agent.remember(state, action, reward, next_state, done)
        agent.learn()
        state = next_state
        episode_reward += reward

        if done:
            break
    if episode_reward > best_value:
        best_value = episode_reward
        best_state = state

    # if episode % 10 == 0:
    print(f"Episode: {episode}, Reward: {float(episode_reward):.2f}, Current state: {state}")

print(f"\nBest solution: {best_state}, Best value: {float(best_value):.3f}")
