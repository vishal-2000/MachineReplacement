# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from env import Env
import numpy as np
import wandb
import copy

# wandb setup
number = 4
NAME = "Reinforce" + str(number)
ID = "Reinforce" + str(number)
run = wandb.init(project='REINFORCE_MachineReplacement', name = NAME, id = ID)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, n_actions)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        output = F.softmax(self.layer3(x), dim=1)
        return output

# %%
n_actions = 2
n_observations = 1

policy_net = PolicyNetwork(n_observations=n_observations, n_actions=n_actions).to(device)
optimizer = optim.Adam(policy_net.parameters())
steps_done = 0

def select_action(state):
    global steps_done
    steps_done += 1
    with torch.no_grad():
        action_probs = policy_net(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob

# %%
max_steps_per_episode = 500
n_episodes = 10000
GAMMA = 0.99
R = 35 # Cost of replacement of a machine

wandb.config.update({
    'max_timesteps_per_episode': max_steps_per_episode,
    'num_of_episodes': n_episodes,
    'R': R,
    'optimizer': 'Adam',
    'learning_rate': 'default',
    'n_actions': n_actions,
    'n_observations': n_observations,
})

env = Env(R=R)

# REINFORCE
all_rewards = []
for i in range(n_episodes):
    episode_rewards = []
    episode_log_probs = []
    state = env.reset()
    for j in range(max_steps_per_episode):
        state = torch.tensor([state], dtype=torch.float32, device=device).unsqueeze(0)
        # action, log_prob = select_action(state)
        action_probs = policy_net(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        next_state, reward = env.step(action)
        episode_rewards.append(reward)
        episode_log_probs.append(log_prob)
        state = next_state
    all_rewards.append(sum(episode_rewards))
    discounted_rewards = []
    total_cur_return = 0
    for t in range(len(episode_rewards)):
        Gt = sum([GAMMA**(k-t-1)*episode_rewards[k] for k in range(t, len(episode_rewards))])
        discounted_rewards.append(Gt)
        if t==0:
            total_cur_return = copy.deepcopy(Gt)
    discounted_rewards = torch.tensor(discounted_rewards)
    log_probs = torch.stack(episode_log_probs)
    loss = -(log_probs*discounted_rewards).mean()

    # print(f'Log of probs: {log_probs[0:10]}')
    # print(f'Discounted returns: {discounted_rewards[0:10]}')

    wandb.log({'loss': loss, 'Current_return': total_cur_return, 'n_episode': i}) #, 'batch': t})
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"\rEpisode: {i}\tLoss: {loss}\tCurrent Discounted Return: {total_cur_return}", end="")

    if i%100 == 0:
        SAVE_PATH = './checkpoints/REINFORCE/REINFORCE_{}.pt'.format(i)
        torch.save(policy_net.state_dict(), SAVE_PATH)
    

# %%
