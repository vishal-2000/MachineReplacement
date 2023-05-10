# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from env import Env
import numpy as np
import wandb
from copy import deepcopy
from icecream import ic
from tqdm import tqdm

# wandb setup
number = 1
NAME = "AC" + str(number)
ID = "AC" + str(number)
run = wandb.init(project='actorcritic_MachineReplacement', name = NAME, id = ID)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ic(device)

# %%
class PolicyNetwork(nn.Module):
    def __init__(self, n_obs, n_act):
        super(PolicyNetwork, self).__init__()

        self.layer = nn.Linear(n_obs, 16)
        self.actor = nn.Linear(16, n_act)
        self.critic = nn.Linear(16, 1)

    def forward(self, state):
        x = self.layer(state)
        x = F.relu(x)
        action_prob = F.softmax(self.actor(x), dim=-1)
        state_vals = self.critic(x)

        return action_prob, state_vals

# %%
n_obs = 1
n_act = 2

model_policy = PolicyNetwork(n_obs=n_obs, n_act=n_act).to(device)
optimizer = optim.Adam(model_policy.parameters(), lr=3e-2)
steps_done = 0

def select_action(state):
    global steps_done
    steps_done += 1
    with torch.no_grad():
        action_probs, state_val = model_policy(state)
        action_dist = torch.distributions.Categorical(action_probs)

        action = action_dist.sample()
        logprob = action_dist.log_prob(action)

        return action, logprob, state_val

# %%
max_steps_per_episode = 50
n_episodes = 400
GAMMA = 1
R = 35 # Cost of replacement of a machine

wandb.config.update({
    'max_timesteps_per_episode': max_steps_per_episode,
    'num_of_episodes': n_episodes,
    'R': R,
    'optimizer': 'Adam',
    'learning_rate': 'default',
    'n_actions': n_act,
    'n_observations': n_obs,
})

env = Env(R=R)

# %%
all_rewards = []
entropy_term = 0


for eps in range(n_episodes):
    logprobs = []
    values = []
    rewards = []

    state = env.reset()

    for steps in range(max_steps_per_episode):
        state = torch.tensor([state], dtype=torch.float32, device=device).unsqueeze(0)
        policy_dist, value = model_policy(state)
        value = value.cpu().detach().numpy()[0, 0]
        dist = policy_dist.cpu().detach().numpy()

        action = np.random.choice(n_act, p=np.squeeze(dist))
        logprob = torch.log(policy_dist.squeeze(0)[action])
        # entropy = -np.sum(np.mean(dist)*np.log(dist))
        new_state, reward = env.step(action)

        rewards.append(reward)
        values.append(value)
        logprobs.append(logprob)
        # entropy_term += entropy
        state = new_state

        if steps == max_steps_per_episode - 1:
            state = torch.tensor([state], dtype=torch.float32, device=device).unsqueeze(0)
            Qval, _ = model_policy(state)
            Qval = Qval.cpu().detach().numpy()[0, 0]
            all_rewards.append(np.sum(rewards))

    Qvals = np.zeros_like(values)
    for t in reversed(range(len(rewards))):
        Qval = rewards[t] + GAMMA*Qval
        Qvals[t] = Qval

    values = torch.FloatTensor(values)
    Qvals = torch.FloatTensor(Qvals)
    logprobs = torch.stack(logprobs)

    advantage = Qvals - values
    advantage = advantage.to(device)
    actor_loss = (-logprobs*advantage).mean()
    critic_loss = 0.5*advantage.pow(2).mean()

    # ic(actor_loss, critic_loss, entropy_term)
    # ac_loss = actor_loss + critic_loss + 0.001*entropy_term
    ac_loss = actor_loss + critic_loss

    wandb.log({'loss': ac_loss, 'Current_return': all_rewards[-1], 'n_episode': eps}) #, 'batch': t})
    optimizer.zero_grad()
    ac_loss.backward()
    optimizer.step()

    print(f"\rEpisode: {eps}\tLoss: {ac_loss}\tCurrent Discounted Return: {all_rewards[-1]}", end="")

    if eps%100 == 0:
        SAVE_PATH = './checkpoints/AC/AC_{}.pt'.format(eps)
        torch.save(model_policy.state_dict(), SAVE_PATH)

def evaluate_policy(env: Env, policy: torch.nn.Module):
    sum_rewards = 0
    for episode_num in range(n_episodes):
        episode_reward = 0
        curr_state = env.reset()
        curr_state = torch.tensor(
            [curr_state], dtype=torch.float32, device=device
        ).unsqueeze(0)
        for step_num in range(NUM_STEPS):
            action = policy(curr_state).max(1)[1].item()
            next_state, reward = env.step(action)
            episode_reward += reward
            curr_state = torch.tensor(
                [next_state], dtype=torch.float32, device=device
            ).unsqueeze(0)
        episode_reward /= NUM_STEPS
        sum_rewards += episode_reward
    sum_rewards /= n_episodes
    return sum_rewards

def print_policy(policy: torch.nn.Module):
    for s in range(1, 101):
        inp = torch.tensor([s], dtype=torch.float32, device=device).unsqueeze(0)
        print(policy(inp).max(1)[1].item(), end=" ")
        if s % 25 == 0:
            print()

# %%
NUM_STEPS = n_episodes*max_steps_per_episode

best_reward = -torch.inf
best_policy = PolicyNetwork(n_obs, n_act)
for i in tqdm(range(100, NUM_STEPS, 100), desc="Evaluating", leave=False):
    LOAD_PATH = f'./checkpoints/AC/AC_{i}.pt'
    policy_net = PolicyNetwork(n_obs, n_act).to(device)
    checkpoint = torch.load(LOAD_PATH)
    policy_net.load_state_dict(checkpoint)
    reward = evaluate_policy(Env, policy_net)
    ic("Reward (over policy)", reward, i)
    if reward > best_reward:
        best_reward = reward
        best_policy = deepcopy(policy_net)

print_policy(best_policy)

