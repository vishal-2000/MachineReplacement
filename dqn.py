# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
from env import Env
import numpy as np
import wandb
import copy
from tqdm import tqdm
from copy import deepcopy

number = 2
NAME = "DQN_RL" + str(number)
ID = "DQN_RL" + str(number)
run = wandb.init(project="DQN_RL", name=NAME, id=ID)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="DQN_RL"
# )


# %%
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "next_action")
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# %%
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# %%
torch.cuda.empty_cache()
env = Env(R=35)


# %%
BATCH_SIZE = 512
GAMMA = 0.7  # 1 # 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
REPLAY_MEMORY_SIZE = 10000

n_actions = env.n_actions
state = env.reset()
n_observations = 1

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(REPLAY_MEMORY_SIZE)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[np.random.randint(0, 2)]], device=device, dtype=torch.long
        )


# %%
def optimize_model(timestep=0, batch_num=0, reward=0):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    next_action_batch = torch.cat(batch.next_action)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    wandb.log({"loss": loss, "timestep": timestep, "batch": batch_num})
    wandb.log({"loss": loss, "reward": reward, "timestep": timestep})  # , 'batch': t})

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss


# %%

num_episodes = 100
num_time_per_episode = 500

wandb.config.update(
    {
        "max_timesteps": num_episodes * num_time_per_episode,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": "default",
        "replay_memory": REPLAY_MEMORY_SIZE,  # 10000
        "n_actions": n_actions,
        "n_observations": n_observations,
    }
)


# %%

loss_rec = []
reward_rec = []
timestep_rec = []
for i in range(num_episodes):
    prev_state = 0
    prev_action = 0
    prev_reward = 0
    cur_state = env.reset()
    for j in range(num_time_per_episode):
        cur_state = torch.tensor(
            [cur_state], dtype=torch.float32, device=device
        ).unsqueeze(0)
        cur_action = select_action(cur_state)
        # print(cur_action.item())
        next_state, reward = env.step(cur_action.item())

        reward = torch.tensor([reward], device=device)

        if j > 0:
            memory.push(prev_state, prev_action, prev_reward, cur_state, cur_action)

        prev_state = copy.deepcopy(cur_state)
        prev_action = copy.deepcopy(cur_action)
        prev_reward = copy.deepcopy(reward)

        loss = optimize_model(
            timestep=(i * num_time_per_episode) + j, reward=reward.item()
        )
        print(f"\rTrain itr: {i}:{j}", end="")
        # print((i*num_time_per_episode) + j, loss, reward.item())
        loss_rec.append(loss)
        timestep_rec.append((i * num_time_per_episode) + j)
        reward_rec.append(reward.item())

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if ((i * num_time_per_episode) + j) % 100 == 0:
            SAVE_PATH = "./checkpoints/DQN_{}.pt".format((i * num_time_per_episode) + j)
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(policy_net.state_dict(), SAVE_PATH)

        torch.cuda.empty_cache()


# %%
# import matplotlib.pyplot as plt
# plt.plot(timestep_rec, loss)


def evaluate_policy(env: Env, policy: torch.nn.Module):
    sum_rewards = 0
    for episode_num in range(num_episodes):
        episode_reward = 0
        curr_state = env.reset()
        curr_state = torch.tensor(
            [curr_state], dtype=torch.float32, device=device
        ).unsqueeze(0)
        for step_num in range(num_time_per_episode):
            action = policy(curr_state).max(1)[1].item()
            next_state, reward = env.step(action)
            episode_reward += reward
            curr_state = torch.tensor(
                [next_state], dtype=torch.float32, device=device
            ).unsqueeze(0)
        episode_reward /= num_time_per_episode
        sum_rewards += episode_reward
    sum_rewards /= num_episodes
    return sum_rewards


def print_policy(policy: torch.nn.Module):
    for s in range(1, 101):
        inp = torch.tensor([s], dtype=torch.float32, device=device).unsqueeze(0)
        print(policy(inp).max(1)[1].item(), end=" ")
        if s % 25 == 0:
            print()


best_reward = -torch.inf
best_policy = DQN(n_observations, n_actions)
for i in tqdm(range(100, num_time_per_episode, 100), desc="Evaluating", leave=False):
    LOAD_PATH = f"./checkpoints/DQN_{i}.pt"
    policy_net = DQN(n_observations, n_actions).to(device)
    checkpoint = torch.load(LOAD_PATH)
    policy_net.load_state_dict(checkpoint)
    reward = evaluate_policy(env, policy_net)
    print("Reward (over policy)", reward, i)
    if reward > best_reward:
        best_reward = reward
        best_policy = deepcopy(policy_net)

print_policy(best_policy)
# %%
