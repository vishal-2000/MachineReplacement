import torch

from env import Env

NUM_EPSIODES = 20
NUM_STEPS = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_policy(env: Env, policy: torch.nn.Module):
    sum_rewards = 0
    for episode_num in range(NUM_EPSIODES):
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
    sum_rewards /= NUM_EPSIODES
    return sum_rewards


def print_policy(policy: torch.nn.Module):
    for s in range(1, 101):
        inp = torch.tensor([s], dtype=torch.float32, device=device).unsqueeze(0)
        print(policy(inp).max(1)[1].item(), end=" ")
        if s % 25 == 0:
            print()