import numpy as np


class Env:
    def __init__(self, R) -> None:
        self.R = R
        self.init_state = 1
        self.state = 1
        self.n_actions = 2

    def reset(self):
        self.state = self.init_state
        return self.state

    def step(self, action):
        if action == 0:  # Replace
            next_state = self.sample_next_state(self.init_state)
            cost = self.get_reward(action)
        elif action == 1:  # Don't replace
            next_state = self.sample_next_state(self.state)
            cost = self.get_reward(action)

        self.state = next_state

        return next_state, cost

    def get_reward(self, action):
        if action == 0:
            C = self.R + 2 - 1 / self.init_state
        elif action == 1:
            C = 2 - 1 / self.state
        return -C  # Reward is negative of cost

    def sample_next_state(self, state):
        next_state = np.random.poisson(state) + self.init_state
        return next_state


if __name__ == "__main__":
    env = Env(5)
    for i in range(100):
        if i % 10:
            print(env.step(2))
        else:
            print(env.step(1))
