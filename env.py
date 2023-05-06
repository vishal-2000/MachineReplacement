import numpy as np

class Env:
    def __init__(self, R) -> None:
        self.R = R
        self.state = 50
        self.max_state = 1e15

    def step(self, action):
        next_state = 0
        cost = 0
        if action==1: # Replace
            next_state = self.sample_next_state(1)
            cost = self.get_cost(1)
        elif action==2: # Don't replace
            next_state = self.sample_next_state(self.state)
            cost = self.get_cost(2)

        self.state = next_state

        return next_state, cost
    
    def get_cost(self, action):
        if action==1:
            C = self.R + 1
        elif action==2:
            C = self.state
        return -C
    
    def sample_next_state(self, state):
        next_state = np.random.poisson(state)
        # while next_state < self.max_state:
        #     next_state = np.random.poisson(state)
        return next_state


if __name__=="__main__":
    env = Env(5)
    for i in range(100):
        print(env.step(np.random.randint(1, 3)))
