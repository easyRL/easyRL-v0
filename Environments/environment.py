import gym
from abc import ABC

class Environment(ABC):
    displayName = 'Environment'

    def __init__(self):
        self.action_size = None
        self.state_size = None
        self.state = None
        self.done = None

    def step(self, action):
        pass

    def reset(self):
        pass

    def sample_action(self):
        pass

    def render(self):
        pass

    def close(self):
        pass