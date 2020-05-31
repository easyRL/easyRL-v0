import gym
from abc import ABC

class Environment(ABC):
    displayName = 'Environment'

    def __init__(self):
        self.action_size = None
        self.state_size = None

    def step(self, action):
        pass

    def reset(self):
        pass

    def sample_action(self):
        pass

    def render(self, mode):
        pass

    def close(self):
        pass