import environment
import gym
import numpy as np
import random
from PIL import Image, ImageDraw
import math

class FrozenLakeEnv(environment.Environment):

    def __init__(self):
        self.env = gym.make('FrozenLake-v0')
        self.action_size = self.env.action_space.n
        self.state_size = env.observation_space.n
        print(self.env.action_space, self.env.observation_space)
        print(self.action_size, self.state_size)

        self.state = None
        self.done = None
        self.total_rewards = None

    def step(self, action):
        new_state, reward, self.done, info = self.env.step(action)
        self.state = new_state
        return reward

    def reset(self):
        self.state = env.reset()
        self.done = False
        self.total_rewards = 0

    def sample_action(self):
        return self.env.action_space.sample()

    
