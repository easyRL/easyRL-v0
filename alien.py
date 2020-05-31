import environment
import gym
from PIL import Image, ImageDraw
import math
import cv2
import numpy as np

class AlienEnv(environment.Environment):
    displayName = 'Alien'

    def __init__(self):
        self.image_width = 84
        self.image_height = 84
        self.env = gym.make("Alien-v0")
        self.action_size = self.env.action_space.n
        self.state_size = (self.image_width, self.image_height, 1) # self.env.observation_space.shape
        print(self.env.action_space, self.env.observation_space)
        print(self.action_size, self.state_size)
        self.state = None
        self.rawImg = None
        self.done = None
        self.total_rewards = None

    def step(self, action):
        observation, reward, self.done, info = self.env.step(action)
        self.state = self.preprocess(observation)
        return reward

    def reset(self):
        self.rawImg = self.env.reset()
        self.state = self.preprocess(self.rawImg)
        self.done = False
        self.total_rewards = 0

    def sample_action(self):
        return self.env.action_space.sample()

    def preprocess(self, image):
        self.rawImg = image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        return np.reshape(image, self.state_size)

    def render(self, mode='RGB'):
        return Image.fromarray(self.rawImg.astype('uint8'), 'RGB')
