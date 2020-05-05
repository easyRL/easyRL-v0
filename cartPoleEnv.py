import environment
import gym
import pandas
import numpy as np
import matplotlib.pyplot as plt

class CartPoleEnv(environment.Environment):
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]
        print(self.env.action_space, self.env.observation_space)
        print(self.action_size, self.state_size)

        self.n_bins = 8
        self.n_bins_angle = 10
        self.cart_position_bins = pandas.cut([-2.4, 2.4], bins=self.n_bins, retbins=True)[1][1:-1]
        self.pole_angle_bins = pandas.cut([-2, 2], bins=self.n_bins_angle, retbins=True)[1][1:-1]
        self.cart_velocity_bins = pandas.cut([-1, 1], bins=self.n_bins, retbins=True)[1][1:-1]
        self.angle_rate_bins = pandas.cut([-3.5, 3.5], bins=self.n_bins_angle, retbins=True)[1][1:-1]

        self.state = None
        self.step = None
        self.done = None
        self.total_rewards = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
        self.state = self.build_state([self.to_bin(cart_position, self.cart_position_bins),
                             self.to_bin(pole_angle, self.pole_angle_bins),
                             self.to_bin(cart_velocity, self.cart_velocity_bins),
                             self.to_bin(angle_rate_of_change, self.angle_rate_bins)])
        return reward

    def reset(self):
        cart_position, pole_angle, cart_velocity, angle_rate_of_change = self.env.reset()
        self.state = self.build_state([self.to_bin(cart_position, self.cart_position_bins),
                             self.to_bin(pole_angle, self.pole_angle_bins),
                             self.to_bin(cart_velocity, self.cart_velocity_bins),
                             self.to_bin(angle_rate_of_change, self.angle_rate_bins)])

        prev_screen = self.env.render(mode='rgb_array')
        plt.imshow(prev_screen)

        self.step = 0
        self.done = False
        self.total_rewards = 0

    def sample_action(self):
        return self.env.action_space.sample()

    def to_bin(self, value, bins):
        return np.digitize(x=[value], bins=bins)[0]

    def build_state(self, features):
        return int("".join(map(lambda feature: str(int(feature)), features)))
