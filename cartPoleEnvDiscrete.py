import cartPoleEnv
import gym
import pandas
import numpy as np
from PIL import Image, ImageDraw
import math

class CartPoleEnvDiscrete(cartPoleEnv.CartPoleEnv):
    def __init__(self):
        super().__init__()
        self.state_size = 1
        self.n_bins = 8
        self.n_bins_angle = 10
        self.cart_position_bins = pandas.cut([-2.4, 2.4], bins=self.n_bins, retbins=True)[1][1:-1]
        self.pole_angle_bins = pandas.cut([-2, 2], bins=self.n_bins_angle, retbins=True)[1][1:-1]
        self.cart_velocity_bins = pandas.cut([-1, 1], bins=self.n_bins, retbins=True)[1][1:-1]
        self.angle_rate_bins = pandas.cut([-3.5, 3.5], bins=self.n_bins_angle, retbins=True)[1][1:-1]

    def step(self, action):
        reward = super().step(action)
        cart_position, pole_angle, cart_velocity, angle_rate_of_change = self.state
        self.state = self.build_state([self.to_bin(cart_position, self.cart_position_bins),
                             self.to_bin(pole_angle, self.pole_angle_bins),
                             self.to_bin(cart_velocity, self.cart_velocity_bins),
                             self.to_bin(angle_rate_of_change, self.angle_rate_bins)])
        return reward

    def reset(self):
        super().reset()
        cart_position, pole_angle, cart_velocity, angle_rate_of_change = self.state
        self.state = self.build_state([self.to_bin(cart_position, self.cart_position_bins),
                             self.to_bin(pole_angle, self.pole_angle_bins),
                             self.to_bin(cart_velocity, self.cart_velocity_bins),
                             self.to_bin(angle_rate_of_change, self.angle_rate_bins)])

    def to_bin(self, value, bins):
        return np.digitize(x=[value], bins=bins)[0]

    def build_state(self, features):
        return int("".join(map(lambda feature: str(int(feature)), features)))
