import environment
import gym
import pandas
import numpy as np
from PIL import Image, ImageDraw
import math

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

        #prev_screen = self.env.render(mode='rgb_array')

        self.done = False
        self.total_rewards = 0

    def sample_action(self):
        return self.env.action_space.sample()

    def render(self, mode='human'):
        if self.env.state is None: return None

        screen_width = 600
        screen_height = 400

        state = self.env.state

        world_width = self.env.x_threshold*2
        scale = screen_width/world_width
        cartx = state[0] * scale + screen_width / 2.0
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.env.length)
        cartwidth = 50.0
        cartheight = 30.0

        image = Image.new('RGB', (screen_width, screen_height), 'white')
        draw = ImageDraw.Draw(image)
        l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
        axleoffset =cartheight/4.0
        cartPoints = [(cartx + l, carty + b), (cartx + l, carty + t), (cartx + r, carty + t), (cartx + r, carty + b)]
        draw.polygon(cartPoints, fill='black')
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        t, b = t + axleoffset, b + axleoffset
        l, r, t, b = cartx + l, cartx + r, carty + t, carty + b
        polePoints = [(l,b), (l,t), (r,t), (r,b)]
        for i, (x, y) in enumerate(polePoints):
            x -= cartx
            y -= carty
            x, y = x*math.cos(state[2])+y*math.sin(state[2]), -x*math.sin(state[2])+y*math.cos(state[2])
            x += cartx
            y += carty
            polePoints[i] = x, y
        draw.polygon(polePoints, fill=(204, 153, 102))
        draw.chord([cartx-polewidth/2, carty+axleoffset-polewidth/2, cartx+polewidth/2, carty+axleoffset+polewidth/2], 0, 360, fill=(127,127,284))
        draw.line([(0,carty), (screen_width,carty)], fill='black')

        return image.transpose(method=Image.FLIP_TOP_BOTTOM)

    def to_bin(self, value, bins):
        return np.digitize(x=[value], bins=bins)[0]

    def build_state(self, features):
        return int("".join(map(lambda feature: str(int(feature)), features)))
