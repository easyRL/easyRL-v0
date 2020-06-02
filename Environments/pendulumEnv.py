from Environments import classicControlEnv
import gym, random
from PIL import Image, ImageDraw
import numpy as np
import math

class PendulumEnv(classicControlEnv.ClassicControlEnv):
    displayName = 'Pendulum'

    def __init__(self):
        self.env = gym.make('Pendulum-v0')
        self.action_size = 10
        self.action_low = self.env.action_space.low[0]
        self.action_high = self.env.action_space.high[0]
        self.action_range = self.action_high - self.action_low
        self.action_tick = self.action_range/(self.action_size-1)
        self.state_size = self.env.observation_space.shape

    def step(self, action):
        action = [self.action_low + action*self.action_tick]
        return super().step(action)

    def sample_action(self):
        return random.randrange(self.action_size)

    def render(self):
        if self.env.state is None: return None

        screen_width = 500
        screen_height = 500

        state = self.env.state

        cartx = screen_width / 2.0
        carty = screen_height / 2.0
        polewidth = 10.0
        polelen = 200
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
            y -= carty+axleoffset
            x, y = x*math.cos(state[0])+y*math.sin(state[0]), -x*math.sin(state[0])+y*math.cos(state[0])
            x += cartx
            y += carty+axleoffset
            polePoints[i] = x, y
        draw.polygon(polePoints, fill=(204, 153, 102))
        draw.chord([cartx-polewidth/2, carty+axleoffset-polewidth/2, cartx+polewidth/2, carty+axleoffset+polewidth/2], 0, 360, fill=(127,127,284))
        draw.line([(0,carty), (screen_width,carty)], fill='black')

        return image.transpose(method=Image.FLIP_TOP_BOTTOM)
