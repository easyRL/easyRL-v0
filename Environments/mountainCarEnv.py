from Environments import classicControlEnv
import gym
from PIL import Image, ImageDraw
from math import cos, sin
import numpy as np

class MountainCarEnv(classicControlEnv.ClassicControlEnv):
    displayName = 'Mountain Car'

    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape

    def step(self, action):
        return super().step(action) + 0.1*self.state[0]

    def height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def rotateTrans(self, x, y, tx, ty, ang):
        return tx + x * cos(-ang) + y * sin(-ang), ty - x * sin(-ang) + y * cos(-ang)

    def render(self):
        screen_width = 600
        screen_height = 400

        world_width = self.env.max_position - self.env.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20

        image = Image.new('RGB', (screen_width, screen_height), 'white')
        draw = ImageDraw.Draw(image)

        xs = np.linspace(self.env.min_position, self.env.max_position, 100)
        ys = self.height(xs)
        xys = list(zip((xs - self.env.min_position) * scale, ys * scale))

        draw.line(xys, fill='black')

        pos = self.env.state[0]
        carx, cary = (pos - self.env.min_position)*scale, self.height(pos)*scale
        rot = cos(3 * pos)

        clearance = 10
        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0

        x1,y1 = l, b+clearance
        x2,y2 = l, t+clearance
        x3,y3 = r,t+clearance
        x4,y4 = r,b+clearance

        draw.polygon([self.rotateTrans(x1, y1, carx, cary, rot), self.rotateTrans(x2, y2, carx, cary, rot), self.rotateTrans(x3, y3, carx, cary, rot), self.rotateTrans(x4, y4, carx, cary, rot)], fill='black')

        rad = carheight/2.5
        x1 = carwidth / 4
        y1 = clearance
        x1, y1 = self.rotateTrans(x1, y1, carx, cary, rot)
        draw.chord([x1-rad, y1-rad, x1+rad, y1+rad], 0, 360, fill=(127, 127, 127))

        rad = carheight/2.5
        x1 = -carwidth / 4
        y1 = clearance
        x1, y1 = self.rotateTrans(x1, y1, carx, cary, rot)
        draw.chord([x1-rad, y1-rad, x1+rad, y1+rad], 0, 360, fill=(127, 127, 127))

        flagx = (self.env.goal_position - self.env.min_position) * scale
        flagy1 = self.height(self.env.goal_position) * scale
        flagy2 = flagy1 + 50
        draw.line([(flagx, flagy1), (flagx, flagy2)], fill=(204, 204, 0))
        draw.polygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)], fill=(204,204,0))

        return image.transpose(method=Image.FLIP_TOP_BOTTOM)