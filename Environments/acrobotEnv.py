from Environments import classicControlEnv
import gym
from PIL import Image, ImageDraw
from math import cos, sin, pi
import numpy as np

class AcrobotEnv(classicControlEnv.ClassicControlEnv):
    displayName = 'Acrobot'

    def __init__(self):
        self.env = gym.make('Acrobot-v1')
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape

    def boundToScreen(self, x, y):
        bound = 2.2
        screen = 500
        return (x+bound)*screen/(2*bound), (y+bound)*screen/(2*bound)

    def rotateTrans(self, x, y, tx, ty, ang):
        return tx + x * cos(-ang) + y * sin(-ang), ty - x * sin(-ang) + y * cos(-ang)

    def render(self):
        if self.env.state is None: return None

        screen_width = 500
        screen_height = 500

        s = self.env.state

        p1 = [-self.env.LINK_LENGTH_1 *
              cos(s[0]), self.env.LINK_LENGTH_1 * sin(s[0])]

        p2 = [p1[0] - self.env.LINK_LENGTH_2 * cos(s[0] + s[1]),
              p1[1] + self.env.LINK_LENGTH_2 * sin(s[0] + s[1])]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
        link_lengths = [self.env.LINK_LENGTH_1, self.env.LINK_LENGTH_2]

        image = Image.new('RGB', (screen_width, screen_height), 'white')
        draw = ImageDraw.Draw(image)

        draw.line([self.boundToScreen(-2.2, 1), self.boundToScreen(2.2, 1)], fill='black')
        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, .1, -.1
            x1, y1 = self.boundToScreen(*self.rotateTrans(l, b, x, y, th))
            x2, y2 = self.boundToScreen(*self.rotateTrans(l, t, x, y, th))
            x3, y3 = self.boundToScreen(*self.rotateTrans(r, t, x, y, th))
            x4, y4 = self.boundToScreen(*self.rotateTrans(r, b, x, y, th))
            draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], fill=(0, 204, 204))
            x1,y1 = self.boundToScreen(x-0.1,y-0.1)
            x2,y2 = self.boundToScreen(x+0.1, y+0.1)
            draw.chord([x1, y1, x2, y2], 0, 360, fill=(204, 204, 0))

        return image.transpose(method=Image.FLIP_TOP_BOTTOM)
