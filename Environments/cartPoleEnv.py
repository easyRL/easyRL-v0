from Environments import classicControlEnv
import gym
from PIL import Image, ImageDraw
import math

class CartPoleEnv(classicControlEnv.ClassicControlEnv):
    displayName = 'Cart Pole'

    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape

    def render(self):
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
            y -= carty+axleoffset
            x, y = x*math.cos(state[2])+y*math.sin(state[2]), -x*math.sin(state[2])+y*math.cos(state[2])
            x += cartx
            y += carty+axleoffset
            polePoints[i] = x, y
        draw.polygon(polePoints, fill=(204, 153, 102))
        draw.chord([cartx-polewidth/2, carty+axleoffset-polewidth/2, cartx+polewidth/2, carty+axleoffset+polewidth/2], 0, 360, fill=(127,127,284))
        draw.line([(0,carty), (screen_width,carty)], fill='black')

        return image.transpose(method=Image.FLIP_TOP_BOTTOM)
