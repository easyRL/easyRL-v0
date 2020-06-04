from Environments import cartPoleEnv
from PIL import Image, ImageDraw
import math

class CartPoleEnvDiscrete(cartPoleEnv.CartPoleEnv):
    displayName = 'Cart Pole Discrete'

    def __init__(self):
        super().__init__()

        self.state_size = (4,)
        self.n_bins = 3
        self.n_bins_angle = 12
        self.cart_position_range = (-2.4, 2.4)
        self.pole_angle_range = (-2, 2)
        self.cart_velocity_range = (-1, 1)
        self.angle_rate_range = (-3.5, 3.5)

    def step(self, action):
        reward = super().step(action)
        self.state = self.build_state(self.state)
        return reward

    def render(self):
        if self.env.state is None: return None

        screen_width = 600
        screen_height = 400

        state = self.env.state

        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        cartx = state[0] * scale + screen_width / 2.0
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.env.length)
        cartwidth = 50.0
        cartheight = 30.0

        image = Image.new('RGB', (screen_width, screen_height), 'white')
        draw = ImageDraw.Draw(image)
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartPoints = [(cartx + l, carty + b), (cartx + l, carty + t), (cartx + r, carty + t), (cartx + r, carty + b)]
        draw.polygon(cartPoints, fill='black')
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        t, b = t + axleoffset, b + axleoffset
        l, r, t, b = cartx + l, cartx + r, carty + t, carty + b
        polePoints = [(l, b), (l, t), (r, t), (r, b)]
        for i, (x, y) in enumerate(polePoints):
            x -= cartx
            y -= carty + axleoffset
            x, y = x * math.cos(state[2]) + y * math.sin(state[2]), -x * math.sin(state[2]) + y * math.cos(state[2])
            x += cartx
            y += carty + axleoffset
            polePoints[i] = x, y
        draw.polygon(polePoints, fill=(204, 153, 102))
        draw.chord([cartx - polewidth / 2, carty + axleoffset - polewidth / 2, cartx + polewidth / 2,
                    carty + axleoffset + polewidth / 2], 0, 360, fill=(127, 127, 284))
        draw.line([(0, carty), (screen_width, carty)], fill='black')

        return image.transpose(method=Image.FLIP_TOP_BOTTOM)

    def reset(self):
        super().reset()
        self.state = self.build_state(self.state)

    def to_bin(self, value, range, bins):
        bin = int((value-range[0]) // ((range[1] - range[0]) / bins))
        bin = max(min(bin, bins-1), 0)
        return bin

    def build_state(self, state):
        cart_position, cart_velocity, pole_angle, angle_rate_of_change = state

        new_cart_position = self.to_bin(cart_position, self.cart_position_range, self.n_bins)
        new_cart_velocity = self.to_bin(cart_velocity, self.cart_velocity_range, self.n_bins)
        new_pole_angle = self.to_bin(pole_angle, self.pole_angle_range, self.n_bins_angle)
        new_angle_rate_of_change = self.to_bin(angle_rate_of_change, self.angle_rate_range, self.n_bins)

        state = new_cart_position, new_cart_velocity, new_pole_angle, new_angle_rate_of_change

        return state
