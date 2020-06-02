from Environments import cartPoleEnv
from PIL import ImageDraw, ImageFont


class CartPoleEnvDiscrete(cartPoleEnv.CartPoleEnv):
    displayName = 'Cart Pole Discrete'

    def __init__(self):
        super().__init__()

        self.state_size = (1,)
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
        image = super().render()
        fnt = ImageFont.truetype('arial.ttf', 30)
        draw = ImageDraw.Draw(image)
        draw.text((50,50), str(self.state), fill='black', font=fnt)
        #draw.text((50,100), str(self.env.state), fill='black', font=fnt)
        return image

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
