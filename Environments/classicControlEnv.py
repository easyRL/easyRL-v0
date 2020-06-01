from Environments import environment


class ClassicControlEnv(environment.Environment):
    displayName = 'Classic Control'

    def __init__(self):
        self.env = None
        self.state = None
        self.done = None
        self.total_rewards = None

    def step(self, action):
        observation, reward, self.done, info = self.env.step(action)
        self.state = observation
        return reward

    def reset(self):
        self.state = self.env.reset()
        self.done = False
        self.total_rewards = 0

    def sample_action(self):
        return self.env.action_space.sample()
