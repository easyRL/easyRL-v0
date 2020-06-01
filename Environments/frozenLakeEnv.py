from Environments import environment
import gym
import sys
from gym import utils

class FrozenLakeEnv(environment.Environment):
    displayName = 'Frozen Lake'

    def __init__(self):
        self.env = gym.make('FrozenLake-v0')
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.n
        print(self.env.action_space, self.env.observation_space)
        print(self.action_size, self.state_size)

        self.state = None
        self.done = None
        self.total_rewards = None

    def step(self, action):
        new_state, reward, self.done, info = self.env.step(action)
        self.state = new_state
        return reward

    def reset(self):
        self.state = self.env.reset()
        self.done = False
        self.total_rewards = 0

    def sample_action(self):
        return self.env.action_space.sample()

    def render(self, mode='human'):
        # Output to console
        # self.env.render()

        # Render method from environment, change to output in window
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.env.s // self.env.ncol, self.env.s % self.env.ncol
        desc = self.env.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.env.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.env.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
