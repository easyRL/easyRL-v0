import random

import qTable
import numpy as np


class QLearning(qTable.QTable):
    displayName = 'Q Learning'

    def __init__(self, *args):
        super().__init__(*args)

    def remember(self, state, action, reward, new_state, done=False):
        prevQValue = self.getQvalue(state, action)
        newQValue = self.getQvalue(new_state, self.choose_action(new_state))
        if done:
            target = reward
        else:
            target = reward + self.gamma * newQValue
        loss = target - prevQValue
        self.qtable[(state, action)] = prevQValue + self.alpha * loss
        return loss**2

    def choose_action(self, state):
        q = [self.getQvalue(state, a) for a in range(self.action_size)]
        maxQ = max(q)
        epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * self.time_steps)
        # TODO: Put epsilon at a level near this
        # if random.random() > epsilon:
        action = q.index(maxQ)
        # else:
        #     action = self.state_size.sample()
        return action

    def __deepcopy__(self, memodict={}):
        pass
