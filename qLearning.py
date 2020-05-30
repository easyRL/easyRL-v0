import random

import qTable
import numpy as np


class QLearning(qTable.QTable):
    def __init__(self, state_size, action_size, gamma, alpha, min_epsilon, max_epsilon, decay_rate):
        super().__init__(state_size, action_size, gamma, alpha, min_epsilon, max_epsilon, decay_rate)

    def remember(self, state, action, reward, new_state):
        prevQValue = self.getQvalue(state, action)
        newQValue = self.getQvalue(new_state, self.choose_action(new_state))
        loss = reward + self.gamma * newQValue - prevQValue
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
