from abc import ABC

import modelFreeAgent


class QTable(modelFreeAgent.ModelFreeAgent, ABC):
    def __init__(self, state_size, output_size, gamma, alpha, min_epsilon, max_epsilon, decay_rate):
        super().__init__(state_size, output_size, gamma, alpha, min_epsilon, max_epsilon, decay_rate)
        self.qtable = {}

    def getQvalue(self, state, action):
        return self.qtable.get((state, action), 0.0)

    def reset(self):
        self.qtable.clear()
