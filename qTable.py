from abc import ABC

import modelFreeAgent


class QTable(modelFreeAgent.ModelFreeAgent, ABC):
    def __init__(self, action_size, learning_rate, gamma):
        super().__init__()
        self.action_size = action_size
        self.qtable = {}
        self.learning_rate = learning_rate
        self.gamma = gamma

    def getQvalue(self, state, action):
        return self.qtable.get((state, action), 0.0)

    def reset(self):
        self.qtable.clear()
