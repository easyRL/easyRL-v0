import modelFreeAgent

class QTable(modelFreeAgent.ModelFreeAgent):
    def __init__(self, action_size):
        super().__init__()
        self.action_size = action_size
        self.qtable = {}

    def getQvalue(self, state, action):
        return self.qtable.get((state, action), 0.0)

    def reset(self):
        self.qtable.clear()